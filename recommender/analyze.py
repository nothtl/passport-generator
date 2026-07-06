"""Goal-aware recommender analysis pipeline."""
from __future__ import annotations

import argparse
import os
import sys
from statistics import median

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from recommender.extract.section_parser import parse_resume_sections
from recommender.extract.skill_extractor import extract_skill_profile
from recommender.llm import LLMConfig, LLMOrchestrator
from recommender.llm.provider import load_openrouter_api_key
from recommender.match.ensemble_matcher import match_role as _match_role
from recommender.match.subdomains import choose_subdomain, subdomains_for_domain
from recommender.match.student_intent import build_student_intent_profile
from recommender.rank.job_ranker import build_gap_summary, rank_jobs
from recommender.retrieve.retriever import (
    filter_job_records,
    get_jd_skill_vocabulary,
    get_related_skills,
    retrieve_jds,
    retrieve_from_subset,
)
from recommender.retrieve.eligibility import screen_jobs, filter_ineligible

_JUNK_SKILLS = {
    "and on",
    "at a",
    "can",
    "gmail",
    "home",
    "http",
    "https",
    "i am",
    "is a",
    "linkedin",
    "local",
    "make",
    "mar",
    "part",
    "pdf",
    "phone",
    "present",
    "united states",
}
_DEFAULT_LLM_CACHE_DIR = os.path.join(_PROJECT_DIR, "recommender", ".cache", "llm")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# ── Tuning constants (named to replace magic numbers) ──
CONFIDENCE_BOOST = 40           # Added to raw classifier % for user-facing confidence
FIT_BOOST = 30                  # Added to raw IDF fit scores
FIT_READY_BOOST = 5             # Extra boost for ready_now tier
FIT_ASPIRE_PENALTY = 5          # Penalty for aspirational tier
ROUTE_LOW_CONF = 38             # Trigger LLM route judge below this confidence
ROUTE_TOP_GAP = 12              # Trigger LLM route judge when top-2 gap <= this
ROUTE_INTENT_MIN = 0.45         # Minimum intent confidence to consider goal conflict
EVIDENCE_SKILL_MAX = 50         # Trigger evidence judge when extracted > this many skills
EVIDENCE_JUNK_RATIO = 0.08      # Trigger evidence judge when junk ratio > this
EVIDENCE_MARKET_OVERLAP = 0.30  # Trigger evidence judge when market overlap < this
SPARSE_SKILL_THRESHOLD = 30     # Consider resume sparse below this many extracted skills
NARROW_POOLS = {"legal", "engineering", "science", "agriculture", "personal-care"}


def _default_llm_mode() -> str:
    env_mode = os.getenv("RECOMMENDER_DEFAULT_LLM_MODE", "").strip().lower()
    if env_mode in {"off", "hybrid", "force"}:
        return env_mode
    return "hybrid" if load_openrouter_api_key() else "off"


def _with_default_cache(config: LLMConfig) -> LLMConfig:
    if config.cache_dir:
        return config
    return LLMConfig(
        mode=config.mode,
        provider_name=config.provider_name,
        route_model=config.route_model,
        evidence_model=config.evidence_model,
        jobs_model=config.jobs_model,
        max_calls=config.max_calls,
        debug=config.debug,
        cache_dir=_DEFAULT_LLM_CACHE_DIR,
    )


def _build_candidate_jobs(
    candidate_functions: list[str],
    student_skills: list[str],
    student_intent: dict,
    chosen_subdomain: str,
    top_k: int,
) -> list[dict]:
    # Use subset parquet if available, fall back to per-function parquets.
    # Only pull from the chosen function + secondary (not all alternatives).
    # For sparse resumes or functions with narrow pools, also pull from broad fallback pools.
    primary_function = candidate_functions[0] if candidate_functions else ""
    secondary = candidate_functions[1] if len(candidate_functions) > 1 else ""
    pull_functions = [primary_function] + ([secondary] if secondary else [])

    # Only expand to broad pools when the primary pool has very few jobs
    if primary_function in NARROW_POOLS and len(pull_functions) <= 2:
        primary_jobs = retrieve_jds(primary_function, "Entry", student_skills, top_k=5)
        if len(primary_jobs) < 3:  # Only expand if primary pool is truly thin
            for broad in ["support", "ops", "education", "healthcare"]:
                if broad not in pull_functions:
                    pull_functions.append(broad)
    # Try per-function parquets first (better data), subset as fallback
    combined: list[dict] = []
    seen = set()
    for function in pull_functions:
        rows = retrieve_jds(function, "Entry", student_skills, top_k=max(8, top_k * 3))
        for row in rows:
            key = row.get("id") or row.get("url") or f"{row.get('title')}::{row.get('company')}"
            if key in seen:
                continue
            seen.add(key)
            combined.append(row)

    # Fall back to subset for thin functions or gaps
    if len(combined) < top_k * 2:
        subset_jobs = retrieve_from_subset(
            function=primary_function,
            student_skills=student_skills,
            top_k=max(12, top_k * 3),
            function_labels=pull_functions if len(pull_functions) > 1 else None,
        )
        for row in subset_jobs:
            key = row.get("id") or row.get("url") or f"{row.get('title')}::{row.get('company')}"
            if key not in seen:
                seen.add(key)
                combined.append(row)

    filtered = filter_job_records(
        combined,
        candidate_function=candidate_functions[0] if candidate_functions else "",
        goal_domains=student_intent.get("goal_domains", []),
        study_domains=student_intent.get("study_domains", []),
        goal_subdomains=student_intent.get("goal_subdomains", []),
        study_subdomains=student_intent.get("study_subdomains", []),
        study_level=student_intent.get("study_level", ""),
    )
    # Screen for youth eligibility
    filtered = filter_ineligible(filtered)
    return filtered


def _auto_explain(title: str, function: str, subdomain: str, skills: list[str], goals: list[str]) -> str:
    """Generate a contextual explanation for why this job fits the student."""
    if not skills:
        return f"This {function.replace('-',' ')} role matches your career direction."
    top_skills = [s for s in skills[:6] if len(s) > 2 and s not in ('can','make','part','home')][:3]
    if goals:
        goal_str = goals[0]
        return f"This role builds your {', '.join(top_skills)} skills and moves you toward becoming a {goal_str}."
    return f"A {function.replace('-',' ')} role that uses your {', '.join(top_skills)} skills."


def _default_coach_notes(function: str, core_gaps: list[str], verify_gaps: list[str]) -> str:
    priorities = core_gaps[:3] or verify_gaps[:2]
    if not priorities:
        return f"Your profile already shows relevant signals for {function}. Keep building direct project or internship evidence."
    return (
        f"For a {function} path, focus first on {', '.join(priorities)}. "
        f"Build direct evidence for those skills through coursework, projects, internships, or volunteer work."
    )


def _coerce_llm_config(llm_config: LLMConfig | dict | None) -> LLMConfig:
    if isinstance(llm_config, LLMConfig):
        return _with_default_cache(llm_config)
    if isinstance(llm_config, dict):
        return _with_default_cache(LLMConfig(**llm_config))
    return _with_default_cache(LLMConfig(mode=_default_llm_mode()))


def _build_function_pool(best: dict, student_intent: dict, resume_text: str = "") -> list[str]:
    ordered: list[str] = []
    for value in [best.get("function"), *(best.get("candidate_functions", []))]:
        if value and value not in ordered:
            ordered.append(value)
    for alt in best.get("alternatives", []):
        func = alt.get("function")
        if func and func not in ordered:
            ordered.append(func)
    for signal in (
        student_intent.get("goal_signal", {}),
        student_intent.get("study_signal", {}),
        student_intent.get("experience_signal", {}),
    ):
        for func, _score in sorted(signal.items(), key=lambda item: -item[1]):
            if func not in ordered:
                ordered.append(func)
    # Always include ops + support as fallback options for admin-like resumes
    # (the classifier never predicts these but they're common for project coordinators)
    for fallback in ["ops", "support", "administrative"]:
        if fallback not in ordered:
            ordered.append(fallback)

    # Deterministic pre-filter: eliminate functions with zero keyword support
    resume_lower = (resume_text or "").lower()
    _TECH_KEYWORDS = ["software", "developer", "engineer", "engineering", "programming",
                      "python", "java", "react", "aws", "cloud", "devops", "system",
                      "hardware", "firmware", "embedded", "full stack", "frontend",
                      "backend", "machine learning", "data science", "network",
                      "computer", "database", "server", "linux", "automation"]
    if "technology" in ordered and not any(kw in resume_lower for kw in _TECH_KEYWORDS):
        ordered.remove("technology")
        if "ops" not in ordered:
            ordered.insert(0, "ops")

    return ordered[:8]


def _route_conflict(best: dict, student_intent: dict) -> bool:
    top_function = best.get("function")
    goal_domains = set(student_intent.get("goal_domains", []))
    study_domains = set(student_intent.get("study_domains", []))
    return bool(goal_domains or study_domains) and top_function not in goal_domains | study_domains


def _should_run_route_stage(best: dict, student_intent: dict, llm_config: LLMConfig) -> bool:
    if llm_config.mode == "force":
        return True
    if llm_config.mode == "off":
        return False
    alternatives = best.get("alternatives", [])
    top_gap = 100
    if alternatives:
        top_gap = abs(best.get("match_pct", 0) - alternatives[0].get("match_pct", 0))
    return (
        best.get("match_pct", 0) < ROUTE_LOW_CONF
        or top_gap <= ROUTE_TOP_GAP
        or (
            student_intent.get("intent_confidence", 0.0) >= ROUTE_INTENT_MIN
            and _route_conflict(best, student_intent)
        )
    )


def _skill_junk_ratio(skills: list[str]) -> float:
    if not skills:
        return 1.0
    junk = sum(1 for skill in skills if skill.lower() in _JUNK_SKILLS)
    return junk / max(1, len(skills))


def _market_overlap(function: str, skills: list[str]) -> float:
    if not skills:
        return 0.0
    vocab = get_jd_skill_vocabulary(function)
    if not vocab:
        return 0.0
    normalized = {
        "".join(ch for ch in skill.lower() if ch.isalnum())
        for skill in skills
        if skill
    }
    overlap = len(normalized & vocab)
    return overlap / max(1, len(normalized))


def _should_run_evidence_stage(
    extracted_skills: list[str],
    chosen_function: str,
    current_needs_review: bool,
    llm_config: LLMConfig,
) -> bool:
    if llm_config.mode == "force":
        return True
    if llm_config.mode == "off":
        return False
    return (
        len(extracted_skills) > EVIDENCE_SKILL_MAX
        or _skill_junk_ratio(extracted_skills) > EVIDENCE_JUNK_RATIO
        or _market_overlap(chosen_function, extracted_skills) < EVIDENCE_MARKET_OVERLAP
        or current_needs_review
    )


def _job_titles_span_domains(titles: list[str]) -> bool:
    groups = {
        "healthcare": {"medical", "patient", "clinical", "care", "physician", "hospital"},
        "education": {"teacher", "tutor", "student", "school", "youth"},
        "technology": {"engineer", "technical", "it", "systems", "robotics", "software"},
        "service": {"front desk", "delivery", "driver", "retail", "customer"},
    }
    seen = set()
    for title in titles:
        lowered = title.lower()
        for group, tokens in groups.items():
            if any(token in lowered for token in tokens):
                seen.add(group)
    return len(seen) >= 2


def _should_run_jobs_stage(ranked_jobs: list[dict], chosen_function: str, llm_config: LLMConfig) -> bool:
    if llm_config.mode == "force":
        return True
    if llm_config.mode == "off":
        return False
    top_five = ranked_jobs[:5]
    if not top_five:
        return False
    titles = [job.get("title", "") for job in top_five]
    licensure_terms = ("physician", "dvm", "licensed", "director", "senior", "principal", "manager")
    return (
        any(any(term in title.lower() for term in licensure_terms) for title in titles)
        or median(job.get("fit", 0) for job in top_five) < 35
        or _job_titles_span_domains(titles[:3])
        or chosen_function in {"healthcare", "education", "technology"}
    )


def _candidate_subdomains(candidate_functions: list[str]) -> list[str]:
    ordered: list[str] = []
    for function in candidate_functions:
        for subdomain in subdomains_for_domain(function):
            if subdomain not in ordered:
                ordered.append(subdomain)
    return ordered


def _determine_subdomain(function: str, student_intent: dict, parsed_sections: dict[str, str]) -> tuple[str, float, list[str]]:
    weighted_texts = {
        "goal": (student_intent.get("intent_summary", {}).get("summary_text", ""), 3.0),
        "study": (student_intent.get("intent_summary", {}).get("study_text", ""), 2.5),
        "summary": (parsed_sections.get("summary", ""), 1.5),
        "experience": (
            "\n".join(
                parsed_sections.get(key, "")
                for key in ("experience", "leadership", "projects", "research", "skills")
            ).strip(),
            1.0,
        ),
    }
    chosen_subdomain, confidence, alternatives = choose_subdomain(function, weighted_texts)
    for seeded in student_intent.get("goal_subdomains", []):
        if seeded == chosen_subdomain:
            confidence = max(confidence, 0.72)
    for seeded in student_intent.get("study_subdomains", []):
        if seeded == chosen_subdomain:
            confidence = max(confidence, 0.68)
    return chosen_subdomain, confidence, alternatives


def _job_pool_conflict(ranked_jobs: list[dict]) -> bool:
    top = ranked_jobs[:5]
    if len(top) < 2:
        return False
    subdomains = {job.get("subdomain", "") for job in top if job.get("subdomain")}
    return len(subdomains) >= 3


def _should_use_rescue_lane(
    best: dict,
    student_intent: dict,
    subdomain_confidence: float,
    ranked_jobs: list[dict],
    lane_mode: str,
    llm_available: bool,
) -> bool:
    if lane_mode == "fast":
        return False
    if lane_mode == "rescue":
        return llm_available
    top_gap = 100
    alternatives = best.get("alternatives", [])
    if alternatives:
        top_gap = abs(best.get("match_pct", 0) - alternatives[0].get("match_pct", 0))
    return llm_available and (
        best.get("match_pct", 0) < 38
        or top_gap <= 12
        or subdomain_confidence < 0.45
        or _route_conflict(best, student_intent)
        or _job_pool_conflict(ranked_jobs)
    )


def analyze(
    resume_text: str,
    linkedin_text: str = "",
    headline_text: str = "",
    career_goals_text: str = "",
    smart_goals_text: str = "",
    hope_to_gain_text: str = "",
    ideal_career_text: str = "",
    study_text: str = "",
    ideal_careers: list[str] | None = None,
    resume_sections: dict[str, str] | None = None,
    top_k: int = 10,
    lane_mode: str = "hybrid",
    llm_config: LLMConfig | dict | None = None,
    llm_provider=None,
) -> dict:
    """Run the full goal-aware recommender pipeline and return a dict.

    ideal_careers: list of 3-5 career titles the student aspires to (e.g. ["Teacher", "Youth Counselor"]).
    Used to strengthen aspiration routing and generate tiered recommendations.
    """
    combined_text = "\n".join(part for part in [resume_text, linkedin_text] if part).strip()
    if not combined_text:
        return {"error": "Could not classify resume", "skills_found": []}

    # Merge ideal_careers list into goal text for stronger intent signal
    ideal_careers = ideal_careers or []
    if ideal_careers and not ideal_career_text:
        ideal_career_text_merged = "Ideal careers: " + ", ".join(ideal_careers)
    elif ideal_careers and ideal_career_text:
        ideal_career_text_merged = ideal_career_text + " | Ideal careers: " + ", ".join(ideal_careers)
    else:
        ideal_career_text_merged = ideal_career_text

    parsed_sections = parse_resume_sections(combined_text)
    if resume_sections:
        parsed_sections.update({k: v for k, v in resume_sections.items() if v})

    student_intent = build_student_intent_profile(
        resume_text=resume_text,
        linkedin_text=linkedin_text,
        headline_text=headline_text,
        career_goals_text=career_goals_text,
        smart_goals_text=smart_goals_text,
        hope_to_gain_text=hope_to_gain_text,
        ideal_career_text=ideal_career_text_merged,
        study_text=study_text,
        resume_sections=parsed_sections,
    )

    best = _match_role(
        combined_text,
        aspiration_signal=student_intent.get("goal_signal"),
        study_signal=student_intent.get("study_signal"),
        experience_signal=student_intent.get("experience_signal"),
    )
    if not best:
        return {"error": "Could not classify resume", "skills_found": []}

    llm_cfg = _coerce_llm_config(llm_config)
    orchestrator = LLMOrchestrator(config=llm_cfg, provider=llm_provider) if llm_cfg.mode != "off" or llm_provider else None
    llm_stage_results: dict[str, dict] = {}
    review_reasons: list[str] = []
    chosen_function = best["function"]
    pool = _build_function_pool(best, student_intent, combined_text)
    candidate_functions = pool[:3] if pool else [chosen_function]
    chosen_subdomain, subdomain_confidence, subdomain_alternatives = _determine_subdomain(
        chosen_function,
        student_intent,
        parsed_sections,
    )
    lane_used = "fast"
    allow_rescue = lane_mode != "fast"
    force_rescue = lane_mode == "rescue"

    if orchestrator and allow_rescue and (force_rescue or _should_run_route_stage(best, student_intent, llm_cfg)):
        route_payload = {
            "candidate_functions": _build_function_pool(best, student_intent, combined_text),
            "candidate_subdomains": _candidate_subdomains(_build_function_pool(best, student_intent, combined_text)),
            "headline_text": headline_text,
            "summary_text": student_intent.get("intent_summary", {}).get("summary_text", ""),
            "study_text": student_intent.get("intent_summary", {}).get("study_text", ""),
            "signal_breakdown": best.get("signal_breakdown", {}),
            "sections": {
                key: value
                for key, value in parsed_sections.items()
                if key in {"summary", "education", "experience", "leadership", "skills", "projects", "research"}
            },
        }
        route_result = orchestrator.run_stage("route", route_payload)
        llm_stage_results["route"] = route_result
        if route_result.get("accepted"):
            lane_used = "rescue"
            if route_result.get("chosen_function") and route_result["chosen_function"] != chosen_function:
                llm_corrected_function = True
            chosen_function = route_result.get("chosen_function") or chosen_function
            secondary = route_result.get("secondary_function")
            candidate_functions = [chosen_function]
            if secondary and secondary != chosen_function:
                candidate_functions.append(secondary)
            if route_result.get("goal_domains"):
                student_intent["goal_domains"] = route_result["goal_domains"]
            if route_result.get("study_domains"):
                student_intent["study_domains"] = route_result["study_domains"]
            if route_result.get("goal_roles"):
                student_intent["goal_roles"] = route_result["goal_roles"]
            if route_result.get("goal_subdomains"):
                student_intent["goal_subdomains"] = route_result["goal_subdomains"]
            if route_result.get("study_subdomains"):
                student_intent["study_subdomains"] = route_result["study_subdomains"]
            if route_result.get("chosen_subdomain"):
                chosen_subdomain = route_result["chosen_subdomain"]
                subdomain_confidence = max(subdomain_confidence, route_result.get("confidence", 0.0))
            else:
                chosen_subdomain, subdomain_confidence, subdomain_alternatives = _determine_subdomain(
                    chosen_function,
                    student_intent,
                    parsed_sections,
                )
        else:
            review_reasons.extend(route_result.get("review_reasons", []))

    skill_profile = extract_skill_profile(combined_text, resume_sections=parsed_sections)
    extracted_skills = skill_profile["skills"]

    # DeepSeek skill extraction: use zip's proven prompt for informal student language.
    # Only when ideal_careers is provided (engaged student) and DeepSeek key is available.
    from recommender.extract.llm_extractor import extract_skills_deepseek
    deepseek_skills = None
    if ideal_careers and DEEPSEEK_KEY:
        deepseek_skills = extract_skills_deepseek(combined_text)
        if deepseek_skills and deepseek_skills.get("skills"):
            # Merge: DeepSeek skills + vocabulary skills, deduplicated
            all_skills = list(dict.fromkeys(
                deepseek_skills["skills"] + extracted_skills
            ))
            extracted_skills = all_skills
            skill_profile["skills"] = all_skills
            skill_profile["deepseek_detected"] = deepseek_skills.get("detected", [])

    possible_skills: list[str] = []
    rejected_skills: list[str] = []
    verified_skills = list(extracted_skills)
    skills_for_retrieval = list(extracted_skills)  # preserve pre-evidence skills for job matching

    llm_corrected_function = False
    # For sparse resumes (< 30 skills extracted), be less aggressive about evidence cleaning.
    # Keep more skills as "possible" rather than rejecting them entirely.
    is_sparse = len(extracted_skills) < 30
    current_needs_review = best["match_pct"] < 35 or not extracted_skills
    if orchestrator and allow_rescue and (force_rescue or _should_run_evidence_stage(extracted_skills, chosen_function, current_needs_review, llm_cfg)):
        evidence_payload = {
            "skills": extracted_skills[:120],
            "section_skills": skill_profile.get("section_skills", {}),
            "allowed_sections": ["skills", "experience", "projects", "research", "leadership", "education"],
        }
        evidence_result = orchestrator.run_stage("evidence", evidence_payload)
        llm_stage_results["evidence"] = evidence_result
        if evidence_result.get("accepted"):
            lane_used = "rescue"
            verified = evidence_result.get("scored_skills", extracted_skills) or extracted_skills
            poss = evidence_result.get("possible_skills", [])
            rej = evidence_result.get("rejected_skills", [])
            # For sparse resumes, keep possible skills — don't strip too aggressively
            if is_sparse and poss:
                extracted_skills = list(verified) + poss
            else:
                extracted_skills = list(verified)
            verified_skills = list(verified)
            possible_skills = poss
            rejected_skills = rej
            skill_profile["skills"] = extracted_skills
            skill_profile["implicit_skills"] = evidence_result.get("implicit_skills", skill_profile["implicit_skills"])
            skill_profile["skill_evidence"] = [
                {
                    "skill": item.get("canonical_skill", ""),
                    "source_section": item.get("source_section", ""),
                    "matched_text": item.get("exact_span", ""),
                    "match_type": item.get("classification", ""),
                }
                for item in evidence_result.get("evidence_items", [])
            ]
        else:
            review_reasons.extend(evidence_result.get("review_reasons", []))

    raw_jobs = _build_candidate_jobs(candidate_functions, skills_for_retrieval, student_intent, chosen_subdomain, top_k)
    ranked_jobs = rank_jobs(
        jobs=raw_jobs,
        student_intent=student_intent,
        extracted_skills=extracted_skills,
        experience_skills=skill_profile["experience_skills"],
        chosen_function=chosen_function,
        chosen_subdomain=chosen_subdomain,
        lane=lane_used,
        top_k=max(top_k, 12),
    )
    if _should_use_rescue_lane(best, student_intent, subdomain_confidence, ranked_jobs, lane_mode, orchestrator is not None):
        lane_used = "rescue"
    # Skip the complex jobs judge — it causes DeepSeek JSON parse errors.
    # Instead, generate simple explanations in the output assembly below.
    gaps = build_gap_summary(
        ranked_jobs=ranked_jobs[:top_k],
        extracted_skills=extracted_skills,
        implicit_skills=skill_profile["implicit_skills"],
    )
    for possible_skill in possible_skills:
        if possible_skill not in gaps["verify_gaps"]:
            gaps["verify_gaps"].append(possible_skill)
    related_skills = get_related_skills(chosen_function, extracted_skills)
    coach_notes = _default_coach_notes(chosen_function, gaps["core_gaps"], gaps["verify_gaps"])

    # ── Tiered job recommendations ──
    # ready_now: top 2 jobs by skill overlap + high attainability (>0.45)
    # aspirational: top 3 jobs by goal alignment, lower attainability OK
    from recommender.llm.bridge_generator import generate_bridge, generate_skill_path

    ready_now = []
    aspirational = []
    seen_titles = set()
    for job in ranked_jobs:
        title = job.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)

        bd = job.get("job_score_breakdown", {})
        llm_adj = bd.get("llm_adjustments", {})
        why_text = llm_adj.get("why_fits", "") or job.get("why", "")

        entry = {
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "url": job.get("url", ""),
            "fit": job.get("fit", 0),
            "job_type": job.get("job_type", ""),
            "subdomain": job.get("subdomain", chosen_subdomain),
            "function": job.get("function", chosen_function),
            "why": _auto_explain(job.get("title", ""), chosen_function, chosen_subdomain, extracted_skills[:8], ideal_careers),
            "gaps": ", ".join(gaps["core_gaps"][:3]),
            "job_score_breakdown": job.get("job_score_breakdown", {}),
            "eligible": job.get("eligible", True),
            "eligibility_reason": job.get("eligibility_reason", ""),
        }

        if len(ready_now) < 2 and bd.get("attainability", 0) >= 0.45:
            # Check if this ready-now job differs from aspirational path
            job_func = job.get("function", chosen_function)
            if job_func != chosen_function and ideal_careers:
                entry["bridge"] = generate_bridge(
                    job_title=entry["title"],
                    job_function=job_func,
                    job_skills=job.get("skills", [])[:8],
                    student_skills=extracted_skills[:8],
                    ideal_careers=ideal_careers,
                    aspirational_function=chosen_function,
                )
            ready_now.append(entry)
            continue

        if len(aspirational) < 3:
            aspirational.append(entry)
            continue

        if len(ready_now) >= 2 and len(aspirational) >= 3:
            break

    # Fill any empty slots from remaining jobs
    used = {j["title"] for j in ready_now + aspirational}
    for job in ranked_jobs:
        title = job.get("title", "")
        if title in used:
            continue
        entry = {
            "title": title,
            "company": job.get("company", ""),
            "url": job.get("url", ""),
            "fit": job.get("fit", 0),
            "job_type": job.get("job_type", ""),
            "subdomain": job.get("subdomain", chosen_subdomain),
            "function": job.get("function", chosen_function),
            "why": _auto_explain(title, chosen_function, chosen_subdomain, extracted_skills[:8], ideal_careers),
            "gaps": ", ".join(gaps["core_gaps"][:3]),
            "job_score_breakdown": job.get("job_score_breakdown", {}),
        }
        if len(ready_now) < 2:
            ready_now.append(entry)
        elif len(aspirational) < 3:
            aspirational.append(entry)
        else:
            break

    # Generate skill progression tree + path
    skill_tree = None
    skill_path = None
    if ideal_careers:
        from recommender.llm.skill_tree import generate_skill_tree
        skill_tree = generate_skill_tree(
            student_skills=extracted_skills[:15],
            implicit_skills=[s.get("skill","") for s in skill_profile.get("implicit_skills", [])],
            core_gaps=gaps["core_gaps"],
            bridge_gaps=gaps["bridge_gaps"],
            stretch_gaps=gaps["stretch_gaps"],
            current_function=chosen_function,
            ideal_careers=ideal_careers,
        )
        skill_path = skill_tree.get("summary", "") if skill_tree else None

    # Legacy flat job list for backward compatibility
    jobs = ready_now + aspirational

    # Calibrate scores to 50-80% range while preserving spread.
    # Flat boost of 30 points keeps differentiation intact.
    for j in jobs:
        raw = j.get("fit", 0)
        if raw > 0:
            j["fit"] = min(round(raw + FIT_BOOST), 85)
    for j in ready_now:
        if j.get("fit", 0) > 0:
            j["fit"] = min(j["fit"] + FIT_READY_BOOST, 88)
    for j in aspirational:
        if j.get("fit", 0) > 0:
            j["fit"] = max(j["fit"] - FIT_ASPIRE_PENALTY, 50)

    return {
        "resume": resume_text.strip(),
        "function": chosen_function,
        "confidence": max(50, min(95, best["match_pct"] + CONFIDENCE_BOOST)),
        "skills": extracted_skills,
        "verified_skills": verified_skills,
        "possible_skills": possible_skills,
        "rejected_skills": rejected_skills,
        "skill_weights": {},
        "market_relevant": extracted_skills,
        "inferred": [item["skill"] for item in skill_profile["implicit_skills"]],
        "implicit_skills": skill_profile["implicit_skills"],
        "skill_evidence": skill_profile["skill_evidence"],
        "gaps": gaps["core_gaps"],
        "core_gaps": gaps["core_gaps"],
        "bridge_gaps": gaps["bridge_gaps"],
        "stretch_gaps": gaps["stretch_gaps"],
        "nice_to_have_gaps": gaps["nice_to_have_gaps"],
        "verify_gaps": gaps["verify_gaps"],
        "related": [{"skill": s, "pmi": round(sc, 2)} for s, sc in related_skills],
        "related_skills": [{"skill": s, "pmi": round(sc, 2)} for s, sc in related_skills],
        "alternatives": [{"function": a["function"], "pct": a["match_pct"]} for a in best.get("alternatives", [])[:5]],
        "coach_notes": coach_notes,
        "jobs": jobs,
        "ready_now": ready_now,
        "aspirational": aspirational,
        "skill_path": skill_path,
        "skill_tree": skill_tree,
        "ideal_careers": ideal_careers,
        "student_intent": {
            "goal_domains": student_intent.get("goal_domains", []),
            "goal_roles": student_intent.get("goal_roles", []),
            "goal_subdomains": student_intent.get("goal_subdomains", []),
            "study_domains": student_intent.get("study_domains", []),
            "study_subdomains": student_intent.get("study_subdomains", []),
            "study_program": student_intent.get("study_program", ""),
            "study_level": student_intent.get("study_level", ""),
            "intent_confidence": student_intent.get("intent_confidence", 0.0),
            "intent_strength": student_intent.get("intent_strength", 0.0),
            "has_student_intent": student_intent.get("has_student_intent", False),
        },
        "intent_summary": student_intent.get("intent_summary", {}),
        "signal_breakdown": best.get("signal_breakdown", {}),
        "candidate_functions": candidate_functions,
        "subdomain": chosen_subdomain,
        "subdomain_confidence": subdomain_confidence,
        "subdomain_alternatives": subdomain_alternatives,
        "lane_used": lane_used,
        "job_cluster": [job.get("subdomain", chosen_subdomain) for job in jobs[:5]],
        "needs_review": (best["match_pct"] < 35 and not llm_corrected_function) or not extracted_skills or bool(review_reasons),
        "review_reasons": sorted(set(review_reasons)),
        "llm_stage_results": llm_stage_results,
        "llm_trace": [
            {
                "stage": stage,
                "accepted": result.get("accepted", False),
                "review_reasons": result.get("review_reasons", []),
            }
            for stage, result in llm_stage_results.items()
        ],
        "normalization_source": "hybrid" if llm_stage_results else "deterministic",
    }


def main():
    parser = argparse.ArgumentParser(description="Run the SpeakHire recommender pipeline.")
    parser.add_argument("input", nargs="?", help="Resume text or a path to a resume text file.")
    parser.add_argument("--resume-file", help="Path to a resume text file.")
    parser.add_argument("--linkedin-file", help="Path to a LinkedIn markdown/text file.")
    parser.add_argument("--headline-text", default="", help="Explicit headline text.")
    parser.add_argument("--career-goals-text", default="", help="Explicit career goals text.")
    parser.add_argument("--smart-goals-text", default="", help="SMART goals text.")
    parser.add_argument("--hope-to-gain-text", default="", help="What the student hopes to gain.")
    parser.add_argument("--ideal-career-text", default="", help="Ideal career text.")
    parser.add_argument("--study-text", default="", help="Study or major text.")
    parser.add_argument("--ideal-careers", nargs="*", default=None, help="3-5 ideal career titles (e.g. Teacher 'Youth Counselor').")
    parser.add_argument("--top-k", type=int, default=10, help="Number of jobs to return.")
    parser.add_argument("--lane-mode", choices=["fast", "hybrid", "rescue"], default="hybrid")
    parser.add_argument("--disable-embeddings", action="store_true")
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--subdomain-debug", action="store_true")
    parser.add_argument("--llm-mode", choices=["off", "hybrid", "force"], default=None)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model-route", default=None)
    parser.add_argument("--llm-model-evidence", default=None)
    parser.add_argument("--llm-model-jobs", default=None)
    parser.add_argument("--llm-max-calls", type=int, default=None)
    parser.add_argument("--llm-debug", action="store_true")
    args = parser.parse_args()

    linkedin_text = ""
    if args.linkedin_file:
        with open(args.linkedin_file, encoding="utf-8") as handle:
            linkedin_text = handle.read()

    resume_text = ""
    if args.resume_file:
        with open(args.resume_file, encoding="utf-8") as handle:
            resume_text = handle.read()
    elif args.input:
        if os.path.isfile(args.input):
            with open(args.input, encoding="utf-8") as handle:
                resume_text = handle.read()
        else:
            resume_text = args.input
    else:
        print("Paste resume text (Ctrl+Z then Enter on Windows, Ctrl+D on Unix):")
        resume_text = sys.stdin.read()

    if not resume_text.strip():
        print("Error: no resume text provided.")
        sys.exit(1)

    if args.disable_embeddings:
        os.environ["RECOMMENDER_DISABLE_EMBEDDINGS"] = "1"

    llm_config = None
    effective_llm_mode = "off" if args.disable_llm else (args.llm_mode or _default_llm_mode())
    if any(
        value is not None
        for value in [
            args.llm_mode,
            args.llm_provider,
            args.llm_model_route,
            args.llm_model_evidence,
            args.llm_model_jobs,
            args.llm_max_calls,
        ]
    ) or args.llm_debug:
        defaults = LLMConfig()
        llm_config = LLMConfig(
            mode=effective_llm_mode,
            provider_name=args.llm_provider or "openrouter",
            route_model=args.llm_model_route or defaults.route_model,
            evidence_model=args.llm_model_evidence or defaults.evidence_model,
            jobs_model=args.llm_model_jobs or defaults.jobs_model,
            max_calls=args.llm_max_calls or defaults.max_calls,
            debug=args.llm_debug,
        )
    elif args.disable_llm:
        llm_config = LLMConfig(mode="off")

    result = analyze(
        resume_text,
        linkedin_text=linkedin_text,
        headline_text=args.headline_text,
        career_goals_text=args.career_goals_text,
        smart_goals_text=args.smart_goals_text,
        hope_to_gain_text=args.hope_to_gain_text,
        ideal_career_text=args.ideal_career_text,
        study_text=args.study_text,
        ideal_careers=args.ideal_careers,
        top_k=args.top_k,
        lane_mode=args.lane_mode,
        llm_config=llm_config,
    )
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    if args.subdomain_debug:
        result["subdomain_debug"] = {
            "subdomain": result.get("subdomain", ""),
            "subdomain_confidence": result.get("subdomain_confidence", 0.0),
            "subdomain_alternatives": result.get("subdomain_alternatives", []),
            "candidate_functions": result.get("candidate_functions", []),
            "job_cluster": result.get("job_cluster", []),
        }

    print("=" * 60)
    print("BEST MATCH")
    print("=" * 60)
    print(f"  Function:   {result['function']}")
    print(f"  Confidence: {result.get('confidence', 0)}%")
    print()

    print("=" * 60)
    print("SKILLS FOUND")
    print("=" * 60)
    print("  " + ", ".join(result.get("skills", [])[:20]))
    print()

    if result.get("gaps"):
        print("=" * 60)
        print("CORE GAPS")
        print("=" * 60)
        print("  " + ", ".join(result["gaps"][:10]))
        print()

    if result.get("coach_notes"):
        print("=" * 60)
        print("COACH NOTES")
        print("=" * 60)
        print("  " + result["coach_notes"])
        print()

    if result.get("jobs"):
        print("=" * 60)
        print("TOP JOB OPENINGS")
        print("=" * 60)
        for idx, job in enumerate(result["jobs"], 1):
            print(f"  {idx}. {job['title']} [{job.get('fit', 0)}% {job.get('job_type', '')}]")
            if job.get("company"):
                print(f"     @ {job['company']}")
            if job.get("why"):
                print(f"     Why: {job['why']}")
            if job.get("gaps"):
                print(f"     Gaps: {job['gaps']}")
            if job.get("url"):
                print(f"     {job['url'][:100]}")


if __name__ == "__main__":
    main()
