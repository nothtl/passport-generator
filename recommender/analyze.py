"""Goal-aware recommender analysis pipeline."""
from __future__ import annotations

import argparse
import os
import sys
from statistics import median

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from recommender.extract.section_parser import parse_resume_sections, extract_experience_from_sections
from recommender.extract.skill_extractor import extract_skill_profile
from recommender.llm import LLMConfig, LLMOrchestrator
from recommender.llm.provider import load_openrouter_api_key
from recommender.match.ensemble_matcher import match_role as _match_role
from recommender.match.subdomains import choose_subdomain, subdomains_for_domain
from recommender.match.student_intent import build_student_intent_profile
from recommender.rank.job_ranker import build_gap_summary, rank_jobs
from recommender.retrieve.retriever import (
    _SUBSET_FUNCTION_MAP,
    filter_job_records,
    get_jd_skill_vocabulary,
    get_related_skills,
    retrieve_jds,
    retrieve_from_subset,
)
from recommender.retrieve.eligibility import screen_jobs, filter_ineligible
from recommender.config import get_config, get_pipeline, get_resume, get_functions, get_cross_function, get_function_keywords, get_student_lifecycle

_CFG = get_config()
_PIPE = get_pipeline()
_RESUME = get_resume()
_FUNCS = get_functions()
_XFUNC = get_cross_function()
_LIFECYCLE = get_student_lifecycle()

_JUNK_SKILLS = set(_RESUME.junk_skills)
_DEFAULT_LLM_CACHE_DIR = os.path.join(_PROJECT_DIR, "recommender", ".cache", "llm")

# Job title function-keyword mapping for internship relevance scoring
_TITLE_KEYWORDS = {
    "technology": ["software", "engineer", "developer", "data", "systems", "it ", "tech", "programmer", "web", "cloud", "devops", "frontend", "backend", "java", "python", "react", "aws", "ai ", "machine learning", "embedded", "network", "database", "automation"],
    "healthcare": ["nurse", "medical", "patient", "clinical", "health", "care", "therapy", "therapist", "pharmacy", "physician", "hospital", "cna", "rbt", "behavior", "respite", "oncology"],
    "education": ["teacher", "tutor", "instructor", "teaching", "education", "school", "classroom", "faculty", "lecturer", "instructional", "academic", "substitute", "curriculum", "youth", "after school", "mentor"],
    "finance": ["finance", "accounting", "analyst", "accountant", "banking", "investment", "audit", "tax", "payroll", "bookkeeping", "fp&a", "financial", "advisor", "underwriter"],
    "sales": ["sales", "retail", "representative", "account", "client", "business development", "stylist", "merchandising", "associate"],
    "arts-media": ["design", "graphic", "photo", "video", "editor", "content", "creative", "media", "marketing", "social media", "illustrator", "photographer", "videographer", "ui ", "ux ", "visual", "brand", "art "],
    "social-service": ["social", "community", "outreach", "peer", "youth", "counselor", "advocate", "case", "nonprofit", "volunteer", "support specialist"],
    "ops": ["operations", "logistics", "supply", "warehouse", "inventory", "dispatch", "coordinator"],
    "support": ["support", "help desk", "service desk", "technician", "customer service", "representative"],
}

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# ── Tuning constants (from config) ──
CONFIDENCE_BOOST = _PIPE.confidence_boost
FIT_BOOST = _PIPE.fit_boost
FIT_READY_BOOST = _PIPE.fit_ready_boost
FIT_ASPIRE_PENALTY = _PIPE.fit_aspire_penalty
ROUTE_LOW_CONF = _PIPE.route_low_conf
ROUTE_TOP_GAP = _PIPE.route_top_gap
ROUTE_INTENT_MIN = _PIPE.route_intent_min
EVIDENCE_SKILL_MAX = _PIPE.evidence_skill_max
EVIDENCE_JUNK_RATIO = _PIPE.evidence_junk_ratio
EVIDENCE_MARKET_OVERLAP = _PIPE.evidence_market_overlap
SPARSE_SKILL_THRESHOLD = _PIPE.sparse_skill_threshold
NARROW_POOLS = set(_PIPE.narrow_pools)
XF_BOOST = _PIPE.xf_boost

# All function labels in the corpus — used for cross-function exploration
_ALL_FUNCTIONS = list(_FUNCS.all)


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
    ideal_careers: list[str] | None = None,
    level_filter: str = "Entry",
) -> list[dict]:
    """Build candidate job list.

    level_filter: the seniority level to retrieve — defaults to "Entry".
    When the student needs internships (high_school / college_freshman),
    pass "Intern" to prioritize intern-level jobs first, falling back to
    Entry-level only if the intern pool is too thin.
    """
    # Use subset parquet if available, fall back to per-function parquets.
    # Only pull from the chosen function + secondary (not all alternatives).
    # For sparse resumes or functions with narrow pools, also pull from broad fallback pools.
    primary_function = candidate_functions[0] if candidate_functions else ""
    secondary = candidate_functions[1] if len(candidate_functions) > 1 else ""
    # Primary + secondary both pulled. Secondary jobs are filtered by a
    # higher skill-match bar to prevent pollution (healthcare in tech slots).
    pull_functions = [primary_function]
    if secondary and secondary != primary_function:
        pull_functions.append(secondary)

    # Only expand to broad pools when the primary pool has very few jobs
    if primary_function in NARROW_POOLS and len(pull_functions) <= 2:
        primary_jobs = retrieve_jds(primary_function, level_filter, student_skills, top_k=5)
        if len(primary_jobs) < 3:  # Only expand if primary pool is truly thin
            for broad in ["support", "ops", "education", "healthcare"]:
                if broad not in pull_functions:
                    pull_functions.append(broad)
    # Detect career level: college grads get mid-level jobs included
    study_level = student_intent.get("study_level", "")
    is_college_grad = study_level in ("bachelor", "master", "doctorate", "associate")

    # ── Phase 1a: internship-specific retrieval ──
    # When level_filter indicates internships, retrieve intern-level jobs
    # FIRST from per-function parquets. Only fall back to entry-level if the
    # intern pool is too thin (< top_k * 2 results).
    is_intern_mode = level_filter.lower().startswith("intern")
    intern_jobs: list[dict] = []
    if is_intern_mode:
        _intern_levels = list(_LIFECYCLE.level_filter.intern)
        for function in pull_functions:
            for _lvl in _intern_levels:
                try:
                    _ijobs = retrieve_jds(function, _lvl, student_skills, top_k=max(10, top_k * 2))
                    for _j in _ijobs:
                        _key = _j.get("id") or _j.get("url") or f"{_j.get('title', '')}::{_j.get('company', '')}"
                        if _key not in seen:
                            seen.add(_key)
                            _j["_internship"] = True
                            intern_jobs.append(_j)
                except Exception:
                    pass
            if len(intern_jobs) >= top_k * 2:
                break
        # If we got enough intern jobs, return them directly (with eligibility filter)
        if len(intern_jobs) >= top_k * 2:
            intern_jobs = filter_job_records(
                intern_jobs,
                candidate_function=candidate_functions[0] if candidate_functions else "",
                goal_domains=student_intent.get("goal_domains", []),
                study_domains=student_intent.get("study_domains", []),
                goal_subdomains=student_intent.get("goal_subdomains", []),
                study_subdomains=student_intent.get("study_subdomains", []),
                study_level=student_intent.get("study_level", ""),
            )
            intern_jobs = filter_ineligible(intern_jobs)
            return intern_jobs

    # Try per-function parquets first (better data), subset as fallback
    combined: list[dict] = []
    seen = set()
    # Primary function: use hybrid BM25 + dense retrieval
    # Secondary functions: use structured v2 retrieval (cheaper)
    from recommender.retrieve.hybrid import hybrid_retrieve
    for fi, function in enumerate(pull_functions):
        if fi == 0:
            rows = hybrid_retrieve(function, resume_text=" ".join(student_intent.get("goal_roles", [])) + " " + " ".join(student_skills[:20]), student_skills=student_skills, top_k=max(15, top_k * 3))
        else:
            rows = retrieve_from_subset(function=function, student_skills=student_skills, top_k=max(15, top_k * 3))
        # For college grads, search full dataset with skill keywords for mid-level jobs
        if is_college_grad and student_skills:
            from recommender.retrieve.retriever import _stream_full_parquet
            # Use the student's actual skills + ideal careers as search keywords
            skill_kw = [s.lower() for s in student_skills[:15] if len(s) > 3 and s.isalpha()]
            career_kw = [w.lower() for c in (ideal_careers or []) for w in c.split() if len(w) > 3]
            search_kw = list(set(skill_kw + career_kw))[:15]
            mid_jobs = _stream_full_parquet(
                function_labels={function},
                keyword_filters=search_kw,
                top_k=3, include_all_levels=True,
            )
            for mj in mid_jobs:
                key = mj.get("id") or mj.get("url") or f"{mj.get('title', '')}::{mj.get('company', '')}"
                if key not in seen:
                    seen.add(key)
                    # Boost mid-level jobs so they rank above student-level jobs
                    mj["_college_grad"] = True
                    rows.append(mj)
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

    # The 21GB dataset has very few entry-level jobs for niche functions
    # (aerospace, protective-service, etc). We use per-function parquets + subset
    # as the primary source, with O*NET fallback for the thinnest functions.

    filtered = filter_job_records(
            combined,
            candidate_function=candidate_functions[0] if candidate_functions else "",
            goal_domains=student_intent.get("goal_domains", []),
            study_domains=student_intent.get("study_domains", []),
            goal_subdomains=student_intent.get("goal_subdomains", []),
            study_subdomains=student_intent.get("study_subdomains", []),
            study_level=student_intent.get("study_level", ""),
            prefer_state=student_intent.get("preferred_state", ""),
        )
    # Screen for youth eligibility
    filtered = filter_ineligible(filtered)
    return filtered


def _auto_explain(title: str, function: str, subdomain: str, skills: list[str], goals: list[str], breakdown: dict | None = None) -> str:
    """Generate a contextual explanation for why this job fits the student.

    Uses the score breakdown to give specific, actionable reasons instead of
    generic skill-listing.
    """
    reasons = []
    bd = breakdown or {}

    # Skill overlap
    skill_pct = bd.get("skill_overlap_raw", 0)
    if skill_pct > 0.4:
        reasons.append(f"Strong skill match ({int(skill_pct * 100)}% overlap)")
    elif skill_pct > 0.2:
        reasons.append(f"Moderate skill match ({int(skill_pct * 100)}% overlap)")

    # Goal alignment
    goal_score = bd.get("goal_alignment", 0)
    if goal_score > 0.3:
        reasons.append("aligns with your career goals")

    # Attainability
    attain = bd.get("attainability", 1.0)
    if attain < 0.5:
        reasons.append("may require credentials you don't have yet")
    elif attain >= 0.8:
        reasons.append("attainable with your current profile")

    # Study alignment
    study_score = bd.get("study_alignment", 0)
    if study_score > 0.2:
        reasons.append("matches your field of study")

    if reasons:
        # Capitalize first letter of each reason, join with ". "
        formatted = []
        for r in reasons[:3]:
            r = r.strip()
            if r:
                formatted.append(r[0].upper() + r[1:] if len(r) > 1 else r.upper())
        return ". ".join(formatted) + "."

    # Fallback to original logic
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
    """Build ordered list of functions to search for jobs.

    Includes intent-aware conflict resolution: when the student's stated goals
    strongly conflict with the classifier's signal (>30% gap), BOTH functions
    are kept as primary candidates so the student sees options from both paths.
    """
    ordered: list[str] = []
    classifier_func = best.get("function")
    classifier_pct = best.get("match_pct", 0)

    # Always include the classifier's top pick
    if classifier_func and classifier_func not in ordered:
        ordered.append(classifier_func)

    # Include candidate functions (top-2 when close)
    for value in best.get("candidate_functions", []):
        if value and value not in ordered:
            ordered.append(value)

    # Intent signals
    intent_funcs: set[str] = set()
    for signal in (
        student_intent.get("goal_signal", {}),
        student_intent.get("study_signal", {}),
        student_intent.get("experience_signal", {}),
    ):
        for func, score in sorted(signal.items(), key=lambda item: -item[1]):
            if score > 0.15 and func not in ordered:
                intent_funcs.add(func)

    # Intent worship fix: when classifier strongly disagrees with intent,
    # keep BOTH paths. A student saying "I want education" whose skills
    # scream arts-media should see jobs from both pools.
    intent_top = next(iter(intent_funcs), None)
    if intent_top and classifier_func and classifier_func not in intent_funcs:
        # Check if classifier signal is materially different from intent
        signal_breakdown = best.get("signal_breakdown", {})
        classifier_probs = signal_breakdown.get("classifier", {})
        intent_max = max(
            signal_breakdown.get("aspiration", {}).get(intent_top, 0),
            signal_breakdown.get("study", {}).get(intent_top, 0),
        )
        classifier_support = classifier_probs.get(classifier_func, 0)

        # If classifier strongly supports a different function (>35%)
        # AND intent strongly supports its direction, keep both
        if classifier_support >= 35 and intent_max >= 25 and intent_top != classifier_func:
            if intent_top not in ordered:
                ordered.insert(1, intent_top)  # right after classifier, before other candidates

    # Add remaining intent functions
    for func in intent_funcs:
        if func not in ordered:
            ordered.append(func)

    # Alternatives from ensemble
    for alt in best.get("alternatives", []):
        func = alt.get("function")
        if func and func not in ordered:
            ordered.append(func)

    # Only add fallback functions when the original pool is very thin.
    # For sparse resumes (soft skills only), prefer education/support/administrative.
    # NEVER add skilled-trade, food-service etc. unless the classifier explicitly
    # routed there — these are manual labor pools that don't match professional resumes.
    _VALID_FALLBACKS = list(_PIPE.valid_fallbacks)
    if len(ordered) < 5:
        for fallback in _VALID_FALLBACKS:
            if fallback not in ordered:
                ordered.append(fallback)

    # Deterministic pre-filter: eliminate technology if resume has no tech keywords
    resume_lower = (resume_text or "").lower()
    _TECH_KEYWORDS = list(_RESUME.tech_keywords)
    _ADMIN_KEYWORDS = list(_RESUME.admin_keywords)
    has_tech = any(kw in resume_lower for kw in _TECH_KEYWORDS)
    has_admin = any(kw in resume_lower for kw in _ADMIN_KEYWORDS)
    if "technology" in ordered and not has_tech and has_admin:
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
    student_stage: str | None = None,
) -> dict:
    """Run the full goal-aware recommender pipeline and return a dict.

    ideal_careers: list of 3-5 career titles the student aspires to (e.g. ["Teacher", "Youth Counselor"]).
    Used to strengthen aspiration routing and generate tiered recommendations.

    student_stage: override the auto-detected lifecycle stage. When None, the
    stage is auto-detected from student_intent.study_level via the
    study_level_mapping in config. Early-stage students (high_school,
    college_freshman) get internship recommendations as their primary tier,
    with entry-level jobs as aspirational (post-graduation goals).
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

    # Extract experience level from date ranges
    experience_months, experience_level = extract_experience_from_sections(parsed_sections)

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

    # ── Phase 1b: student lifecycle awareness ──────────────────────
    # Auto-detect student stage from study_level unless caller overrides.
    _study_lvl_raw = student_intent.get("study_level", "")
    _study_level_mapping = dict(_LIFECYCLE.study_level_mapping)
    if student_stage is None:
        student_stage = _study_level_mapping.get(_study_lvl_raw, "")
    # Resolve stage label and recommended job type from config.
    _stages_cfg = dict(_LIFECYCLE.stages)
    _stage_info = _stages_cfg.get(student_stage, {})
    stage_label = _stage_info.get("label", "")
    # Early-stage students (high_school, college_freshman) need internships
    # as their primary recommendations; entry-level jobs are aspirational.
    _needs_internships = student_stage in ("high_school", "college_freshman")
    _intern_level_filter = "Intern" if _needs_internships else "Entry"

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

    # ── Phase 2: skill extraction (fast, synchronous) ──────────────
    skill_profile = extract_skill_profile(combined_text, resume_sections=parsed_sections)
    extracted_skills = skill_profile["skills"]

    # ── Phase 3: parallel LLM calls (route + deepseek + evidence) ──
    # Previously sequential: route(30s) → deepseek(30s) → evidence(30s) = 90s.
    # Now all three fire concurrently and we collect results together.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    route_payload = None
    evidence_payload = None
    needs_deepseek = bool(ideal_careers and DEEPSEEK_KEY)
    needs_route = bool(orchestrator and allow_rescue and (force_rescue or _should_run_route_stage(best, student_intent, llm_cfg)))
    is_sparse = len(extracted_skills) < 30
    current_needs_review = best["match_pct"] < 35 or not extracted_skills
    needs_evidence = bool(orchestrator and allow_rescue and (force_rescue or _should_run_evidence_stage(extracted_skills, chosen_function, current_needs_review, llm_cfg)))

    if needs_route:
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
    if needs_evidence:
        evidence_payload = {
            "skills": extracted_skills[:120],
            "section_skills": skill_profile.get("section_skills", {}),
            "allowed_sections": ["skills", "experience", "projects", "research", "leadership", "education"],
        }

    llm_corrected_function = False
    futures: dict = {}
    if needs_route or needs_evidence or needs_deepseek:
        with ThreadPoolExecutor(max_workers=3) as pool:
            if needs_route and orchestrator:
                futures["route"] = pool.submit(orchestrator.run_stage, "route", route_payload)
            if needs_deepseek:
                from recommender.extract.llm_extractor import extract_skills_deepseek
                futures["deepseek"] = pool.submit(extract_skills_deepseek, combined_text)
            if needs_evidence and orchestrator:
                futures["evidence"] = pool.submit(orchestrator.run_stage, "evidence", evidence_payload)

            # Collect results as they complete
            for future in as_completed(futures.values()):
                pass  # results collected below

    # Process route result
    if "route" in futures:
        route_result = futures["route"].result()
        llm_stage_results["route"] = route_result
        if route_result.get("accepted"):
            lane_used = "rescue"
            if route_result.get("chosen_function") and route_result["chosen_function"] != chosen_function:
                llm_corrected_function = True
            chosen_function = route_result.get("chosen_function") or chosen_function
            secondary = route_result.get("secondary_function")
            # Preserve diverse pool: route picks first, original pool follows
            route_picks = [chosen_function]
            if secondary and secondary != chosen_function:
                route_picks.append(secondary)
            # Keep original pool functions that route judge didn't pick
            for f in candidate_functions:
                if f not in route_picks:
                    route_picks.append(f)
            candidate_functions = route_picks[:4]
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
                    chosen_function, student_intent, parsed_sections)
        else:
            review_reasons.extend(route_result.get("review_reasons", []))

    # Process DeepSeek result
    if "deepseek" in futures:
        deepseek_skills = futures["deepseek"].result()
        if deepseek_skills and deepseek_skills.get("skills"):
            all_skills = list(dict.fromkeys(deepseek_skills["skills"] + extracted_skills))
            extracted_skills = all_skills
            skill_profile["skills"] = all_skills
            skill_profile["deepseek_detected"] = deepseek_skills.get("detected", [])

    possible_skills: list[str] = []
    rejected_skills: list[str] = []
    verified_skills = list(extracted_skills)
    # Deduplicate multi-resume skills to prevent K-inflation in hypergeometric scoring.
    # Students with 3 resume versions concatenated can get 260+ skills, making K>100
    # and causing any overlap to look statistically insignificant.
    _seen_normed = set()
    _deduped = []
    for _s in extracted_skills:
        _normed = "".join(ch for ch in str(_s).lower() if ch.isalnum())
        if _normed not in _seen_normed:
            _seen_normed.add(_normed)
            _deduped.append(_s)
    extracted_skills = _deduped

    # Cap at 80 most distinctive skills for ranking (prevents K-inflation).
    # Skills beyond 80 add noise without improving match quality.
    if len(extracted_skills) > 80:
        try:
            from recommender.retrieve.retriever import _compute_idf
            idf = _compute_idf(chosen_function.lower())
            def _norm_idf(s):
                return "".join(ch for ch in str(s).lower() if ch.isalnum())
            scored = [(s, idf.get(_norm_idf(s), 0.0)) for s in extracted_skills]
            scored.sort(key=lambda x: -x[1])
            extracted_skills = [s for s, _ in scored[:80]]
        except Exception:
            extracted_skills = extracted_skills[:80]

    def _norm_skill(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    # Expand skills via knowledge graph (1-2 hop traversal).
    # Bridges "computer vision" → "machine learning" → "AI" etc.
    try:
        from recommender.data.skill_graph import expand_skills
        _expanded = expand_skills(extracted_skills, max_hops=2, top_k=40)
        # Merge: expanded skills first, then original ones not yet included
        _seen_sk = set(_norm_skill(s) for s in _expanded)
        for _s in extracted_skills:
            if _norm_skill(_s) not in _seen_sk:
                _expanded.append(_s)
                _seen_sk.add(_norm_skill(_s))
        skills_for_retrieval = _expanded
    except Exception:
        skills_for_retrieval = list(extracted_skills)

    # Process evidence result
    if "evidence" in futures:
        evidence_result = futures["evidence"].result()
        llm_stage_results["evidence"] = evidence_result
        if evidence_result.get("accepted"):
            lane_used = "rescue"
            verified = evidence_result.get("scored_skills", extracted_skills) or extracted_skills
            poss = evidence_result.get("possible_skills", [])
            rej = evidence_result.get("rejected_skills", [])
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

    # ── Skill-match guard: verify the chosen function's pool has matching jobs ──
    # If route judge picked a function where the student's skills don't match any JDs,
    # fall back to the classifier's top function (which has actual skill overlap).
    if llm_corrected_function and lane_used == "rescue":
        test_jobs = _build_candidate_jobs([chosen_function], skills_for_retrieval, student_intent, chosen_subdomain, 5, ideal_careers)
        if len(test_jobs) >= 3:
            test_ranked = rank_jobs(
                jobs=test_jobs, student_intent=student_intent,
                extracted_skills=extracted_skills,
                experience_skills=skill_profile["experience_skills"],
                chosen_function=chosen_function, chosen_subdomain=chosen_subdomain,
                lane=lane_used, top_k=5,
            )
            low_skill = sum(1 for j in test_ranked
                          if j.get("job_score_breakdown", {}).get("skill_overlap_raw", 0) < 0.10)
            if low_skill >= 4:  # 4/5 jobs have near-zero skill overlap
                # Route judge picked a bad pool — fall back to classifier
                classifier_func = best.get("function")
                if classifier_func and classifier_func != chosen_function:
                    chosen_function = classifier_func
                    candidate_functions = [classifier_func]
                    llm_corrected_function = False
                    review_reasons.append("route_judge_pool_rejected_low_skill_match")

    raw_jobs = _build_candidate_jobs(candidate_functions, skills_for_retrieval, student_intent, chosen_subdomain, top_k, ideal_careers)
    ranked_jobs = rank_jobs(
        jobs=raw_jobs,
        student_intent=student_intent,
        extracted_skills=skills_for_retrieval,
        experience_skills=skill_profile["experience_skills"],
        chosen_function=chosen_function,
        chosen_subdomain=chosen_subdomain,
        lane=lane_used,
        top_k=max(top_k, 20),
    )
    # For college grads: inject mid-level AI/tech jobs into aspirational tier
    study_lvl = student_intent.get("study_level", "")
    _is_grad = study_lvl in ("bachelor", "master", "doctorate", "associate")
    if _is_grad and ideal_careers:
        from recommender.retrieve.retriever import _stream_full_parquet
        skill_kw = [s.lower() for s in extracted_skills[:20] if len(s) > 3 and s.isalpha()]
        career_kw = [w.lower() for c in (ideal_careers or []) for w in c.split() if len(w) > 3]
        search_kw = list(set(skill_kw + career_kw))[:15]
        mid = _stream_full_parquet(
            function_labels={chosen_function, "engineering"},
            keyword_filters=search_kw, top_k=5, include_all_levels=True,
        )
        for mj in reversed(mid):
            mj["_college_grad"] = True
            mj["fit"] = 60  # Default score for mid-level matches
            mj.setdefault("job_score_breakdown", {})["source"] = "mid_level"
            ranked_jobs.insert(0, mj)

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
    # Pull PMI from ALL candidate + intent functions BEFORE cross-function check.
    # This ensures healthcare/childcare signals appear when Abigail's bilingual +
    # mentoring + outreach skills co-occur with clinical/care roles.
    for xf in candidate_functions[1:4]:
        if xf and xf != chosen_function:
            try:
                xf_related = get_related_skills(xf, extracted_skills)
                seen_rel = {r[0] if isinstance(r, tuple) else r["skill"] for r in related_skills}
                for r in xf_related:
                    key = r[0] if isinstance(r, tuple) else r["skill"]
                    if key not in seen_rel:
                        seen_rel.add(key)
                        related_skills.append(r)
            except Exception:
                pass
    # Also pull PMI from intent domains even if not in candidate_functions
    for intent_domain in set(student_intent.get("goal_domains", [])) | set(student_intent.get("study_domains", [])):
        if intent_domain and intent_domain not in candidate_functions and intent_domain != chosen_function:
            try:
                xf_related = get_related_skills(intent_domain, extracted_skills)
                seen_rel = {r[0] if isinstance(r, tuple) else r["skill"] for r in related_skills}
                for r in xf_related:
                    key = r[0] if isinstance(r, tuple) else r["skill"]
                    if key not in seen_rel:
                        seen_rel.add(key)
                        related_skills.append(r)
            except Exception:
                pass

    # ── Cross-function exploration ──────────────────────────────────
    # PMI-based: skills like the student's that frequently co-occur with
    # jobs in other functions trigger exploration of those pools.
    # Fixes the Abigail Rodriguez problem: bilingual + mentoring + outreach
    # should surface healthcare/childcare jobs even if intent says education.
    cross_function_jobs: list[dict] = []
    if len(candidate_functions) < 5 and len(ranked_jobs) < top_k * 3:
        pm_functions = set(candidate_functions)

        # Direct skill-to-function adjacency: skills that strongly imply
        # other career paths even without PMI data from the chosen function.
        _SKILL_FUNCTION_HINTS = dict(_XFUNC.skill_hints)
        student_skills_normed = {
            "".join(ch for ch in s.lower() if ch.isalnum())
            for s in extracted_skills[:30]
        }
        for skill_key, funcs in _SKILL_FUNCTION_HINTS.items():
            if skill_key in student_skills_normed:
                for f in funcs:
                    if f not in pm_functions and f not in candidate_functions:
                        pm_functions.add(f)

        # PMI-based exploration
        for rel in related_skills[:8]:
            skill, pmi = rel if isinstance(rel, tuple) else (rel.get("skill", ""), rel.get("pmi", 0))
            if pmi < 3.5:
                continue
            for func in _ALL_FUNCTIONS:
                if func in pm_functions or func in candidate_functions:
                    continue
                func_terms = dict(_XFUNC.function_terms)
                func_kw = func_terms.get(func, [func.replace("-", " ")])
                if any(kw in skill.lower() for kw in func_kw):
                    pm_functions.add(func)
                    break

        for xfunc in pm_functions:
            if xfunc in candidate_functions or xfunc == chosen_function:
                continue
            try:
                # Full-text keyword search for cross-function pools.
                search_keywords = [s.lower() for s in extracted_skills[:20]
                                   if len(s) > 3 and s.isalpha()]
                search_keywords += [s.get("skill","").lower()
                                    for s in skill_profile.get("implicit_skills", [])[:3]]
                search_keywords += [c.lower() for c in (ideal_careers or [])[:3]]

                # Caregiving/childcare keyword expansion from resume text
                resume_lower = (resume_text or "").lower()
                _CAREGIVING_SIGNALS = dict(_XFUNC.caregiving_signals)
                # Only trigger caregiving search for relevant primary functions.
                # A technology student shouldn't get babysitter recommendations.
                _caregiving_funcs = set(_PIPE.caregiving_functions)
                has_caregiving = False
                if chosen_function in _caregiving_funcs:
                    for signal, expansions in _CAREGIVING_SIGNALS.items():
                        if signal in resume_lower:
                            search_keywords.extend(expansions)
                            has_caregiving = True

                search_keywords = list(dict.fromkeys(search_keywords))[:20]

                from recommender.retrieve.retriever import _stream_full_parquet

                # When caregiving signals are present (children, family, mentoring),
                # babysitting/nanny jobs live under "other" and "hr" — not just
                # education/healthcare. Search ALL functions to catch them.
                # Use targeted keywords — "childcare" alone matches hundreds of
                # "Residential Childcare" (UK social care) jobs, so prefer specific terms.
                if has_caregiving and xfunc in ("education", "healthcare", "social-service"):
                    # Search per-function parquets directly (1-20MB each, instant load)
                    # instead of streaming the 21GB full parquet.
                    # Babysitter/nanny jobs are sparse (<0.02% of rows) so full
                    # streaming scans every row group — 6 scans = 700s per student.
                    # Search the full 21GB dataset for babysitter/nanny titles.
                    # These jobs are scattered across "other", "education", "hr" etc.
                    # Use v2 with no function filter + targeted title keywords.
                    _sitter_keywords = list(_XFUNC.sitter_keywords)
                    # Search multiple functions since babysitter jobs are scattered
                    xjobs = []
                    seen_ids = set()
                    for _cf in ("other", "education", "healthcare"):
                        _batch = retrieve_from_subset(
                            function=_cf,
                            student_skills=extracted_skills,
                            top_k=4,
                        )
                        for _j in _batch:
                            _jid = _j.get("id", "") or _j.get("url", "")
                            if _jid not in seen_ids:
                                seen_ids.add(_jid)
                                _j["_cross_function"] = True
                                _j["_source_function"] = _j.get("function", _cf)
                                xjobs.append(_j)
                    # Broader childcare keywords
                    _care_kw = list(_XFUNC.care_keywords)
                    for _cf in ("education", "healthcare", "social-service"):
                        _batch = retrieve_from_subset(
                            function=_cf,
                            student_skills=extracted_skills,
                            top_k=3,
                        )
                        for _j in _batch:
                            _jid = _j.get("id", "") or _j.get("url", "")
                            if _jid not in seen_ids:
                                seen_ids.add(_jid)
                                _j["_cross_function"] = True
                                _j["_source_function"] = _j.get("function", _cf)
                                xjobs.append(_j)
                    xjobs = xjobs[:12]
                else:
                    xjobs = _stream_full_parquet(
                        function_labels={xfunc},
                        keyword_filters=search_keywords,
                        top_k=5,
                        include_all_levels=False,
                    )

                # Fall back to skill-vector retrieval if full-text returns nothing
                if not xjobs:
                    xjobs = retrieve_jds(xfunc, "Entry", extracted_skills, top_k=3)

                xjobs = filter_job_records(xjobs, candidate_function=xfunc)
                for xj in xjobs:
                    xj["_cross_function"] = True
                    xj["_source_function"] = xfunc
                if xjobs:
                    xranked = rank_jobs(
                        jobs=xjobs,
                        student_intent=student_intent,
                        extracted_skills=extracted_skills,
                        experience_skills=skill_profile["experience_skills"],
                        chosen_function=xfunc,
                        chosen_subdomain="",
                        lane=lane_used,
                        top_k=2,
                    )
                    cross_function_jobs.extend(xranked[:2])
            except Exception:
                pass

        # Quality gate: reject cross-function noise.
        # With zero goal/study alignment, a cross-function job is only
        # relevant if (a) it came from a caregiving-triggered search
        # (healthcare/education — matches hidden experience like mentoring
        # or childcare) AND (b) it has at least some skill overlap (>0.05).
        # Generic skill-hint jobs (arts-media, hospitality) with no goal
        # alignment are noise — e.g., Wine Steward with "customer service"
        # matching but zero career relevance.
        _NOISE_FUNCTIONS = set(_PIPE.noise_functions)
        _filtered_xf = []
        for _xj in cross_function_jobs:
            _bd = _xj.get("job_score_breakdown", {})
            _goal = _bd.get("goal_alignment", 0)
            _study = _bd.get("study_alignment", 0)
            _skill = _bd.get("skill_overlap_raw", _bd.get("skill_overlap", 0))
            _src = _xj.get("_source_function", "") or _xj.get("function", "")
            _has_intent = (_goal > 0.10 or _study > 0.15)
            _is_caregiving = (_src in ("healthcare", "education", "social-service", "other"))
            if _has_intent:
                _filtered_xf.append(_xj)
            elif _is_caregiving and _skill > 0.05:
                _filtered_xf.append(_xj)
            # else: no intent + not caregiving = noise → drop
        cross_function_jobs = _filtered_xf

        if cross_function_jobs:
            seen_ids = {job.get("url", "") or job.get("title", "") for job in ranked_jobs}
            for xj in cross_function_jobs:
                xid = xj.get("url", "") or xj.get("title", "")
                if xid not in seen_ids:
                    seen_ids.add(xid)
                    xj["_cross_function"] = True
                    ranked_jobs.append(xj)

    coach_notes = _default_coach_notes(chosen_function, gaps["core_gaps"], gaps["verify_gaps"])

    # ── Tiered job recommendations ──
    # ready_now: 3 primary + 2 cross-function, attainable now
    # aspirational: 3 primary + 2 cross-function, goal-aligned
    from recommender.llm.bridge_generator import generate_bridge, generate_skill_path

    def _make_entry(job, is_cross=False):
        jf = job.get("function", chosen_function) or chosen_function
        js = job.get("subdomain", chosen_subdomain) or chosen_subdomain
        e = {
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "url": job.get("url", ""),
            "fit": job.get("fit", 0),
            "job_type": job.get("job_type", ""),
            "subdomain": js,
            "function": jf,
            "why": _auto_explain(job.get("title", ""), jf, js, extracted_skills[:8], ideal_careers, job.get("job_score_breakdown")),
            "gaps": ", ".join(gaps["core_gaps"][:3]),
            "job_score_breakdown": job.get("job_score_breakdown", {}),
            "eligible": job.get("eligible", True),
            "eligibility_reason": job.get("eligibility_reason", ""),
            "cross_function": is_cross,
        }
        if is_cross:
            e["why"] = f"[{jf.title().replace('-',' ')} match] " + e["why"]
        return e

    # Split ranked_jobs into primary and cross-function
    primary_jobs = [j for j in ranked_jobs if not j.get("_cross_function", False)]
    cross_jobs = [j for j in ranked_jobs if j.get("_cross_function", False)]

    ready_now = []
    aspirational = []
    explore_jobs = []
    seen_titles = set()

    # Fill primary slots: guarantee at least 1 slot per candidate function.
    # Students with split signals (design + education) shouldn't get all
    # slots from one function just because its jobs rank slightly higher.
    func_slots_ready: dict[str, int] = {}
    func_slots_aspire: dict[str, int] = {}
    for cf in candidate_functions[:3]:
        func_slots_ready[cf] = 0
        func_slots_aspire[cf] = 0

    for job in primary_jobs:
        title = job.get("title", "")
        if title in seen_titles:
            continue
        bd = job.get("job_score_breakdown", {})
        _skill_ol = bd.get("skill_overlap_raw", bd.get("skill_overlap", 0))
        if _skill_ol < 0.05:
            continue
        entry = _make_entry(job)
        job_func = job.get("function", chosen_function) or chosen_function

        if job_func != chosen_function and ideal_careers:
            entry["bridge"] = generate_bridge(
                job_title=entry["title"], job_function=job_func,
                job_skills=job.get("skills", [])[:8], student_skills=extracted_skills[:8],
                ideal_careers=ideal_careers, aspirational_function=chosen_function)

        # Ready now: attainable jobs, max 1 per function until all functions have 1
        if len(ready_now) < 3 and bd.get("attainability", 0) >= 0.45:
            fn = job_func if job_func in func_slots_ready else chosen_function
            if fn not in func_slots_ready:
                func_slots_ready[fn] = 0
            # Allow if this function doesn't have a slot yet OR all functions have at least 1
            all_have_one = all(v >= 1 for v in func_slots_ready.values())
            if func_slots_ready.get(fn, 0) == 0 or all_have_one:
                func_slots_ready[fn] = func_slots_ready.get(fn, 0) + 1
                seen_titles.add(title)
                ready_now.append(entry)
                continue

        # Aspirational: max 1 per function until all functions have 1
        if len(aspirational) < 5:
            fn = job_func if job_func in func_slots_aspire else chosen_function
            if fn not in func_slots_aspire:
                func_slots_aspire[fn] = 0
            all_have_one = all(v >= 1 for v in func_slots_aspire.values())
            if func_slots_aspire.get(fn, 0) == 0 or all_have_one:
                func_slots_aspire[fn] = func_slots_aspire.get(fn, 0) + 1
                seen_titles.add(title)
                aspirational.append(entry)
                continue

        if len(ready_now) >= 5 and len(aspirational) >= 5:
            break

    # Tune cross-function jobs: sort by attainability for ready_now, goal for aspirational
    xf_ready = sorted(cross_jobs, key=lambda j: -j.get("job_score_breakdown", {}).get("attainability", 0))
    xf_aspire = sorted(cross_jobs, key=lambda j: -j.get("job_score_breakdown", {}).get("goal_alignment", 0))

    # Add 2 cross-function to ready_now, gated by skill overlap
    for job in xf_ready:
        if len(ready_now) >= 5:
            break
        title = job.get("title", "")
        if title in seen_titles:
            continue
        _bd = job.get("job_score_breakdown", {})
        _ol = _bd.get("skill_overlap_raw", _bd.get("skill_overlap", 0))
        if _ol < 0.08:
            continue
        seen_titles.add(title)
        ready_now.append(_make_entry(job, is_cross=True))

    # Add 2 cross-function to aspirational, gated by skill overlap
    for job in xf_aspire:
        if len(aspirational) >= 5:
            break
        title = job.get("title", "")
        if title in seen_titles:
            continue
        _bd = job.get("job_score_breakdown", {})
        _ol = _bd.get("skill_overlap_raw", _bd.get("skill_overlap", 0))
        if _ol < 0.08:
            continue
        seen_titles.add(title)
        aspirational.append(_make_entry(job, is_cross=True))

    # Remaining cross-function jobs → explore
    for job in cross_jobs:
        title = job.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        explore_jobs.append(_make_entry(job, is_cross=True))

    # Fill any empty primary slots from remaining primary jobs
    used = {j["title"] for j in ready_now + aspirational + explore_jobs}
    for job in primary_jobs:
        title = job.get("title", "")
        if title in used:
            continue
        if len(ready_now) >= 5 and len(aspirational) >= 5:
            break
        used.add(title)
        entry = _make_entry(job)
        bd = job.get("job_score_breakdown", {})
        if len(ready_now) < 3 and bd.get("attainability", 0) >= 0.45:
            ready_now.append(entry)
        elif len(aspirational) < 3:
            aspirational.append(entry)

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

    # ── LLM Reranking (Phase 6) ──────────────────────────────────
    # DeepSeek scores top-8 on skill_fit, experience_match, career_alignment.
    # Blend: 0.7 × deterministic + 0.3 × LLM. Cost: ~$0.001/student.
    if DEEPSEEK_KEY:
        try:
            from recommender.llm.pairwise_ranker import rerank_top_n, build_student_context
            _ctx = build_student_context({
                "function": chosen_function,
                "verified_skills": verified_skills,
                "ideal_careers": ideal_careers,
                "student_intent": student_intent,
            })
            ranked_jobs = rerank_top_n(
                ranked_jobs, _ctx, resume_text=resume_text, top_n=8, blend_weight=0.3,
            )
            # Propagate LLM scores from ranked_jobs to tier entries.
            # Match by first 8 words of title (handles whitespace/encoding diffs).
            def _title_key(t):
                return " ".join(str(t).lower().split()[:8])
            _by_key = {}
            for _j in ranked_jobs:
                _by_key[_title_key(_j.get("title", ""))] = _j
            for _tier in (ready_now, aspirational, explore_jobs):
                for _entry in _tier:
                    _src = _by_key.get(_title_key(_entry.get("title", "")))
                    if _src:
                        _entry["fit"] = _src.get("fit", _entry["fit"])
                        _bd = _src.get("job_score_breakdown", {})
                        if _bd.get("llm_score") is not None:
                            _entry["job_score_breakdown"] = _bd
        except Exception:
            pass

    # Legacy flat job list for backward compatibility
    jobs = ready_now + aspirational

    # Calibrate scores to 50-80% range while preserving spread.
    # Calibrate scores. Cross-function jobs get a smaller boost so their
    # honest (lower) scores reflect that they're exploratory, not primary matches.
    for j in jobs:
        raw = j.get("fit", 0)
        if raw > 0:
            boost = XF_BOOST if j.get("cross_function") else FIT_BOOST
            j["fit"] = min(round(raw + boost), 85)
    for j in ready_now:
        if j.get("fit", 0) > 0:
            boost = FIT_READY_BOOST if not j.get("cross_function") else 0
            j["fit"] = min(j["fit"] + boost, 88)
    for j in aspirational:
        if j.get("fit", 0) > 0:
            penalty = FIT_ASPIRE_PENALTY if not j.get("cross_function") else 0
            j["fit"] = max(j["fit"] - penalty, 50)

    # ── Phase 1a/1b: internship recommendations ──
        # Fix 1: minimum skill overlap gate (≥10% to avoid "Subway Sandwich Artist")
        # Fix 2: job title function-keyword scoring (relevance check without LLM)
        internships: list[dict] = []
        if _needs_internships:
            _func_pool = [
                j for j in ranked_jobs
                if not j.get("_cross_function")
                and str(j.get("level", "")).lower() in ("intern", "internship")
                and str(j.get("function", "")).lower() in (
                    chosen_function.lower(),
                    _SUBSET_FUNCTION_MAP.get(chosen_function.lower(), chosen_function.lower()),
                )
            ]
            # Fallback: broaden to all functions but gate with skill overlap + title relevance
            if not _func_pool:
                _func_pool = [
                    j for j in ranked_jobs
                    if not j.get("_cross_function")
                    and str(j.get("level", "")).lower() in ("intern", "internship")
                ]
            # Filter: require ≥10% skill overlap OR title matches function keywords
            _func_kw = _TITLE_KEYWORDS.get(chosen_function, [chosen_function.replace("-", " ")])
            _student_skill_set = {_norm_skill(s) for s in extracted_skills[:30]}
            _filtered = []
            for _j in _func_pool:
                _title = str(_j.get("title", "")).lower()
                _raw = _j.get("skills")
                _job_skills = [s for s in (_raw if isinstance(_raw, list) else []) if isinstance(s, str)]
                _job_set = {_norm_skill(s) for s in _job_skills}
                _overlap = len(_student_skill_set & _job_set) / max(1, len(_job_set))
                _title_match = any(kw.lower() in _title for kw in _func_kw)
                if _overlap >= 0.10 or _title_match:
                    _j["_intern_quality"] = _overlap + (0.3 if _title_match else 0)
                    _filtered.append(_j)
            _filtered.sort(key=lambda j: -j.get("_intern_quality", 0))
            # Safety net: if nothing passes, take top attainable intern-level jobs
            if not _filtered:
                _filtered = sorted(
                    [j for j in ranked_jobs if not j.get("_cross_function")
                     and str(j.get("level", "")).lower() in ("intern", "internship")],
                    key=lambda j: -j.get("job_score_breakdown", {}).get("attainability", 0)
                )[:10]
            # Last resort: any attainable job
            if not _filtered:
                _filtered = sorted(
                    [j for j in ranked_jobs if not j.get("_cross_function")],
                    key=lambda j: -j.get("job_score_breakdown", {}).get("attainability", 0)
                )[:8]
            _intern_seen = set()
            for _ij in _filtered:
                _title = _ij.get("title", "")
                if _title in _intern_seen:
                    continue
                _intern_seen.add(_title)
                _ij_entry = _make_entry(_ij)
                _ij_entry["internship"] = True
                _raw_fit = _ij_entry.get("fit", 0)
                if _raw_fit > 0:
                    _ij_entry["fit"] = min(round(_raw_fit + FIT_BOOST + FIT_READY_BOOST), 88)
                internships.append(_ij_entry)
                if len(internships) >= 5:
                    break

    return {
        "resume": resume_text.strip(),
        "function": chosen_function,
        "confidence": max(50, min(95, best["match_pct"] + CONFIDENCE_BOOST)),
        "experience_months": experience_months,
        "experience_level": experience_level,
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
        "related": [
            {"skill": s if isinstance(r, tuple) else r.get("skill",""),
             "pmi": round(sc if isinstance(r, tuple) else r.get("pmi",0), 2)}
            for r in related_skills
            for s, sc in [(r[0], r[1]) if isinstance(r, tuple) else (r.get("skill",""), r.get("pmi",0))]
        ],
        "related_skills": [
            {"skill": s if isinstance(r, tuple) else r.get("skill",""),
             "pmi": round(sc if isinstance(r, tuple) else r.get("pmi",0), 2)}
            for r in related_skills
            for s, sc in [(r[0], r[1]) if isinstance(r, tuple) else (r.get("skill",""), r.get("pmi",0))]
        ],
        "alternatives": [{"function": a["function"], "pct": a["match_pct"]} for a in best.get("alternatives", [])[:5]],
        "coach_notes": coach_notes,
        "jobs": jobs,
        "ready_now": ready_now,
        "aspirational": aspirational,
        "internships": internships,
        "student_stage": student_stage or "",
        "stage_label": stage_label,
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
        # Diagnostics for intent vs classifier conflict and cross-function exploration
        "classifier_top_function": best.get("signal_breakdown", {}).get("classifier", {}),
        "cross_functions_explored": [j.get("_source_function", "") for j in ranked_jobs if j.get("_cross_function")],
        "cross_function_count": len([j for j in ranked_jobs if j.get("_cross_function")]),
        "cross_function_jobs": [
            {"title": j.get("title",""), "function": j.get("_source_function",""), "fit": j.get("fit",0)}
            for j in ranked_jobs if j.get("_cross_function")
        ],
        "explore_jobs": explore_jobs,  # cross-func + wrong-func jobs, separate tier
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
    parser.add_argument("--student-stage", default=None,
                        choices=["high_school", "college_freshman", "college_senior"],
                        help="Override auto-detected student lifecycle stage.")
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
        student_stage=args.student_stage,
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

    if result.get("stage_label"):
        print()
        print("=" * 60)
        print("STUDENT STAGE")
        print("=" * 60)
        print(f"  {result['stage_label']} ({result.get('student_stage', '')})")

    if result.get("internships"):
        print()
        print("=" * 60)
        print("INTERNSHIPS")
        print("=" * 60)
        for idx, job in enumerate(result["internships"], 1):
            print(f"  {idx}. {job['title']} [{job.get('fit', 0)}%]")
            if job.get("company"):
                print(f"     @ {job['company']}")
            if job.get("why"):
                print(f"     Why: {job['why']}")
            if job.get("url"):
                print(f"     {job['url'][:100]}")


if __name__ == "__main__":
    main()
