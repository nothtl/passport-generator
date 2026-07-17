from __future__ import annotations

import math
import os
import re
from collections import Counter

from recommender.match.subdomains import canonical_domain, score_subdomains
from recommender.config import get_ranking, get_subdomains
from recommender.retrieve.retriever import _SUBSET_FUNCTION_MAP

# ── Embedding similarity (cached sentence-transformer) ──────────────
_CFG_RANK = get_ranking()
_SUB = get_subdomains()
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        if os.getenv("RECOMMENDER_DISABLE_EMBEDDINGS", "").strip() == "1":
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            return None
    return _embedder


_resume_vec_cache: dict[int, object] = {}


def _batch_embedding_similarity(job_texts: list[str], resume_text: str) -> list[float]:
    """Batch-encode all jobs at once for speed. Returns list of [0,1] scores."""
    embedder = _get_embedder()
    if embedder is None or not resume_text or not job_texts:
        return [0.0] * len(job_texts)
    try:
        import numpy as np
        resume_key = hash(resume_text[:500])
        if resume_key not in _resume_vec_cache:
            _resume_vec_cache.clear()
            _resume_vec_cache[resume_key] = embedder.encode(
                [resume_text[:3000]], convert_to_numpy=True, show_progress_bar=False)[0]
        resume_vec = _resume_vec_cache[resume_key]

        truncated = [t[:2000] for t in job_texts]
        job_vecs = embedder.encode(truncated, convert_to_numpy=True, show_progress_bar=False)

        scores = []
        for job_vec in job_vecs:
            dot = float(job_vec.dot(resume_vec))
            norm = float(job_vec.dot(job_vec) ** 0.5 * resume_vec.dot(resume_vec) ** 0.5)
            if norm == 0:
                scores.append(0.0)
            else:
                sim = dot / norm
                scores.append(max(0.0, min(1.0, (sim + 0.3) / 1.3)))
        return scores
    except Exception:
        return [0.0] * len(job_texts)


_DOMAIN_EXPANSIONS: dict = dict(_SUB.domain_expansions)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokenize(text: str) -> set[str]:
    cleaned = _normalize(text)
    if not cleaned:
        return set()
    words = cleaned.split()
    tokens = set(words)
    tokens.update(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
    return tokens


def _overlap_score(job_text: str, phrases: list[str]) -> float:
    if not phrases:
        return 0.0
    tokens = _tokenize(job_text)
    matches = 0.0
    for phrase in phrases:
        p = phrase.lower().strip()
        if not p:
            continue
        if p in tokens or any(part in tokens for part in p.split()):
            matches += 1.0
    return min(1.0, matches / max(1, len(phrases)))


def _expand_terms(terms: list[str]) -> list[str]:
    expanded = []
    for term in terms:
        expanded.append(term)
        expanded.extend(_DOMAIN_EXPANSIONS.get(term, []))
    return sorted({term for term in expanded if term})


def _skill_overlap(job_skills: list[str], extracted_skills: list[str]) -> float:
    if job_skills is None or len(job_skills) == 0 or not extracted_skills:
        return 0.0
    job_norm = {_normalize(skill) for skill in job_skills}
    extracted_norm = {_normalize(skill) for skill in extracted_skills}
    overlap = len(job_norm & extracted_norm)
    return overlap / max(1, len(job_norm))


def _function_match(job_function: str, candidate_function: str) -> float:
    """Check if job function matches the candidate function, using synonym mapping.

    The parquet uses different labels than our pipeline (e.g., "engineering" vs "technology").
    We normalize both to the parquet vocabulary before comparing.
    Returns 1.0 for exact match, 0.5 for adjacent, 0.0 for mismatch.
    """
    if not job_function or not candidate_function:
        return 0.5  # neutral when unknown

    # Normalize both to parquet vocabulary
    job_norm = _SUBSET_FUNCTION_MAP.get(job_function.lower(), job_function.lower())
    cand_norm = _SUBSET_FUNCTION_MAP.get(candidate_function.lower(), candidate_function.lower())

    if job_norm == cand_norm:
        return 1.0

    # Adjacent functions that share career paths
    _ADJACENT = {frozenset(pair) for pair in _CFG_RANK.adjacent_functions}
    pair = frozenset({job_norm, cand_norm})
    if pair in _ADJACENT:
        return 0.5

    return 0.0


def _attainability(job: dict, extracted_skills: list[str]) -> tuple[float, list[str]]:
    title = job.get("title", "").lower()
    jd = job.get("jd_markdown", "").lower()
    penalties = 0.0
    reasons: list[str] = []

    if any(term in title for term in ["senior", "manager", "director", "principal"]):
        penalties += 0.45
        reasons.append("seniority mismatch")
    if any(term in title for term in ["physician", "doctor", "surgeon", "dvm", "vmd", "registered nurse"]):
        penalties += 0.55
        reasons.append("credential mismatch")

    if "bachelor" in jd and not any("bachelor" in skill.lower() for skill in extracted_skills):
        penalties += 0.1
        reasons.append("degree requirement")

    missing_required = 0
    extracted_norm = {_normalize(skill) for skill in extracted_skills}
    for skill in job.get("skills", [])[:8]:
        if _normalize(skill) not in extracted_norm:
            missing_required += 1
    if missing_required >= 5:
        penalties += 0.25
        reasons.append("multiple missing job skills")
    elif missing_required >= 3:
        penalties += 0.15
        reasons.append("some missing job skills")

    return max(0.0, 1.0 - penalties), reasons


def _job_type(goal_alignment: float, attainability: float) -> str:
    if goal_alignment >= 0.25 and attainability >= 0.55:
        return "goal"
    if attainability >= 0.65 and goal_alignment < 0.45:
        return "bridge"
    return "mixed"


def _subdomain_alignment(function: str, selected_subdomains: list[str], job_text: str) -> float:
    function = canonical_domain(function)
    if not function or not selected_subdomains:
        return 0.0
    scores = score_subdomains(function, job_text)
    if not scores:
        return 0.0
    return max(scores.get(subdomain, 0.0) for subdomain in selected_subdomains)


def rank_jobs(
    jobs: list[dict],
    student_intent: dict,
    extracted_skills: list[str],
    experience_skills: list[str],
    chosen_function: str = "",
    chosen_subdomain: str = "",
    lane: str = "fast",
    top_k: int = 10,
) -> list[dict]:
    goal_terms = _expand_terms(
        list(student_intent.get("goal_domains", [])) + list(student_intent.get("goal_roles", []))
    )
    study_terms = _expand_terms(list(student_intent.get("study_domains", [])))
    if student_intent.get("study_program"):
        study_terms.append(student_intent["study_program"])
    selected_subdomains = []
    if chosen_subdomain:
        selected_subdomains.append(chosen_subdomain)
    for subdomain in list(student_intent.get("goal_subdomains", [])) + list(student_intent.get("study_subdomains", [])):
        if subdomain and subdomain not in selected_subdomains:
            selected_subdomains.append(subdomain)

    ranked = []
    for job in jobs:
        job_text = f"{job.get('title', '')}\n{job.get('jd_markdown', '')}"
        goal_alignment = _overlap_score(job_text, goal_terms)
        study_alignment = _overlap_score(job_text, study_terms)
        subdomain_alignment = _subdomain_alignment(chosen_function, selected_subdomains, job_text)
        experience_alignment = max(
            _overlap_score(job_text, experience_skills),
            _skill_overlap(job.get("skills", []), experience_skills),
        )
        skill_overlap = _skill_overlap(job.get("skills", []), extracted_skills)
        function_match = _function_match(job.get("function", ""), chosen_function)
        attainability, attainability_reasons = _attainability(job, extracted_skills)
        quality_penalty = float(job.get("_quality_penalty", 0.0))

        # Weights: skill_overlap and function_match are the strongest signals.
                # goal/study alignment are secondary (they help when skills are sparse).
        weights = dict(_CFG_RANK.weights)

        fit = (
            weights["skill_overlap"] * skill_overlap
            + weights["function_match"] * function_match
            + weights["goal_alignment"] * goal_alignment
            + weights["study_alignment"] * study_alignment
            + weights["subdomain_alignment"] * subdomain_alignment
            + weights["experience_alignment"] * experience_alignment
            + weights["attainability"] * attainability
            - quality_penalty
        )
        fit = max(0.0, min(1.0, fit))
        enriched = dict(job)
        enriched["fit"] = round(fit * 100)
        enriched["job_type"] = _job_type(goal_alignment, attainability)
        sub_scores = score_subdomains(chosen_function, job_text) if chosen_function else {}
        if sub_scores:
            enriched["subdomain"] = max(sub_scores.items(), key=lambda item: item[1])[0]
        enriched["job_score_breakdown"] = {
            "skill_overlap": round(skill_overlap, 3),
            "function_match": round(function_match, 3),
            "goal_alignment": round(goal_alignment, 3),
            "study_alignment": round(study_alignment, 3),
            "subdomain_alignment": round(subdomain_alignment, 3),
            "experience_alignment": round(experience_alignment, 3),
            "attainability": round(attainability, 3),
            "quality_penalty": round(quality_penalty, 3),
        }
        if attainability_reasons:
            enriched["why"] = "; ".join(attainability_reasons)
        ranked.append(enriched)

    ranked.sort(key=lambda job: (-job["fit"], job.get("title", "")))
    return ranked[:top_k]


def build_gap_summary(
    ranked_jobs: list[dict],
    extracted_skills: list[str],
    implicit_skills: list[dict],
    top_n: int = 5,
) -> dict:
    extracted_norm = {_normalize(skill) for skill in extracted_skills}
    implicit_norm = {_normalize(item["skill"]) for item in implicit_skills}
    counts: Counter[str] = Counter()
    raw_names: dict[str, str] = {}

    for job in ranked_jobs[:top_n]:
        for skill in job.get("skills", [])[:10]:
            normed = _normalize(skill)
            if not normed or normed in extracted_norm:
                continue
            counts[normed] += 1
            raw_names.setdefault(normed, str(skill))

    ranked_missing = sorted(
        counts.items(),
        key=lambda item: (-item[1], raw_names[item[0]].lower()),
    )
    core_gaps: list[str] = []
    nice_to_have: list[str] = []
    bridge_gaps: list[str] = []
    stretch_gaps: list[str] = []
    verify_gaps: list[str] = []

    for normed, count in ranked_missing:
        label = raw_names[normed]
        if normed in implicit_norm:
            verify_gaps.append(label)
            continue
        if count >= 2:
            core_gaps.append(label)
        elif count == 1:
            source_job = next((job for job in ranked_jobs[:top_n] if label in job.get("skills", [])), None)
            if source_job and source_job.get("job_type") == "goal":
                stretch_gaps.append(label)
            else:
                bridge_gaps.append(label)
            nice_to_have.append(label)

    return {
        "core_gaps": core_gaps[:10],
        "bridge_gaps": bridge_gaps[:10],
        "stretch_gaps": stretch_gaps[:10],
        "nice_to_have_gaps": nice_to_have[:10],
        "verify_gaps": verify_gaps[:10],
    }
