"""College matcher — main entry point for college recommendations.

``match_colleges`` ties together the :class:`ScorecardClient` and
:class:`CollegeRanker` to produce tiered college recommendations
(reach / match / safety) for a student.

The student profile is expected to follow the structure produced by
``recommender.match.student_intent.build_student_intent_profile``,
augmented with preferred location, size, and study level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recommender.college.college_ranker import CollegeRanker, CollegeResult, _classify_tier
from recommender.college.scorecard_client import ScorecardClient
from recommender.config import get_college

# ── Size buckets (based on Carnegie size setting) ──────────────────

_SIZE_BUCKETS = {
    "small": (0, 5_000),
    "medium": (5_000, 15_000),
    "large": (15_000, 999_999),
}


@dataclass
class StudentCollegeProfile:
    """Lightweight profile for college matching.

    Most fields are sourced from the student intent profile, but
    callers can also construct this directly.
    """

    skills: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)  # career function labels
    preferred_state: str = ""
    preferred_size: str = ""  # "small" | "medium" | "large" | ""
    study_level: str = ""  # "certificate" | "associate" | "bachelor" | ...
    gpa: float | None = None
    max_budget: float | None = None


def _build_profile_from_intent(student_profile: dict[str, Any]) -> StudentCollegeProfile:
    """Convert a student_intent profile dict into :class:`StudentCollegeProfile`."""
    # Extract career functions from goal/study domains
    interests: list[str] = []
    for key in ("goal_domains", "study_domains"):
        domains = student_profile.get(key, [])
        if domains:
            interests.extend(d for d in domains if d)

    # Skills — flatten from extracted_skills if available
    skills = student_profile.get("skills", [])
    if not skills:
        # Fallback: use goal_roles + subdomains as pseudo-skills
        skills = list(student_profile.get("goal_roles", []))
        skills.extend(student_profile.get("goal_subdomains", []))

    return StudentCollegeProfile(
        skills=skills,
        interests=interests,
        preferred_state=student_profile.get("preferred_state", ""),
        preferred_size=student_profile.get("preferred_size", ""),
        study_level=student_profile.get("study_level", ""),
        gpa=student_profile.get("gpa"),
        max_budget=student_profile.get("max_budget"),
    )


def _function_to_cip_codes(
    functions: list[str], cip_to_function: dict[str, str]
) -> list[str]:
    """Map career function labels back to CIP code prefixes.

    Given ``{"11": "technology", "14": "engineering"}`` and
    ``["technology"]`` returns ``["11"]``.
    """
    func_set = {f.lower() for f in functions}
    return [
        cip
        for cip, func in cip_to_function.items()
        if func.lower() in func_set
    ]


def _filter_by_size(
    schools: list[dict[str, Any]], preferred_size: str
) -> list[dict[str, Any]]:
    """Filter schools by preferred student body size bucket."""
    if not preferred_size:
        return schools
    lo, hi = _SIZE_BUCKETS.get(preferred_size.lower(), (0, 999_999))
    filtered = []
    for s in schools:
        size = s.get("student_size")
        if size is None:
            filtered.append(s)  # keep unknown-size schools
            continue
        if lo <= size <= hi:
            filtered.append(s)
    return filtered


def _compute_target_admission(gpa: float | None) -> float:
    """Estimate a target admission rate based on GPA.

    Higher GPA → student can target more selective schools → lower target rate.
    """
    if gpa is None:
        return 0.45  # neutral default
    if gpa >= 3.8:
        return 0.30
    if gpa >= 3.5:
        return 0.40
    if gpa >= 3.0:
        return 0.55
    if gpa >= 2.5:
        return 0.70
    return 0.85


def match_colleges(
    student_profile: dict[str, Any] | StudentCollegeProfile,
    top_k: int = 10,
    client: ScorecardClient | None = None,
    ranker: CollegeRanker | None = None,
) -> dict[str, Any]:
    """Main entry point — produce tiered college recommendations.

    Parameters
    ----------
    student_profile:
        Either a :class:`StudentCollegeProfile` or a dict following the
        ``student_intent`` profile structure (with optional additions:
        ``preferred_state``, ``preferred_size``, ``gpa``, ``max_budget``).
    top_k:
        Maximum number of colleges to return per tier (total ≤ 3 × top_k).
    client:
        Optional pre-constructed :class:`ScorecardClient`.  When ``None``,
        one is created using the environment API key.
    ranker:
        Optional pre-constructed :class:`CollegeRanker`.

    Returns
    -------
    dict with keys:
        - ``reach``:  list[CollegeResult]
        - ``match``:  list[CollegeResult]
        - ``safety``: list[CollegeResult]
        - ``total``:  int
        - ``error``:  str  (empty on success)
    """
    # ── Normalise profile ───────────────────────────────────────────
    if isinstance(student_profile, StudentCollegeProfile):
        profile = student_profile
    else:
        profile = _build_profile_from_intent(student_profile)

    # ── Config ──────────────────────────────────────────────────────
    cfg = get_college()
    cip_to_function: dict[str, str] = dict(cfg.get("cip_to_function", {}) or {})
    tier_cfg = cfg.get("tiers", {}) or {}
    ranker_cfg = cfg.get("ranker", {}) or {}

    # ── Client / ranker ──────────────────────────────────────────────
    if client is None:
        client = ScorecardClient()
    if ranker is None:
        weights = dict(ranker_cfg.get("weights", {})) or None
        max_cost = ranker_cfg.get("max_cost", 60_000)
        if profile.max_budget:
            max_cost = profile.max_budget
        target_admission = _compute_target_admission(profile.gpa)
        ranker = CollegeRanker(
            weights=weights or None,
            max_cost=float(max_cost),
            target_admission_rate=float(target_admission),
            min_earnings=float(ranker_cfg.get("min_earnings", 25_000)),
            max_earnings=float(ranker_cfg.get("max_earnings", 90_000)),
            preferred_state=profile.preferred_state or "",
        )

    # ── Determine CIP codes from interests ───────────────────────────
    cip_codes = _function_to_cip_codes(profile.interests, cip_to_function)

    # ── Fetch schools from Scorecard API ─────────────────────────────
    # Fetch without CIP filter (program-specific filtering needs API tuning).
    # Still return useful general college recommendations scored on cost,
    # admission rate, earnings, and size fit.
    all_schools: list[dict] = []
    err = ""
    # Query 1: preferred state (e.g. NY) — get local schools
    if profile.preferred_state:
        local, local_err = client.fetch_schools(state=profile.preferred_state, per_page=100, max_pages=2)
        if local_err:
            err = local_err
        else:
            all_schools.extend(local)
    # Query 2: national diversity — paginate deeper for geographic mix
    diverse, div_err = client.fetch_schools(per_page=100, max_pages=4)
    if div_err and not err:
        err = div_err
    all_schools.extend(diverse)

    # ── Graceful degradation ────────────────────────────────────────
    if err:
        return {
            "reach": [],
            "match": [],
            "safety": [],
            "total": 0,
            "error": err,
        }

    if not all_schools:
        return {
            "reach": [],
            "match": [],
            "safety": [],
            "total": 0,
            "error": "No colleges found matching the criteria.",
        }

    # ── Deduplicate by school id ────────────────────────────────────
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for s in all_schools:
        sid = str(s.get("id", s.get("name", "")))
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(s)

    # ── Size filter ─────────────────────────────────────────────────
    deduped = _filter_by_size(deduped, profile.preferred_size)

    # ── Rank ────────────────────────────────────────────────────────
    ranked = ranker.rank(deduped, profile.interests, cip_to_function)

    # ── Tier split using config thresholds ──────────────────────────
    reach_max = float(tier_cfg.get("reach", {}).get("admission_pct_max", 0.25))
    match_min = float(tier_cfg.get("match", {}).get("admission_pct_min", 0.25))
    match_max = float(tier_cfg.get("match", {}).get("admission_pct_max", 0.60))
    safety_min = float(tier_cfg.get("safety", {}).get("admission_pct_min", 0.60))

    reach: list[CollegeResult] = []
    match: list[CollegeResult] = []
    safety: list[CollegeResult] = []

    for result in ranked:
        # Override tier with config-based thresholds
        rate = result.admission_rate
        if rate is not None and rate > 0:
            if rate <= reach_max:
                tier = "reach"
            elif rate >= safety_min:
                tier = "safety"
            else:
                tier = "match"
        else:
            tier = "match"

        result.tier = tier
        if tier == "reach":
            reach.append(result)
        elif tier == "safety":
            safety.append(result)
        else:
            match.append(result)

    # Cap each tier
    reach = reach[:top_k]
    match = match[:top_k]
    safety = safety[:top_k]

    return {
        "reach": reach,
        "match": match,
        "safety": safety,
        "total": len(reach) + len(match) + len(safety),
        "error": "",
    }
