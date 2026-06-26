from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .models import SkillGap

# Skills that are really requirements, not skills
_NON_SKILL_PATTERNS = [
    r"valid.?driver", r"driver.?s? license", r"background check",
    r"lifting.*pounds?", r"able to lift", r"stand(ing)? for",
    r"high school (diploma|ged)", r"bachelor", r"associate degree",
    r"authorized to work", r"must be (able|willing)",
    r"(state|california|texas|florida|new.york).*licens",
    r"pass(ing)?.*(drug|criminal|background)",
]

# Skills that only make sense if they appear across multiple JDs
_MIN_SKILL_FREQUENCY_RATIO = 0.05  # must appear in at least 5% of JDs

# Synonym/normalization map: canonical → [aliases]
_SKILL_SYNONYMS: dict[str, list[str]] = {
    "customer service": ["customer-service", "customer support", "client service",
                         "customer care", "guest service", "client support"],
    "data entry": ["data-entry", "data input", "record keeping", "record-keeping",
                   "records management", "clerical", "data collection"],
    "communication": ["communication skills", "interpersonal skills",
                     "written communication", "verbal communication", "people skills"],
    "project management": ["project-management", "project coordination",
                          "project coordinator", "project lead"],
    "program management": ["program-management", "program coordination",
                          "program coordinator", "program lead"],
    "time management": ["time-management", "multitasking", "multi-tasking",
                       "prioritization", "organization"],
    "problem solving": ["problem-solving", "critical thinking", "troubleshooting",
                       "analytical", "root cause analysis"],
    "teamwork": ["team work", "team-work", "team player", "team collaboration",
                "team-collaboration", "collaboration"],
    "leadership": ["team lead", "team leader", "supervisor", "supervisory"],
    "teaching": ["teaching skills", "instruction", "instructional", "training",
                "trainer", "facilitation"],
    "mentoring": ["mentor", "mentorship", "coaching", "coach"],
    "writing": ["writing skills", "written", "copywriting", "editing"],
    "sales": ["selling", "retail sales", "upselling", "cross-selling"],
    "inventory management": ["inventory", "inventory control", "stock management",
                            "stocking", "merchandise", "supply chain"],
    "scheduling": ["schedule management", "calendar management", "appointment booking",
                  "appointment scheduling", "shift planning"],
    "data analysis": ["data analytics", "analytics", "analysis", "data science",
                     "statistical analysis", "visualization"],
    "healthcare": ["health care", "patient care", "clinical", "medical"],
    "social media management": ["social media", "social media marketing",
                               "social platform", "social account"],
    "content creation": ["content creator", "content writing", "content development",
                        "content strategy"],
    "graphic design": ["graphic-design", "visual design", "design skills"],
    "public speaking": ["public-speaking", "presentation skills", "presenting"],
    "event planning": ["event-planning", "event coordination", "event management"],
    "budgeting & finance": ["budgeting", "budget management", "accounting",
                           "bookkeeping", "financial management"],
    "volunteer coordination": ["volunteer management", "volunteer recruitment",
                              "volunteer scheduling"],
    "community outreach": ["outreach", "community engagement", "community service"],
    "fundraising": ["fundraising", "fundraiser", "donor relations", "grant writing"],
    "logistics & driving": ["logistics", "delivery", "driving", "shipping", "dispatch"],
    "software & technical": ["software development", "programming", "coding",
                            "web development", "developer"],
    "certifications": ["certification", "certified", "licensed"],
    "trades & physical": ["construction", "carpentry", "plumbing", "electrical",
                         "hvac", "welding", "masonry"],
    "food service": ["food-service", "cooking", "culinary", "kitchen", "restaurant"],
    "manufacturing": ["production", "assembly", "fabrication", "machining"],
    "administrative": ["admin", "clerical", "office administration", "secretarial"],
    "childcare": ["child care", "daycare", "nanny", "babysitting"],
    "marketing": ["digital marketing", "branding", "marketing strategy", "market research"],
}


def _normalize_skill_name(raw: str) -> str:
    """Map a skill string to its canonical form using the synonym map."""
    s = raw.strip().lower()
    for canonical, aliases in _SKILL_SYNONYMS.items():
        if s == canonical.lower() or s in [a.lower() for a in aliases]:
            return canonical
    return s


def _clean_skill(raw: str) -> str | None:
    """Normalize a skill string: remove annotations, dedup, map to canonical."""
    s = raw.strip().lower()
    if not s or len(s) < 2 or len(s) > 40:
        return None
    # Remove extraction artifacts
    s = re.sub(r"\?\s*$", "", s)                 # trailing ?
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)          # parenthetical notes
    s = re.sub(r"\s+", " ", s).strip()
    # Normalize hyphens to spaces
    s = s.replace("-", " ")
    if not s or len(s) < 2:
        return None
    # Map to canonical form
    return _normalize_skill_name(s)


def _is_non_skill(skill: str) -> bool:
    for pat in _NON_SKILL_PATTERNS:
        if re.search(pat, skill):
            return True
    return False


def aggregate_skills(
    jds: list[dict[str, Any]],
    matched_skills: list[str],
    missing_skills: list[str],
    min_frequency: int | None = None,
) -> list[SkillGap]:
    freq: Counter[str] = Counter()
    for jd in jds:
        raw_skills = jd.get("skills", [])
        if isinstance(raw_skills, (list, tuple)) or hasattr(raw_skills, "__iter__"):
            for s in raw_skills:
                if isinstance(s, str):
                    cleaned = _clean_skill(s)
                    if cleaned and not _is_non_skill(cleaned):
                        freq[cleaned] += 1

    # Auto-compute floor: skill must appear in >= 5% of JDs, minimum 2
    if min_frequency is None:
        min_frequency = max(2, int(len(jds) * _MIN_SKILL_FREQUENCY_RATIO))

    # Ensure matched/missing skills are represented even if below threshold
    combined = set(s.lower() for s in matched_skills + missing_skills)
    for skill_name in combined:
        if skill_name not in freq:
            freq[skill_name] = 0

    gaps = []
    for skill_name, count in freq.most_common():
        # Drop skills that appear too rarely (unless it's a matched/missing role skill)
        is_role_skill = skill_name.lower() in combined
        if not is_role_skill and count < min_frequency:
            continue
        student_has = skill_name in combined and skill_name in set(
            s.lower() for s in matched_skills
        )
        gaps.append(SkillGap(
            skill=skill_name,
            frequency=count,
            importance=0.0,
            student_has=student_has,
        ))
    return gaps


def aggregate_passport(jds: list[dict[str, Any]]) -> dict[str, float]:
    if not jds:
        return {}

    scores_collected = {p: [] for p in ["EC", "GC", "RFF", "CR", "CT", "CI"]}

    for jd in jds:
        try:
            result = _score_single_jd(jd)
            for pillar, value in result.items():
                scores_collected[pillar].append(value)
        except Exception:
            continue

    averaged = {}
    for pillar, scores in scores_collected.items():
        if scores:
            averaged[pillar] = round(sum(scores) / len(scores), 1)
    return averaged


def _score_single_jd(jd: dict[str, Any]) -> dict[str, float]:
    from passport_agent_v2.pillars import score_ci, score_cr, score_ct, score_ec, score_gc, score_rff
    from passport_agent_v2.tools.ingest import StudentBundle
    from passport_agent_v2.tools.normalization import normalize_student_bundle, classify_evidence_span
    from passport_agent_v2.tools.semantic import SemanticConfig
    from passport_agent_v2.models import NormalizedStudentProfile

    jd_text = jd.get("jd_markdown", "") or ""
    title = jd.get("title", "") or ""
    skills = jd.get("skills", [])
    if isinstance(skills, (list, tuple)) or hasattr(skills, "__iter__"):
        skills = [s for s in skills if isinstance(s, str)]
    else:
        skills = []

    evidence_facts = []
    for para in _jd_paragraphs(jd_text):
        fact = classify_evidence_span(para, source="jd")
        evidence_facts.append(fact)

    combined_text = f"{jd_text} {' '.join(skills)}"

    profile = NormalizedStudentProfile(
        student_name=title,
        slug=title.lower().replace(" ", "_")[:40],
        evidence_facts=evidence_facts,
    )

    try:
        ec_result = score_ec(profile.normalized_fields)
    except Exception:
        ec_result = {"score": 50}

    try:
        ct_result = score_ct(profile.evidence_facts, combined_text, title)
    except Exception:
        ct_result = {"score": 50}

    try:
        ci_result = score_ci(profile.evidence_facts, combined_text, title)
    except Exception:
        ci_result = {"score": 50}

    try:
        cr_result = score_cr(profile.normalized_fields)
    except Exception:
        cr_result = {"score": 50}

    try:
        gc_result = score_gc(profile.normalized_fields)
    except Exception:
        gc_result = {"score": 50}

    try:
        rff_result = score_rff(profile.normalized_fields)
    except Exception:
        rff_result = {"score": 50}

    return {
        "EC": _extract_score(ec_result),
        "GC": _extract_score(gc_result),
        "RFF": _extract_score(rff_result),
        "CR": _extract_score(cr_result),
        "CT": _extract_score(ct_result),
        "CI": _extract_score(ci_result),
    }


def _extract_score(result: Any) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        return float(result.get("score", 50))
    return 50.0


def _jd_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if not parts:
        parts = [text.strip()]
    return parts
