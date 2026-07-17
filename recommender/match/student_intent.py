from __future__ import annotations

import re
from collections import Counter

from recommender.config import get_student_intent
from recommender.extract.section_parser import parse_resume_sections
from recommender.match.subdomains import extract_study_level, top_subdomains

_CFG = get_student_intent()
_DOMAIN_KEYWORDS: dict[str, set[str]] = {k: set(v) for k, v in _CFG.domain_keywords.items()}
_MAJOR_TO_FUNCTION: dict[str, str] = dict(_CFG.major_to_function)
_ROLE_KEYWORDS: set[str] = set(_CFG.role_keywords)


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    words = cleaned.split()
    ngrams = words[:]
    ngrams.extend(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
    ngrams.extend(
        f"{words[i]} {words[i + 1]} {words[i + 2]}" for i in range(len(words) - 2)
    )
    return ngrams


def _keyword_signal(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    scores: dict[str, float] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            score += counts.get(keyword, 0)
        if score > 0:
            scores[domain] = score
    total = sum(scores.values()) or 1.0
    return {domain: value / total for domain, value in scores.items()}


def _classifier_signal(text: str) -> dict[str, float]:
    if len(text.split()) < 3:
        return {}
    try:
        from recommender.match.ensemble_matcher import _classifier_probas

        return _classifier_probas(text)
    except Exception:
        return {}


def _blend(*signals: dict[str, float]) -> dict[str, float]:
    blended: dict[str, float] = {}
    for signal in signals:
        for key, value in signal.items():
            blended[key] = blended.get(key, 0.0) + value
    total = sum(blended.values()) or 1.0
    return {key: value / total for key, value in blended.items()}


def _extract_roles(text: str) -> list[str]:
    roles = []
    lowered = text.lower()
    for role in sorted(_ROLE_KEYWORDS):
        if role in lowered:
            roles.append(role)
    return roles


def _derive_study_domains(text: str) -> list[str]:
    signal = _blend(_keyword_signal(text), _classifier_signal(text))
    ranked = sorted(signal.items(), key=lambda item: -item[1])
    return [domain for domain, score in ranked[:3] if score >= 0.12]


def build_experience_signal(resume_text: str, resume_sections: dict[str, str] | None = None) -> dict[str, float]:
    sections = resume_sections or parse_resume_sections(resume_text)
    experience_text = "\n".join(
        part
        for key, part in sections.items()
        if key in {"experience", "leadership"}
    ).strip()
    if not experience_text:
        return {}
    return _blend(_keyword_signal(experience_text), _classifier_signal(experience_text))


def build_student_intent_profile(
    resume_text: str,
    linkedin_text: str = "",
    headline_text: str = "",
    career_goals_text: str = "",
    smart_goals_text: str = "",
    hope_to_gain_text: str = "",
    ideal_career_text: str = "",
    study_text: str = "",
    resume_sections: dict[str, str] | None = None,
) -> dict:
    combined_sections = parse_resume_sections(f"{linkedin_text}\n{resume_text}")
    if resume_sections:
        combined_sections.update({k: v for k, v in resume_sections.items() if v})

    summary_text = "\n".join(
        value
        for value in [
            headline_text,
            career_goals_text,
            smart_goals_text,
            hope_to_gain_text,
            ideal_career_text,
            combined_sections.get("summary", ""),
        ]
        if value
    ).strip()
    education_text = "\n".join(
        value for value in [study_text, combined_sections.get("education", "")] if value
    ).strip()

    goal_signal = _blend(_keyword_signal(summary_text), _classifier_signal(summary_text))
    study_signal = _blend(_keyword_signal(education_text), _classifier_signal(education_text))
    # Boost study signal from major-to-function mapping
    education_lower = education_text.lower()
    for major, func in _MAJOR_TO_FUNCTION.items():
        if major in education_lower:
            study_signal[func] = study_signal.get(func, 0.0) + 0.25
    experience_signal = build_experience_signal(resume_text, combined_sections)

    goal_domains = [domain for domain, score in sorted(goal_signal.items(), key=lambda item: -item[1]) if score >= 0.12]
    study_domains = _derive_study_domains(education_text)
    goal_roles = _extract_roles(summary_text)
    study_program = education_text.splitlines()[0].strip() if education_text else ""
    goal_subdomains: list[str] = []
    for domain in goal_domains[:2]:
        for subdomain in top_subdomains(domain, (summary_text, 1.0), limit=2):
            if subdomain not in goal_subdomains:
                goal_subdomains.append(subdomain)
    study_subdomains: list[str] = []
    for domain in study_domains[:2]:
        for subdomain in top_subdomains(domain, (education_text, 1.0), limit=2):
            if subdomain not in study_subdomains:
                study_subdomains.append(subdomain)
    study_level = extract_study_level(education_text)
    # Fallback: check full resume text if education section wasn't parsed
    if not study_level:
        study_level = extract_study_level(resume_text)
    has_student_intent = bool(summary_text.strip() or education_text.strip())
    intent_confidence = round(
        max(
            max(goal_signal.values(), default=0.0),
            max(study_signal.values(), default=0.0),
        ),
        3,
    )
    intent_strength = round(
        min(
            1.0,
            intent_confidence
            + (0.1 if goal_subdomains else 0.0)
            + (0.1 if study_subdomains else 0.0)
            + (0.05 if goal_domains and study_domains and goal_domains[0] == study_domains[0] else 0.0),
        ),
        3,
    )

    return {
        "goal_domains": goal_domains,
        "goal_roles": goal_roles,
        "goal_subdomains": goal_subdomains,
        "study_domains": study_domains,
        "study_subdomains": study_subdomains,
        "study_program": study_program,
        "study_level": study_level,
        "intent_confidence": intent_confidence,
        "intent_strength": intent_strength,
        "has_student_intent": has_student_intent,
        "goal_signal": goal_signal,
        "study_signal": study_signal,
        "experience_signal": experience_signal,
        "sections": combined_sections,
        "intent_summary": {
            "headline": headline_text.strip(),
            "summary_text": summary_text[:280],
            "study_text": education_text[:280],
        },
    }
