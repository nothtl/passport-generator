from __future__ import annotations

import re
from collections import Counter

_ALIASES = {
    "arts-media": "design",
}

SUBDOMAIN_TAXONOMY: dict[str, dict[str, set[str]]] = {
    "education": {
        "classroom-support": {
            "classroom",
            "teacher assistant",
            "teaching assistant",
            "lesson",
            "instruction",
            "school",
            "education",
            "childhood education",
        },
        "youth-programs": {
            "after school",
            "after-school",
            "youth program",
            "camp counselor",
            "mentor",
            "mentoring",
            "student support",
            "youth development",
        },
        "tutoring": {
            "tutor",
            "tutoring",
            "peer tutor",
            "academic support",
        },
        "higher-ed-support": {
            "college",
            "university",
            "student affairs",
            "campus",
            "higher education",
            "admissions",
        },
        "community-education": {
            "community education",
            "workshop",
            "facilitator",
            "adult learning",
            "community outreach",
        },
    },
    "healthcare": {
        "clinical-support": {
            "medical assistant",
            "home health",
            "patient care",
            "clinical",
            "patient",
            "hospital",
            "clinic",
            "intake",
            "caregiver",
        },
        "public-health": {
            "public health",
            "community health",
            "health education",
            "wellness",
            "prevention",
            "outreach",
        },
        "health-admin": {
            "health administration",
            "scheduling",
            "billing",
            "records",
            "front desk",
            "reception",
        },
        "research": {
            "research",
            "laboratory",
            "lab",
            "clinical research",
            "data collection",
            "study coordinator",
        },
        "pre-med-shadowing": {
            "pre med",
            "premed",
            "shadowing",
            "medical school",
            "physician",
            "doctor",
        },
    },
    "technology": {
        "it-support": {
            "technical support",
            "it support",
            "help desk",
            "troubleshoot",
            "hardware",
            "software issues",
            "desktop support",
            "user support",
        },
        "software": {
            "software",
            "developer",
            "api",
            "backend",
            "frontend",
            "full stack",
            "web app",
            "react",
            "fastapi",
        },
        "systems": {
            "systems",
            "network",
            "infrastructure",
            "cloud",
            "aws",
            "linux",
            "server",
            "devops",
        },
        "engineering": {
            "engineer",
            "engineering",
            "mechanical",
            "electrical",
            "cad",
            "embedded",
            "firmware",
        },
        "robotics-manufacturing": {
            "robotics",
            "automation",
            "manufacturing",
            "arduino",
            "mechatronics",
            "plc",
            "cnc",
        },
    },
    "design": {
        "graphic-design": {
            "graphic design",
            "illustrator",
            "photoshop",
            "branding",
            "layout",
            "poster",
            "visual design",
        },
        "photo-video": {
            "photography",
            "videography",
            "video editing",
            "lightroom",
            "premiere",
        },
        "content-creation": {
            "content creator",
            "social media",
            "copywriting",
            "ugc",
            "creator",
        },
        "creative-education": {
            "art teacher",
            "arts education",
            "creative workshop",
            "youth arts",
        },
        "marketing-creative": {
            "marketing",
            "campaign",
            "brand strategy",
            "digital marketing",
            "social media marketing",
        },
    },
    "social-service": {
        "youth-support": {
            "youth support",
            "peer support",
            "student support",
            "case management",
            "mentor",
            "counseling",
        },
        "community-support": {
            "community outreach",
            "advocacy",
            "social work",
            "community support",
            "outreach",
        },
    },
}

_DEFAULT_SUBDOMAIN = {
    "education": "classroom-support",
    "healthcare": "clinical-support",
    "technology": "software",
    "design": "graphic-design",
    "social-service": "community-support",
}


def canonical_domain(domain: str) -> str:
    return _ALIASES.get(domain, domain)


def subdomains_for_domain(domain: str) -> list[str]:
    domain = canonical_domain(domain)
    return list(SUBDOMAIN_TAXONOMY.get(domain, {}))


def default_subdomain(domain: str) -> str:
    domain = canonical_domain(domain)
    return _DEFAULT_SUBDOMAIN.get(domain, "")


def extract_study_level(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["phd", "ph.d", "doctor of philosophy", "doctoral"]):
        return "doctoral"
    if any(term in lowered for term in ["master", "m.s.", "m.a.", "mba"]):
        return "master"
    if any(term in lowered for term in ["bachelor", "b.s.", "b.a.", "bs ", "ba "]):
        return "bachelor"
    if any(term in lowered for term in ["associate", "a.a.", "a.s."]):
        return "associate"
    if "certificate" in lowered:
        return "certificate"
    if "high school" in lowered:
        return "high-school"
    return ""


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
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


def score_subdomains(domain: str, text: str) -> dict[str, float]:
    domain = canonical_domain(domain)
    taxonomy = SUBDOMAIN_TAXONOMY.get(domain, {})
    if not taxonomy or not text.strip():
        return {}
    counts = Counter(_tokenize(text))
    scored: dict[str, float] = {}
    for subdomain, keywords in taxonomy.items():
        score = 0.0
        for keyword in keywords:
            score += counts.get(keyword.lower(), 0)
            if keyword.lower() in text.lower():
                score += 0.5
        if score > 0:
            scored[subdomain] = score
    total = sum(scored.values()) or 1.0
    return {key: value / total for key, value in scored.items()}


def merge_subdomain_scores(weighted_scores: list[tuple[dict[str, float], float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for scores, weight in weighted_scores:
        for key, value in scores.items():
            merged[key] = merged.get(key, 0.0) + (value * weight)
    total = sum(merged.values()) or 1.0
    return {key: value / total for key, value in merged.items()}


def top_subdomains(domain: str, *texts: tuple[str, float], limit: int = 3) -> list[str]:
    weighted: list[tuple[dict[str, float], float]] = []
    for entry in texts:
        if not entry:
            continue
        text = entry[0]
        weight = entry[1]
        if text and weight > 0:
            weighted.append((score_subdomains(domain, text), weight))
    merged = merge_subdomain_scores(weighted)
    ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    values = [name for name, score in ranked[:limit] if score >= 0.15]
    return values or ([default_subdomain(domain)] if default_subdomain(domain) else [])


def choose_subdomain(domain: str, weighted_texts: dict[str, tuple[str, float]]) -> tuple[str, float, list[str]]:
    domain = canonical_domain(domain)
    merged = merge_subdomain_scores(
        [
            (score_subdomains(domain, text), weight)
            for text, weight in weighted_texts.values()
            if text and weight > 0
        ]
    )
    ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    if ranked:
        top_name, top_score = ranked[0]
        alternatives = [name for name, _score in ranked[1:4]]
        return top_name, round(top_score, 3), alternatives
    fallback = default_subdomain(domain)
    return fallback, 0.0 if not fallback else 0.3, []

