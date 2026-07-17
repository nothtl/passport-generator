from __future__ import annotations

import re
from collections import Counter

from recommender.config import get_subdomains

_CFG = get_subdomains()
_ALIASES: dict = dict(_CFG.aliases)
SUBDOMAIN_TAXONOMY: dict[str, dict[str, set[str]]] = {
    domain: {sub: set(keywords) for sub, keywords in subs.items()}
    for domain, subs in _CFG.taxonomy.items()
}
_DEFAULT_SUBDOMAIN: dict = dict(_CFG.defaults)


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

