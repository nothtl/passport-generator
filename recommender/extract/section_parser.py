from __future__ import annotations

import re

_SECTION_ALIASES = {
    "summary": "summary",
    "objective": "summary",
    "about": "summary",
    "education": "education",
    "education and training": "education",
    "experience": "experience",
    "work experience": "experience",
    "employment": "experience",
    "leadership": "leadership",
    "leadership experiences": "leadership",
    "activities": "leadership",
    "volunteer": "leadership",
    "volunteering": "leadership",
    "research": "experience",
    "skills": "skills",
    "certifications": "skills",
    "certification": "skills",
    "awards": "leadership",
    "honors & awards": "leadership",
    "honors and awards": "leadership",
}


def _normalize_heading(line: str) -> str:
    text = line.strip().strip("#").strip().strip(":").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _match_heading(line: str) -> str | None:
    normalized = _normalize_heading(line)
    if not normalized:
        return None
    if normalized in _SECTION_ALIASES:
        return _SECTION_ALIASES[normalized]
    if line.lstrip().startswith("##"):
        return _SECTION_ALIASES.get(normalized)
    if normalized.upper() == normalized and len(normalized.split()) <= 4:
        return _SECTION_ALIASES.get(normalized)
    if re.fullmatch(r"[A-Z][A-Z\s&/]{2,30}", line.strip()):
        return _SECTION_ALIASES.get(normalized)
    return None


def parse_resume_sections(text: str) -> dict[str, str]:
    if not text:
        return {}

    sections: dict[str, list[str]] = {}
    current = "summary"
    sections[current] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = _match_heading(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        if line.strip():
            sections.setdefault(current, []).append(line.strip())

    return {
        key: "\n".join(lines).strip()
        for key, lines in sections.items()
        if any(part.strip() for part in lines)
    }

