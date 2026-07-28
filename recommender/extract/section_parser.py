from __future__ import annotations

import re
from datetime import datetime

_SECTION_ALIASES = {
    "summary": "summary",
    "objective": "summary",
    "about": "summary",
    "education": "education",
    "education and training": "education",
    "experience": "experience",
    "work experience": "experience",
    "internship / work experience": "experience",
    "internship/work experience": "experience",
    "internships": "experience",
    "internship": "experience",
    "work history": "experience",
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


# ── Experience years extraction ──────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_DATE_RANGE_RE = re.compile(
    r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\.?\s+(\d{4}))"
    r"\s*[-–—to]+\s*"
    r"(?:(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\.?\s+(\d{4}))"
    r"|(Present|Current|Now|Ongoing))",
    re.IGNORECASE,
)

_YEAR_ONLY_RE = re.compile(
    r"(\d{4})\s*[-–—to]+\s*(\d{4}|Present|Current|Now|Ongoing)",
    re.IGNORECASE,
)


def _parse_month_year(text: str) -> tuple[int, int] | None:
    """Parse 'Jan 2023' or 'September 2024' into (month, year)."""
    parts = text.strip().split()
    if len(parts) != 2:
        return None
    month_name, year_str = parts[0].lower().rstrip("."), parts[1]
    month = _MONTH_MAP.get(month_name)
    if not month:
        return None
    try:
        year = int(year_str)
        if 1990 <= year <= 2030:
            return (month, year)
    except ValueError:
        pass
    return None


def _months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    """Compute months between (month, year) tuples."""
    return (end[1] - start[1]) * 12 + (end[0] - start[0])


def extract_experience_months(text: str) -> int:
    """Extract total professional experience in months from resume text.

    Scans the experience/leadership sections for date ranges like
    'Jan 2023 - Dec 2024' or '2022 - Present' and sums them.
    """
    now = datetime.now()
    current_month_year = (now.month, now.year)
    total_months = 0

    # Try month-year patterns first (more precise)
    for match in _DATE_RANGE_RE.finditer(text):
        start_str = match.group(0).split("-")[0].strip().split("–")[0].split("—")[0].strip()
        start = _parse_month_year(start_str)
        if not start:
            continue

        end_str = match.group(0)
        # Find the end part after the dash
        for sep in [" - ", " – ", " — ", " to ", "-", "–", "—"]:
            if sep in end_str:
                end_part = end_str.split(sep, 1)[1].strip()
                break
        else:
            end_part = ""

        if not end_part:
            continue

        if end_part.lower() in ("present", "current", "now", "ongoing"):
            end = current_month_year
        else:
            end = _parse_month_year(end_part)
            if not end:
                continue

        months = _months_between(start, end)
        if 0 < months <= 120:  # cap at 10 years
            total_months += months

    # Fall back to year-only patterns if no month-year found
    if total_months == 0:
        for match in _YEAR_ONLY_RE.finditer(text):
            try:
                start_year = int(match.group(1))
                end_str = match.group(2)
                if end_str.lower() in ("present", "current", "now", "ongoing"):
                    end_year = now.year
                else:
                    end_year = int(end_str)
                months = (end_year - start_year) * 12
                if 0 < months <= 120:
                    total_months += months
            except (ValueError, IndexError):
                continue

    return total_months


def infer_experience_level(total_months: int) -> str:
    """Infer career level from total months of experience.

    Returns: 'none' (<3mo), 'entry' (<24mo), 'mid' (<60mo), 'senior' (60+mo)
    """
    if total_months < 3:
        return "none"
    if total_months < 24:
        return "entry"
    if total_months < 60:
        return "mid"
    return "senior"


def extract_experience_from_sections(sections: dict[str, str]) -> tuple[int, str]:
    """Extract total experience months and level from parsed resume sections.

    Only counts the 'experience' section (real jobs/internships), NOT
    leadership/volunteer sections. Student volunteer work from high school
    should not inflate professional experience estimates.
    Returns (total_months, level_string).
    """
    # Only count the experience section — leadership/volunteer is student activity, not professional experience
    experience_text = sections.get("experience", "").strip()

    if not experience_text:
        return (0, "none")

    total_months = extract_experience_months(experience_text)
    level = infer_experience_level(total_months)
    return (total_months, level)

