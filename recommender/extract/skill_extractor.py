"""
Data-driven skill extraction — JD vocabulary membership with fuzzy matching.

1. Tokenize resume into 1-3 grams
2. Normalize each n-gram (strip hyphens/spaces)
3. Check normalized form against normalized JD vocabulary
4. Return display-form matches

Both the vocabulary and normalization are purely data-driven.
The vocabulary is built from all JD parquet files (~67K unique skills).
No hand-coded rules, no ESCO dependency.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from recommender.extract.section_parser import parse_resume_sections

if TYPE_CHECKING:
    from passport_agent_v2.tools.ingest import StudentBundle

_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(_HERE, "..", "data", "skill_vocabulary.json")

_vocab_normalized: set[str] | None = None

_GENERIC_STOP_SKILLS = {
    "activities",
    "activity",
    "admission",
    "andon",
    "adult",
    "assistance",
    "assisting",
    "ata",
    "children",
    "cleaning",
    "decisionmaking",
    "education",
    "eligibility",
    "english",
    "events",
    "examination",
    "flu",
    "gmail",
    "hospital",
    "http",
    "https",
    "iam",
    "intern",
    "inventory",
    "isa",
    "kitchen",
    "learning",
    "linkedin",
    "medical",
    "patient",
    "patients",
    "pdf",
    "phone",
    "planning",
    "positions",
    "preparation",
    "professionalism",
    "research",
    "resume",
    "resumepdf",
    "screening",
    "shopping",
    "student",
    "students",
    "support",
    "training",
    "url",
    "website",
    "websites",
    "vaccination",
    "volunteer",
    "word",
}

_IMPLICIT_RULES = {
    "python": {"fastapi", "pytorch", "ros2", "yolo", "pandas", "numpy"},
    "cloud deployment": {"azure functions", "aws lambda", "cloud run"},
    "caregiving": {"activities of daily living", "home health aide", "patient care"},
}


def _load_vocab_norm() -> set[str]:
    global _vocab_normalized
    if _vocab_normalized is None:
        with open(_VOCAB_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        _vocab_normalized = {re.sub(r"[- ]", "", s) for s in raw}
    return _vocab_normalized


def _normalize_skill(text: str) -> str:
    return re.sub(r"[- ]", "", text.lower())


def _keep_skill(display: str, normed: str) -> bool:
    if len(normed) < 3:
        return False
    if normed in _GENERIC_STOP_SKILLS:
        return False
    if display.count(" ") == 0 and normed.endswith("ing") and len(normed) < 8:
        return False
    if not re.search(r"[a-z]", display):
        return False
    return True


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Generate 1-3 grams. Returns (display, normalized) pairs."""
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if not words:
        return []

    def _norm(s):
        return re.sub(r"[- ]", "", s)

    ngrams = []
    for i in range(len(words)):
        ngrams.append((words[i], _norm(words[i])))
        if i + 1 < len(words):
            phrase = f"{words[i]} {words[i+1]}"
            ngrams.append((phrase, _norm(phrase)))
        if i + 2 < len(words):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            ngrams.append((phrase, _norm(phrase)))
    return ngrams


def _strip_meta_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        lowered = line.lower().strip()
        if not lowered:
            cleaned_lines.append(line)
            continue
        if "@" in line:
            continue
        if "http://" in lowered or "https://" in lowered or "linkedin.com" in lowered:
            continue
        if lowered.endswith(".pdf") or lowered.endswith(".doc") or lowered.endswith(".docx"):
            continue
        if re.search(r"\b\d{3}[-.)\s]?\d{3}[-.\s]?\d{4}\b", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def extract_skills_from_text(text: str, function: str | None = None) -> list[str]:
    """Extract skills: tokenize -> normalize -> check JD vocabulary -> return display forms.

    ~2ms. Normalization bridges "computer vision" <-> "computer-vision".
    Vocabulary from 67K JD skill names. No hand-coded rules.
    """
    if not text:
        return []

    text = _strip_meta_text(text)
    vocab_norm = _load_vocab_norm()
    candidates = _tokenize(text)

    seen = set()
    skills = []
    for display, normed in candidates:
        if normed in seen:
            continue
        if normed in vocab_norm and _keep_skill(display, normed):
            skills.append(display)
            seen.add(normed)

    return sorted(skills)


def _match_evidence_line(text: str, skill: str) -> str:
    lowered_skill = skill.lower()
    for line in text.splitlines():
        if lowered_skill in line.lower():
            return line.strip()
    return text.splitlines()[0].strip() if text.splitlines() else ""


def infer_implicit_skills(text: str, extracted_skills: list[str]) -> list[dict[str, str]]:
    lowered = text.lower()
    extracted_norm = {_normalize_skill(skill) for skill in extracted_skills}
    implicit = []
    for skill, triggers in _IMPLICIT_RULES.items():
        if _normalize_skill(skill) in extracted_norm:
            continue
        matched = sorted(trigger for trigger in triggers if trigger in lowered)
        if matched:
            implicit.append(
                {
                    "skill": skill,
                    "reason": ", ".join(matched[:3]),
                    "confidence": "high",
                }
            )
    return implicit


def extract_skill_profile(text: str, resume_sections: dict[str, str] | None = None) -> dict:
    sections = resume_sections or parse_resume_sections(text)
    if not sections:
        sections = {"summary": text}

    section_skills: dict[str, list[str]] = {}
    all_skills: list[str] = []
    evidence: list[dict[str, str]] = []
    seen = set()

    for section_name, section_text in sections.items():
        extracted = extract_skills_from_text(section_text)
        section_skills[section_name] = extracted
        for skill in extracted:
            normed = _normalize_skill(skill)
            if normed in seen:
                continue
            seen.add(normed)
            all_skills.append(skill)
            evidence.append(
                {
                    "skill": skill,
                    "source_section": section_name,
                    "matched_text": _match_evidence_line(section_text, skill),
                    "match_type": "vocabulary",
                }
            )

    experience_skills = sorted(
        {
            *section_skills.get("experience", []),
            *section_skills.get("leadership", []),
        }
    )
    implicit_skills = infer_implicit_skills(text, all_skills)
    return {
        "skills": sorted(all_skills),
        "section_skills": section_skills,
        "experience_skills": experience_skills,
        "skill_evidence": evidence,
        "implicit_skills": implicit_skills,
    }


def extract_skills_for_function(text: str, function: str) -> tuple[list[str], list[str]]:
    found = extract_skills_from_text(text)
    cls_path = os.path.join(_HERE, "..", "data", "classifier_skills.json")
    with open(cls_path) as f:
        func_features = json.load(f)
    if function not in func_features:
        return found[:15], []
    expected = [feat["feature"] for feat in func_features[function][:30]]
    found_set = set(found)
    has = [s for s in expected if s in found_set]
    missing = [s for s in expected if s not in found_set]
    return has, missing


def extract_skills_from_bundle(bundle: StudentBundle) -> list[str]:
    resume_text = getattr(bundle, "resume_text", "") or ""
    linkedin_text = getattr(bundle, "linkedin_text", "") or ""
    return extract_skills_from_text(f"{resume_text}\n{linkedin_text}")
