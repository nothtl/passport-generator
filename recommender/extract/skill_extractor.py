"""
Data-driven skill extraction from the trained classifier.

Replaces hand-coded regex patterns, tech terms, and phrase mappings.
Skills are the top TF-IDF features per function from the ML model
trained on 2,484 real resumes. No hardcoded rules.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passport_agent_v2.tools.ingest import StudentBundle

_HERE = os.path.dirname(os.path.abspath(__file__))
_SKILLS_PATH = os.path.join(_HERE, "..", "data", "classifier_skills.json")

# Cache
_func_features: dict[str, list[dict]] | None = None
_all_skills: set[str] | None = None


def _load_skills() -> dict[str, list[dict]]:
    global _func_features
    if _func_features is None:
        with open(_SKILLS_PATH, encoding="utf-8") as f:
            _func_features = json.load(f)
    return _func_features


def _get_all_skills() -> set[str]:
    global _all_skills
    if _all_skills is None:
        skills = _load_skills()
        _all_skills = set()
        for features in skills.values():
            for f in features:
                _all_skills.add(f["feature"])
    return _all_skills


def _tokenize(text: str) -> list[str]:
    """Extract words and bigrams from text."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if len(w) > 1]
    # Unigrams + bigrams
    tokens = words.copy()
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")
    return tokens


def extract_skills_from_text(text: str, function: str | None = None) -> list[str]:
    """Extract skills from resume text using classifier features.

    If function is provided, only checks that function's features.
    Otherwise, checks against all functions' features.
    """
    if not text:
        return []

    skills = _load_skills()
    tokens = set(_tokenize(text))
    found: set[str] = set()

    if function and function in skills:
        # Targeted: only check this function's features
        func_tokens = {f["feature"] for f in skills[function]}
        found = tokens & func_tokens
    else:
        # Broad: check all functions
        all_tokens = _get_all_skills()
        found = tokens & all_tokens

    return sorted(found)


def extract_skills_for_function(text: str, function: str) -> tuple[list[str], list[str]]:
    """Extract skills and return (has, missing) relative to a function's top features.

    has: skills found in the resume that are top features for this function
    missing: top features for this function NOT found in the resume
    """
    skills = _load_skills()
    if function not in skills:
        return [], []

    tokens = set(_tokenize(text))
    func_features = skills[function]

    has = []
    missing = []
    for f in func_features[:30]:  # top 30 features
        if f["feature"] in tokens:
            has.append(f["feature"])
        else:
            missing.append(f["feature"])

    return sorted(has), missing


def extract_skills_from_bundle(bundle: StudentBundle) -> list[str]:
    resume_text = getattr(bundle, "resume_text", "") or ""
    linkedin_text = getattr(bundle, "linkedin_text", "") or ""
    return extract_skills_from_text(f"{resume_text}\n{linkedin_text}")
