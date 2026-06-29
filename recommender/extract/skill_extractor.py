"""
Data-driven skill extraction — vocabulary membership only.

1. Tokenize resume into 1-3 grams
2. Check each n-gram against 110K-term ESCO + O*NET vocabulary
3. Return matched terms

No skip lists, no length filters, no regex patterns, no proper noun detection.
The 110K vocabulary is the only filter — terms not in the vocabulary are simply
not skills. IDF weighting happens downstream in the retriever.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passport_agent_v2.tools.ingest import StudentBundle

_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(_HERE, "..", "data", "skill_vocabulary.json")

_vocabulary: set[str] | None = None


def _load_vocabulary() -> set[str]:
    global _vocabulary
    if _vocabulary is None:
        with open(_VOCAB_PATH, encoding="utf-8") as f:
            _vocabulary = set(json.load(f))
    return _vocabulary


def _tokenize(text: str) -> list[str]:
    """Generate 1-3 grams from text. Pure algorithm, no hand-coded rules."""
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if not words:
        return []

    ngrams = []
    for i in range(len(words)):
        ngrams.append(words[i])
        if i + 1 < len(words):
            ngrams.append(f"{words[i]} {words[i+1]}")
        if i + 2 < len(words):
            ngrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    return ngrams


def extract_skills_from_text(text: str, function: str | None = None) -> list[str]:
    """Extract skills: tokenize → check vocabulary → return matches.

    ~1ms. No hand-coded rules. The 110K ESCO+O*NET vocabulary is the sole filter.
    """
    if not text:
        return []

    vocabulary = _load_vocabulary()
    candidates = _tokenize(text)

    seen = set()
    skills = []
    for phrase in candidates:
        if phrase in seen or len(phrase) < 3:
            continue
        if phrase in vocabulary:
            skills.append(phrase)
            seen.add(phrase)

    return sorted(skills)


def extract_skills_for_function(text: str, function: str) -> tuple[list[str], list[str]]:
    """Extract skills and compute has/missing vs function's expected skills."""
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
