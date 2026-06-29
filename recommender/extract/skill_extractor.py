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

if TYPE_CHECKING:
    from passport_agent_v2.tools.ingest import StudentBundle

_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(_HERE, "..", "data", "skill_vocabulary.json")

_vocab_normalized: set[str] | None = None


def _load_vocab_norm() -> set[str]:
    global _vocab_normalized
    if _vocab_normalized is None:
        with open(_VOCAB_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        _vocab_normalized = {re.sub(r"[- ]", "", s) for s in raw}
    return _vocab_normalized


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Generate 1-3 grams. Returns (display, normalized) pairs."""
    cleaned = re.sub(r"[^a-z\s-]", " ", text.lower())
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


def extract_skills_from_text(text: str, function: str | None = None) -> list[str]:
    """Extract skills: tokenize -> normalize -> check JD vocabulary -> return display forms.

    ~2ms. Normalization bridges "computer vision" <-> "computer-vision".
    Vocabulary from 67K JD skill names. No hand-coded rules.
    """
    if not text:
        return []

    vocab_norm = _load_vocab_norm()
    candidates = _tokenize(text)

    seen = set()
    skills = []
    for display, normed in candidates:
        if normed in seen or len(normed) < 3:
            continue
        if normed in vocab_norm:
            skills.append(display)
            seen.add(normed)

    return sorted(skills)


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
