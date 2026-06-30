"""
JD retriever — parquet-backed with IDF-weighted skill scoring.

Each extracted skill is weighted by its IDF (inverse document frequency)
computed from the function's JD corpus. Skills that appear in many JDs
get low weight; rare, distinctive skills get high weight. This auto-filters
generic terms like "building", "career", "claims" without any hand-coded rules.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")

# Functions without dedicated parquets fall back to the closest related one
_FALLBACK_MAP = {
    "arts-media": "design",
    "agriculture": "other",
    "building-grounds": "other",
    "personal-care": "other",
    "protective-service": "security",
    "science": "technology",
    "social-service": "education",
    "administrative": "ops",
    "food-service": "other",
    "hospitality": "other",
    "logistics": "ops",
    "manufacturing": "ops",
}

_cached_df: dict[str, Any] = {}
_cached_idf: dict[str, dict[str, float]] = {}
_cached_pmi: dict[str, dict[tuple[str, str], float]] = {}


def _resolve_func(func_lower: str) -> str:
    """Resolve function to parquet file, using fallback if missing."""
    path = os.path.join(_CORPUS_DIR, f"{func_lower}.parquet")
    if os.path.exists(path):
        return func_lower
    fallback = _FALLBACK_MAP.get(func_lower)
    if fallback:
        return fallback
    return func_lower


def _load_df(func_lower: str) -> Any:
    cache_key = func_lower  # always cache under original name
    if cache_key in _cached_df:
        return _cached_df[cache_key]

    resolved = _resolve_func(func_lower)
    path = os.path.join(_CORPUS_DIR, f"{resolved}.parquet")
    if not os.path.exists(path):
        _cached_df[cache_key] = None
        return None
    import pyarrow.parquet as pq
    table = pq.read_table(
        path,
        filters=[("level", "in", ["intern", "entry", "junior", "Intern", "Entry", "Junior"])],
    )
    df = table.to_pandas()
    if df.empty:
        _cached_df[cache_key] = None
        return None
    _cached_df[cache_key] = df
    return df


def _compute_idf(func_lower: str) -> dict[str, float]:
    """Compute IDF for each unique skill in this function's JD corpus.

    IDF(skill) = log(N / df(skill))
    N = total JDs, df(skill) = number of JDs containing this skill.

    Computed once, cached forever.
    """
    if func_lower in _cached_idf:
        return _cached_idf[func_lower]

    df = _load_df(func_lower)
    if df is None or "skills" not in df.columns:
        _cached_idf[func_lower] = {}
        return {}

    N = len(df)

    def _norm(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    # Count document frequency per skill
    skill_df: dict[str, int] = {}
    for skills in df["skills"]:
        if skills is None:
            continue
        seen_in_jd = set()
        for s in (list(skills) if hasattr(skills, "__iter__") else []):
            if isinstance(s, str):
                normed = _norm(s)
                if normed and normed not in seen_in_jd:
                    skill_df[normed] = skill_df.get(normed, 0) + 1
                    seen_in_jd.add(normed)

    idf = {
        skill: math.log(N / max(1, df_count))
        for skill, df_count in skill_df.items()
    }
    _cached_idf[func_lower] = idf
    return idf




def _compute_pmi(func_lower: str) -> dict[tuple[str, str], float]:
    """Compute PMI for skill pairs in this function's JD corpus."""
    df = _load_df(func_lower)
    if df is None or "skills" not in df.columns:
        return {}

    N = len(df)

    def _norm(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    single_freq: dict[str, int] = {}
    pair_freq: dict[tuple[str, str], int] = {}

    for skills in df["skills"]:
        if skills is None:
            continue
        normed = []
        seen = set()
        for s in (list(skills) if hasattr(skills, "__iter__") else []):
            if isinstance(s, str):
                n = _norm(s)
                if n and n not in seen:
                    normed.append(n)
                    seen.add(n)
        for n in normed:
            single_freq[n] = single_freq.get(n, 0) + 1
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                a, b = normed[i], normed[j]
                if a > b:
                    a, b = b, a
                pair_freq[(a, b)] = pair_freq.get((a, b), 0) + 1

    pmi = {}
    for (a, b), pair_count in pair_freq.items():
        if pair_count < 3:
            continue
        p_a = single_freq.get(a, 0) / N
        p_b = single_freq.get(b, 0) / N
        if p_a <= 0 or p_b <= 0:
            continue
        p_ab = pair_count / N
        score = math.log(p_ab / (p_a * p_b))
        if score > 0.15:
            pmi[(a, b)] = round(score, 3)

    return pmi


def get_related_skills(function: str, student_skills: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    """Find skills the student lacks that co-occur with ones they have."""
    def _norm(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    pmi = _compute_pmi(function.lower())
    student_normed = {_norm(s) for s in student_skills}

    related = {}
    for (a, b), score in pmi.items():
        has_a = a in student_normed
        has_b = b in student_normed
        if has_a and not has_b:
            related[b] = max(related.get(b, 0), score)
        elif has_b and not has_a:
            related[a] = max(related.get(a, 0), score)

    ranked = sorted(related.items(), key=lambda x: -x[1])
    return ranked[:top_k]


def get_jd_skill_vocabulary(function: str) -> set[str]:
    """Return the set of all unique skill names in this function's JDs.
    Used for filtering student skills to only market-relevant ones."""
    func_lower = function.lower()
    df = _load_df(func_lower)
    if df is None or 'skills' not in df.columns:
        return set()
    
    import re
    def _norm(s):
        return re.sub(r'[- ,/]', '', str(s).lower())
    
    vocab = set()
    for skills in df['skills']:
        if skills is None: continue
        for s in (list(skills) if hasattr(skills, '__iter__') else []):
            if isinstance(s, str) and len(s) > 2:
                vocab.add(_norm(s))
    return vocab


def retrieve_jds(
    function: str,
    level: str = "Entry",
    student_skills: list[str] | None = None,
    top_k: int = 10,
    broad_sample: int = 0,
) -> list[dict[str, Any]]:
    func_lower = function.lower()
    df = _load_df(func_lower)
    if df is None:
        return []

    import pandas as pd

    if not student_skills or "skills" not in df.columns:
        result = df.head(top_k)
        if broad_sample > 0 and len(df) > top_k:
            extra = df.iloc[top_k:].sample(
                n=min(broad_sample, len(df) - top_k), random_state=42
            )
            result = pd.concat([result, extra])
        return result.to_dict("records")

    def _norm(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    idf = _compute_idf(func_lower)
    student_set = set(_norm(s) for s in student_skills)

    # Build TF weights from the skill list: if a skill like "security" appears
    # multiple times in the extracted skills (from different n-grams), weight it higher.
    # Single-occurrence skills get TF=1.0 baseline.
    from collections import Counter as _Counter
    _tf_counts = _Counter(_norm(s) for s in student_skills)
    student_tf = {n: 1 + math.log(c) for n, c in _tf_counts.items()}

    # Expand via ESCO synonyms (data-driven, 85K alt-labels)
    try:
        _here = os.path.dirname(__file__)
        _syn_path = os.path.join(_here, "..", "data", "esco_synonyms.json")
        with open(_syn_path) as _f:
            _esco = json.load(_f)
        for canonical, aliases in _esco.items():
            if _norm(canonical) in student_set:
                continue
            if any(_norm(a) in student_set for a in aliases):
                student_set.add(_norm(canonical))
                student_tf[_norm(canonical)] = 1.5  # inferred synonyms get slight boost
    except Exception:
        pass

    # TF-IDF weighted scoring: TF(skill) × IDF(corpus)
    def _score_jd(row_skills):
        if row_skills is None:
            return 0
        total = 0.0
        for s in (list(row_skills) if hasattr(row_skills, "__iter__") else []):
            if isinstance(s, str):
                normed = _norm(s)
                if normed in student_set:
                    tf = student_tf.get(normed, 1.0)
                    total += tf * idf.get(normed, 1.0)
        return total

    scores = df["skills"].apply(_score_jd)
    ranked = df.iloc[(-scores).argsort()]

    top = ranked.head(top_k)
    if broad_sample > 0 and len(ranked) > top_k:
        tail = ranked.iloc[top_k:]
        extra = tail.sample(n=min(broad_sample, len(tail)), random_state=42)
        result = pd.concat([top, extra])
        return result.to_dict("records")
    return top.to_dict("records")
