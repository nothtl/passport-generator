from __future__ import annotations

import os
import re
from typing import Any

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")

# In-memory cache: function_name -> filtered DataFrame
# Avoids re-reading the same parquet on every MCP call.
_cached_df: dict[str, Any] = {}


def _load_df(func_lower: str) -> Any:
    """Read parquet once, filter to Entry/Intern/Junior, cache in memory."""
    if func_lower in _cached_df:
        return _cached_df[func_lower]

    path = os.path.join(_CORPUS_DIR, f"{func_lower}.parquet")
    if not os.path.exists(path):
        _cached_df[func_lower] = None
        return None

    import pyarrow.parquet as pq

    # Predicate pushdown: only load rows matching target levels
    table = pq.read_table(
        path,
        filters=[("level", "in", ["intern", "entry", "junior", "Intern", "Entry", "Junior"])],
    )
    df = table.to_pandas()
    if df.empty:
        _cached_df[func_lower] = None
        return None

    _cached_df[func_lower] = df
    return df


def retrieve_jds(
    function: str,
    level: str = "Entry",
    student_skills: list[str] | None = None,
    top_k: int = 10,
    broad_sample: int = 0,
) -> list[dict[str, Any]]:
    """Retrieve JDs for a function. If student_skills provided, ranks by overlap.

    Set broad_sample > 0 to also include randomly sampled JDs (beyond top_k)
    for unbiased skill frequency computation.
    """
    func_lower = function.lower()
    df = _load_df(func_lower)
    if df is None:
        return []

    import pandas as pd

    if not student_skills or "skills" not in df.columns:
        result = df.head(top_k)
        if broad_sample > 0 and len(df) > top_k:
            extra = df.iloc[top_k:].sample(n=min(broad_sample, len(df) - top_k), random_state=42)
            result = pd.concat([result, extra])
        return result.to_dict("records")
    # Normalize: strip hyphens and spaces for fuzzy matching
    def _norm(s):
        return re.sub(r'[- ]', '', s.lower())

    student_set = set(_norm(s) for s in student_skills)
    # Also build expanded set from aggregator's synonym map
    # Expand using ESCO synonym map (85K alt-labels, data-driven)
    try:
        import json, os as _os
        _syn_path = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'esco_synonyms.json')
        with open(_syn_path) as _f:
            _esco_syns = json.load(_f)
        for canonical, aliases in _esco_syns.items():
            if _norm(canonical) in student_set:
                continue  # already have it
            if any(_norm(a) in student_set for a in aliases):
                student_set.add(_norm(canonical))
    except Exception:
        pass
    scores = df["skills"].apply(
        lambda row: sum(1 for s in row if isinstance(s, str) and _norm(s) in student_set)
        if row is not None else 0
    )
    ranked = df.iloc[(-scores).argsort()]

    # Top-k matched + optional broad sample for unbiased gap detection
    top = ranked.head(top_k)
    if broad_sample > 0 and len(ranked) > top_k:
        tail = ranked.iloc[top_k:]
        extra = tail.sample(n=min(broad_sample, len(tail)), random_state=42)
        result = pd.concat([top, extra])
        return result.to_dict("records")
    return top.to_dict("records")
