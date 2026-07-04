from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = _ROOT / "corpus"
_OUT = _ROOT / "data" / "skill_vocabulary.json"

_STOPWORDS = {
    "activities", "activity", "children", "events", "local", "may", "work", "use",
    "using", "student", "students", "support", "resume", "research", "planning",
    "preparation", "intern", "volunteer", "learning",
}


def _normalize(text: str) -> str:
    return re.sub(r"[- ,/]", "", str(text).lower())


def rebuild_vocabulary() -> dict[str, int]:
    docs = 0
    counter: Counter[str] = Counter()
    raw_labels: dict[str, str] = {}

    for path in sorted(_CORPUS.glob("*.parquet")):
        table = pq.read_table(path)
        df = table.to_pandas()
        if "skills" not in df.columns:
            continue
        for skills in df["skills"]:
            docs += 1
            seen = set()
            for skill in (list(skills) if hasattr(skills, "__iter__") else []):
                if not isinstance(skill, str):
                    continue
                normed = _normalize(skill)
                if not normed or normed in seen:
                    continue
                seen.add(normed)
                counter[normed] += 1
                raw_labels.setdefault(normed, skill.strip())

    min_df = max(2, int(max(1, docs) * 0.001))
    kept = []
    for normed, count in counter.items():
        label = raw_labels[normed]
        if len(normed) < 3:
            continue
        if normed.isdigit():
            continue
        if not re.search(r"[a-z]", normed):
            continue
        if normed in _STOPWORDS:
            continue
        if count < min_df:
            continue
        kept.append(label)

    kept = sorted(set(kept), key=str.lower)
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump(kept, handle, indent=2)
    return {"documents": docs, "skills": len(kept)}


if __name__ == "__main__":
    result = rebuild_vocabulary()
    print(json.dumps(result, indent=2))
