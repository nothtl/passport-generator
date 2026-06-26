from __future__ import annotations

import json
import os

_LOOKUP_PATH = os.path.join(os.path.dirname(__file__), "lookup_table.json")


def _load_roles() -> list[dict]:
    with open(_LOOKUP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.values())


def match_role(student_skills: list[str]) -> dict | None:
    """Return best-fit role with match_pct, matched_skills, missing_skills.

    Also includes `alternatives` — the top 3 runner-up roles for exploration.
    Returns None only if the lookup table is empty/corrupt.
    """
    student_set = set(s.lower() for s in student_skills)
    roles = _load_roles()
    if not roles:
        return None

    scored = []
    for role in roles:
        required = role.get("required_skills", [])
        if not required:
            continue

        matched_skills: list[str] = []
        missing_skills: list[str] = []
        matched_weight = 0.0
        total_weight = 0.0

        for entry in required:
            name = entry["name"].lower()
            weight = float(entry.get("importance", 1.0))
            total_weight += weight
            if name in student_set:
                matched_skills.append(entry["name"])
                matched_weight += weight
            else:
                missing_skills.append(entry["name"])

        if total_weight == 0:
            continue

        match_pct = round(matched_weight / total_weight * 100, 1)
        scored.append({
            "key": role.get("function", ""),
            "function": role.get("function", ""),
            "sub_function": role.get("sub_function", ""),
            "level": role.get("level", "Entry"),
            "match_pct": match_pct,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        })

    if not scored:
        return None

    scored.sort(key=lambda r: r["match_pct"], reverse=True)

    # Attach runner-up alternatives for exploration
    best = scored[0]
    best["alternatives"] = [
        {"function": r["function"], "match_pct": r["match_pct"]}
        for r in scored[1:4]
    ]
    return best
