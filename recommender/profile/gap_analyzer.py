from __future__ import annotations

from typing import Any


def analyze_gaps(
    role_title: str,
    function: str,
    level: str,
    match_pct: float,
    matched_skills: list[str],
    missing_skills: list[str],
    all_skills: list[Any],
    ideal_passport: dict[str, float],
    student_passport: dict[str, float] | None = None,
) -> dict[str, Any]:
    top_gaps = [s for s in all_skills if not s.student_has]
    top_gaps.sort(key=lambda s: s.frequency, reverse=True)

    passport_gaps = {}
    if student_passport and ideal_passport:
        for pillar in ["EC", "GC", "RFF", "CR", "CT", "CI"]:
            student = student_passport.get(pillar, 0)
            ideal = ideal_passport.get(pillar, 0)
            passport_gaps[pillar] = {
                "student": student,
                "ideal": ideal,
                "delta": round(ideal - student, 1),
            }

    return {
        "role": role_title,
        "function": function,
        "level": level,
        "match_pct": match_pct,
        "skills_summary": {
            "has": len(matched_skills),
            "missing": len(missing_skills),
            "total": len(matched_skills) + len(missing_skills),
        },
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "ranked_skills": [
            {"skill": s.skill, "frequency": s.frequency, "has": s.student_has}
            for s in sorted(all_skills, key=lambda x: x.frequency, reverse=True)
        ],
        "top_gaps": [
            {"skill": s.skill, "frequency": s.frequency}
            for s in top_gaps[:5]
        ],
        "ideal_passport": ideal_passport,
        "student_passport": student_passport,
        "passport_gaps": passport_gaps,
    }
