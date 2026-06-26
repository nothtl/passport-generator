from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillGap:
    skill: str
    frequency: int
    importance: float
    student_has: bool


@dataclass
class RoleProfile:
    role_title: str
    function: str
    level: str
    match_pct: float
    matched_skills: list[str]
    missing_skills: list[str]
    all_skills: list[SkillGap] = field(default_factory=list)
    ideal_passport: dict[str, float] = field(default_factory=dict)
    open_positions: list[dict] = field(default_factory=list)
    jd_count: int = 0
