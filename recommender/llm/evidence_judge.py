from __future__ import annotations

import json
from typing import Any

from recommender.llm.cache import _jsonable
from recommender.llm._validation import _as_list, _as_text_list


class EvidenceNormalizerJudge:
    def __init__(self, provider, model: str):
        self.provider = provider
        self.model = model

    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        skills_list = _as_text_list(payload.get("skills"))[:80]
        section_skills = payload.get("section_skills", {})

        # Build simple text summary instead of nested JSON (avoids DeepSeek parse errors)
        sections_text = ""
        for sec_name in ["skills", "experience", "education", "leadership", "projects"]:
            sec_skills = _as_text_list(section_skills.get(sec_name, []))[:15]
            if sec_skills:
                sections_text += f"\n{sec_name.upper()}: {', '.join(sec_skills)}"

        prompt = (
            "Review these extracted skills from a student resume. Identify which are REAL skills "
            "and which are noise (generic words, artifacts, non-skills).\n\n"
            f"SKILLS TO REVIEW:\n{', '.join(skills_list)}\n"
            f"{sections_text}\n\n"
            "Return EXACTLY this JSON:\n"
            '{"scored_skills": ["<real skill 1>", "<real skill 2>"], '
            '"rejected_skills": ["<noise word>", "<artifact>"], '
            '"implicit_skills": [{"skill": "<name>", "reason": "<why>"}]}\n\n'
            "Rules:\n"
            "- scored_skills: ONLY real, meaningful skills (keep these)\n"
            "- rejected_skills: generic words (activities, children, events, local, part, computer), "
            "dash artifacts (-community, -present), or non-skills\n"
            "- implicit_skills: skills the person likely has but didn't state explicitly (max 3)\n"
            "- Keep ALL multi-word phrases (community outreach, graphic design, remote learning)\n"
            "- Keep technical terms (Python, AWS, Photoshop, Google Suite, Epic, VMware)\n"
            "- Keep soft skills only if clearly demonstrated (mentoring, public speaking)"
        )

        response = self.provider.complete_json("evidence", "", prompt, self.model)
        scored = _as_text_list(response.get("scored_skills"))
        rejected = _as_text_list(response.get("rejected_skills"))
        implicit = [
            {"skill": str(item.get("skill", "")).strip(),
             "reason": str(item.get("reason", "")).strip(),
             "confidence": str(item.get("confidence", "")).strip() or "medium"}
            for item in _as_list(response.get("implicit_skills"))
            if str(item.get("skill", "")).strip()
        ]

        return {
            "accepted": bool(scored or rejected or implicit),
            "scored_skills": scored,
            "verified_skills": scored,
            "possible_skills": [],
            "rejected_skills": sorted(set(rejected)),
            "implicit_skills": implicit,
            "evidence_items": [],
            "review_reasons": [],
        }
