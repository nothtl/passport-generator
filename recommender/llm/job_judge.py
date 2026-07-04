from __future__ import annotations

import json
from typing import Any

from recommender.llm.cache import _jsonable
from recommender.llm._validation import _as_list, _as_text_list, _normalize_job_id

_GOAL_BAND_ADJUSTMENTS = {"high": 0.08, "medium": 0.0, "low": -0.08}
_ATTAINABILITY_ADJUSTMENTS = {"ready": 0.0, "bridge": -0.05, "reach": -0.15, "blocked": -0.35}
_BLOCKER_PENALTY = 0.15
_MAX_BLOCKER_PENALTY = 0.40


class JobFitGapJudge:
    def __init__(self, provider, model: str):
        self.provider = provider
        self.model = model

    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_ids = [_normalize_job_id(job) for job in _as_list(payload.get("jobs"))]
        # Surface student profile prominently so the LLM writes specific explanations
        intent = payload.get("student_intent", {})
        skills = _as_text_list(payload.get("verified_skills"))[:12]
        goals = _as_text_list(intent.get("goal_domains", []))
        roles = _as_text_list(intent.get("goal_roles", []))
        funcs = _as_text_list(payload.get("chosen_functions", []))

        student_blurb = f"Career direction: {', '.join(funcs) if funcs else 'unknown'}"
        if goals:
            student_blurb += f"\nGoals: {', '.join(goals)}"
        if roles:
            student_blurb += f"\nRoles: {', '.join(roles)}"
        if skills:
            student_blurb += f"\nSkills: {', '.join(skills)}"

        job_lines = []
        for job in _as_list(payload.get("jobs"))[:8]:
            jid = _normalize_job_id(job)
            title = str(job.get("title", "?"))
            jd = str(job.get("jd_markdown", "") or job.get("description", "") or "")[:200]
            skills = _as_text_list(job.get("skills", []))[:6]
            line = f"JOB {jid}: {title}"
            if jd: line += f"\n  Desc: {jd}"
            if skills: line += f"\n  Skills: {', '.join(skills)}"
            job_lines.append(line)

        system_prompt = (
            "You are a career coach helping a student evaluate job opportunities. "
            "For each job, write 1-2 encouraging sentences explaining whether it fits THIS student. "
            "Name their specific skills and experience. Be honest about gaps. "
            "Return EXACTLY this JSON (no markdown):\n"
            '{"jobs": [{"job_id": "<id>", "goal_fit_band": "high|medium|low", '
            '"attainability_band": "ready|bridge|reach|blocked", '
            '"hard_blockers": [], "important_missing_skills": [], '
            '"why_fits": "<1-2 sentences naming the specific skills and experience from this student that match this job>"}]}\n'
            "Rules:\n"
            "- why_fits: Be specific. Say 'Abigail has 2 years of volunteer mentoring experience and strong communication skills' NOT 'the candidate has relevant experience'\n"
            "- goal_fit_band: high=matches their stated career goals, medium=related, low=unrelated\n"
            "- attainability: ready=can apply now, bridge=needs skill building, reach=major gaps, blocked=credential/license wall\n"
            "- Include EVERY job listed"
        )

        user_prompt = (
            f"STUDENT PROFILE:\n{student_blurb}\n\n"
            f"JOBS TO EVALUATE:\n" + "\n".join(job_lines)
        )
        # Send everything in the user message (DeepSeek JSON mode works best this way)
        combined = system_prompt + "\n\n" + user_prompt
        response = self.provider.complete_json("jobs", "", combined, self.model)
        valid_ids = {_normalize_job_id(job) for job in _as_list(payload.get("jobs"))}
        decisions = []
        for item in _as_list(response.get("jobs")):
            job_id = str(item.get("job_id", "")).strip()
            if job_id not in valid_ids:
                continue
            decisions.append(
                {
                    "job_id": job_id,
                    "subdomain": str(item.get("subdomain", "")).strip(),
                    "goal_fit_band": str(item.get("goal_fit_band", "")).strip() or "medium",
                    "attainability_band": str(item.get("attainability_band", "")).strip() or "bridge",
                    "hard_blockers": _as_text_list(item.get("hard_blockers")),
                    "important_missing_skills": _as_text_list(item.get("important_missing_skills")),
                    "reason_codes": _as_text_list(item.get("reason_codes")),
                }
            )
        return {
            "accepted": bool(decisions),
            "jobs": decisions,
            "review_reasons": [],
        }

    def apply(self, jobs: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
        decision_by_id = {item["job_id"]: item for item in decision.get("jobs", [])}
        ranked: list[dict[str, Any]] = []
        for raw_job in jobs:
            job = dict(raw_job)
            job_id = _normalize_job_id(job)
            item = decision_by_id.get(job_id)
            adjustments = {
                "goal_fit_band": "medium",
                "attainability_band": "bridge",
                "hard_blockers": [],
                "subdomain": "",
                "important_missing_skills": [],
                "reason_codes": [],
                "why_fits": "",
            }
            fit = float(job.get("fit", 0)) / 100.0
            if item:
                adjustments = {
                    "goal_fit_band": item["goal_fit_band"],
                    "attainability_band": item["attainability_band"],
                    "hard_blockers": item["hard_blockers"],
                    "subdomain": item["subdomain"],
                    "important_missing_skills": item["important_missing_skills"],
                    "reason_codes": item["reason_codes"],
                    "why_fits": item.get("why_fits", ""),
                }
                fit += _GOAL_BAND_ADJUSTMENTS.get(item["goal_fit_band"], 0.0)
                fit += _ATTAINABILITY_ADJUSTMENTS.get(item["attainability_band"], 0.0)
                fit -= min(_MAX_BLOCKER_PENALTY, _BLOCKER_PENALTY * len(item["hard_blockers"]))
                if item["attainability_band"] == "blocked" or item["hard_blockers"]:
                    job["_suppressed"] = True
            job["fit"] = round(max(0.0, min(1.0, fit)) * 100)
            job.setdefault("job_score_breakdown", {})
            job["job_score_breakdown"]["llm_adjustments"] = adjustments
            ranked.append(job)

        ranked.sort(key=lambda item: (-item.get("fit", 0), item.get("title", "")))
        return ranked
