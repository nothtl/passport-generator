"""LLM-based job relevance ranking — pairwise comparison for better job matching.

Replaces flat IDF scoring with DeepSeek judging which jobs best fit the student's
actual profile, aspirations, and career direction. Modeled after the zip pipeline's
langsort approach but using DeepSeek for 250x cost savings.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"


def rank_jobs_by_relevance(
    jobs: list[dict],
    student_function: str,
    student_subdomain: str,
    student_skills: list[str],
    ideal_careers: list[str],
    student_summary: str = "",
    top_k: int = 8,
) -> list[dict]:
    """Use DeepSeek to rank jobs by pairwise relevance comparison.

    Instead of scoring each job independently, asks the LLM to compare jobs
    head-to-head and explain WHY one is better than another for this specific
    student. This surfaces hidden fits that independent scoring misses.
    """
    if not DEEPSEEK_KEY or len(jobs) < 2:
        return jobs

    profile = f"Career direction: {student_function}"
    if student_subdomain:
        profile += f" / {student_subdomain}"
    if ideal_careers:
        profile += f"\nAspirational careers: {', '.join(ideal_careers)}"
    if student_skills:
        profile += f"\nKey skills: {', '.join(student_skills[:15])}"
    if student_summary:
        profile += f"\nBackground: {student_summary[:200]}"

    # Build job entries
    job_entries = []
    for i, job in enumerate(jobs[:12]):
        title = job.get("title", "Unknown")
        company = job.get("company", "")
        if company and len(str(company)) > 2 and not _is_uuid(str(company)):
            title += f" @ {company}"
        jd = job.get("jd_markdown", "")[:200] if job.get("jd_markdown") else ""
        raw_skills = job.get("skills", [])
        try:
            job_skills = list(raw_skills)[:6] if raw_skills is not None and len(raw_skills) > 0 else []
        except Exception:
            job_skills = []
        job_entries.append({
            "index": i + 1,
            "title": title,
            "skills": job_skills,
            "description": jd[:150],
        })

    prompt = (
        "You are ranking job opportunities for a STUDENT or early-career candidate (16-22). "
        "Compare these jobs PAIRWISE and rank them by how well they fit THIS specific student. "
        "Consider: (1) Does their experience and skills match? (2) Is it attainable now? "
        "(3) Does it move them toward their career goals?\n\n"
        "STUDENT PROFILE:\n"
        f"{profile}\n\n"
        "JOBS:\n"
        + "\n".join(
            f"JOB {j['index']}: {j['title']}"
            + (f"\n  Skills: {', '.join(j['skills'])}" if j['skills'] else "")
            + (f"\n  Desc: {j['description']}" if j['description'] else "")
            for j in job_entries
        )
        + "\n\n"
        "Return EXACTLY this JSON:\n"
        '{"rankings": [\n'
        '  {"rank": <1 is best, 2, 3...>, "job_index": <int>, "relevance": <0-100>, '
        '"goal_fit": "<high|medium|low>", '
        '"verdict": "<solid|stretch|weak>", '
        '"reason": "<1 honest sentence explaining why this specific job fits (or does not fit) THIS student. Be specific — name their actual skills. Be honest about gaps.>"}\n'
        "]}\n\n"
        "Rules:\n"
        "- solid: the student can realistically get this job now with their current skills\n"
        "- stretch: the student could grow into this with some skill-building\n"
        "- weak: major gaps or wrong career direction for this student\n"
        "- Be honest — if a job is a poor fit, say so clearly\n"
        "- Rank EVERY job from 1 (best) to N (worst)"
    )

    payload = {
        "model": MODEL, "temperature": 0.2, "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        req = urllib.request.Request(
            DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
        rankings = {r["job_index"]: r for r in result.get("rankings", [])}

        for i, job in enumerate(jobs[:12]):
            idx = i + 1
            r = rankings.get(idx, {"relevance": 50, "goal_fit": "medium", "verdict": "stretch", "reason": ""})
            job["relevance_score"] = r.get("relevance", 50)
            job["relevance_reason"] = r.get("reason", "")
            job["goal_fit"] = r.get("goal_fit", "medium")
            job["verdict"] = r.get("verdict", "stretch")
            job["rank"] = r.get("rank", idx)

        ranked = sorted(jobs[:12], key=lambda j: j.get("rank", 99))
        for job in ranked:
            orig_fit = job.get("fit", 0)
            rel = job.get("relevance_score", 50)
            job["fit"] = round(rel * 0.7 + orig_fit * 0.3)
            job.setdefault("job_score_breakdown", {})
            bd = job["job_score_breakdown"]
            bd["relevance_score"] = rel
            bd["relevance_reason"] = job.get("relevance_reason", "")
            bd["verdict"] = job.get("verdict", "stretch")
        return ranked[:top_k]
    except Exception:
        return jobs[:top_k]


def _is_uuid(company: str) -> bool:
    import re
    return bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', company))
