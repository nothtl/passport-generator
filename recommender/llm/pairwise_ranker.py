"""Calibrated LLM reranker for top-10 job candidates.

Phase 6 of the SOTA-inspired improvement plan.

Takes the top 10 deterministic candidates, asks DeepSeek to score each on
3 dimensions (skill_fit, experience_match, career_alignment), then blends
0.7 × deterministic + 0.3 × LLM score.

Cost: ~$0.001/student (10 LLM calls × ~$0.0001/call).
Still 250× cheaper than ZIP's $0.25/student GPT-4o-mini pairwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from recommender.config import get_llm
from recommender.utils import retry

_LLM = get_llm()
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", ".cache", "rerank")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = _LLM.deepseek_url
MODEL = _LLM.deepseek_model

PROMPT = """Score this job for this student on 3 dimensions (0-10 each). Be strict —
only give 8+ if there's clear evidence in the resume. Return ONLY this JSON:
{{"skill_fit": <0-10>, "experience_match": <0-10>, "career_alignment": <0-10>,
 "explanation": "<one sentence why this job fits or doesn't>"}}

RUBRIC:
- skill_fit: Does the student have the technical skills? 0=none, 5=some overlap, 8+=strong match
- experience_match: Does their work/internship experience prepare them? 0=unrelated, 5=adjacent, 8+=direct match
- career_alignment: Does this job match their stated goals/education? 0=opposite direction, 5=neutral, 8+=direct path

STUDENT:
{student_context}

JOB:
Title: {title}
Company: {company}
Description: {jd_snippet}"""


def _load_cache() -> dict[str, dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {}
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(CACHE_DIR, fname), encoding="utf-8") as f:
                cache.update(json.load(f))
    return cache


def _cache_key(student_text: str, job_id: str) -> str:
    return hashlib.md5(f"{student_text[:200]}::{job_id}".encode()).hexdigest()[:16]


@retry(max_attempts=3, delay=1.0, backoff=2.0)
def _call_llm(student_context: str, job: dict) -> dict | None:
    """Call DeepSeek to score one job. Returns dict with scores or None on failure."""
    if not DEEPSEEK_KEY:
        return None

    title = job.get("title", "")[:100]
    company = job.get("company", "")[:50]
    jd = str(job.get("jd_markdown", ""))[:800]

    prompt = PROMPT.format(
        student_context=student_context[:1500],
        title=title,
        company=company,
        jd_snippet=jd,
    )

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return None


def rerank_top_n(
    ranked_jobs: list[dict],
    student_context: str,
    resume_text: str = "",
    top_n: int = 8,
    blend_weight: float = 0.3,
) -> list[dict]:
    """Rerank the top-N jobs using LLM factor-wise scoring.

    Args:
        ranked_jobs: Deterministically ranked job list
        student_context: Summary of student (skills, goals, education)
        resume_text: Full resume for caching
        top_n: Number of top jobs to rerank (default 8)
        blend_weight: LLM score weight in final blend (default 0.3)

    Returns:
        Reranked job list with llm_scores added to job_score_breakdown
    """
    if not DEEPSEEK_KEY or not ranked_jobs:
        return ranked_jobs

    cache = _load_cache()
    updated = False

    for i, job in enumerate(ranked_jobs[:top_n]):
        jid = job.get("id", "") or job.get("url", "") or job.get("title", "")
        ck = _cache_key(resume_text or student_context, str(jid))

        if ck in cache:
            llm_result = cache[ck]
        else:
            llm_result = _call_llm(student_context, job)
            if llm_result:
                cache[ck] = llm_result
                updated = True
            time.sleep(0.3)  # rate limit

        if not llm_result:
            continue

        # Compute LLM score from 3 dimensions (0-30 → 0-1)
        skill = max(0, min(10, llm_result.get("skill_fit", 5)))
        exp = max(0, min(10, llm_result.get("experience_match", 5)))
        career = max(0, min(10, llm_result.get("career_alignment", 5)))
        llm_score = (skill + exp + career) / 30.0

        # Blend with existing deterministic score
        old_fit = job.get("fit", 50) / 100.0
        blended = (1 - blend_weight) * old_fit + blend_weight * llm_score

        # Update job
        job["fit"] = round(blended * 100)
        bd = job.setdefault("job_score_breakdown", {})
        bd["llm_skill_fit"] = skill
        bd["llm_experience_match"] = exp
        bd["llm_career_alignment"] = career
        bd["llm_score"] = round(llm_score, 3)
        bd["llm_explanation"] = llm_result.get("explanation", "")
        bd["llm_blend_weight"] = blend_weight

    # Save cache
    if updated:
        batch_key = hashlib.md5(str(sorted(cache.keys())).encode()).hexdigest()[:12]
        with open(os.path.join(CACHE_DIR, f"{batch_key}.json"), "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    # Re-sort by blended score
    ranked_jobs.sort(key=lambda j: -j.get("fit", 0))
    return ranked_jobs


def build_student_context(result: dict) -> str:
    """Build a concise student context string for the LLM prompt."""
    func = result.get("function", "unknown")
    skills = result.get("verified_skills", [])[:12]
    ideal = result.get("ideal_careers", [])[:3]
    study = result.get("student_intent", {}).get("study_program", "")
    summary = result.get("student_intent", {}).get("intent_summary", {}).get("summary_text", "")

    parts = [
        f"Career direction: {func}",
        f"Skills: {', '.join(skills)}",
    ]
    if ideal:
        parts.append(f"Goal careers: {', '.join(ideal)}")
    if study:
        parts.append(f"Education: {study[:150]}")
    if summary:
        parts.append(f"Background: {summary[:300]}")

    return "\n".join(parts)
