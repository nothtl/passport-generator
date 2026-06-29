"""
LLM enhancer — adds explanations, evidence, and zone-based job labeling.

Optional: runs only when OPENROUTER_API_KEY is set in .env or environment.
Falls back gracefully — the core pipeline works without it.

Cost: ~2,000 tokens per resume (~$0.001 or free on OpenRouter free tier).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if not _API_KEY:
    # Try loading from .env file
    try:
        _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(_env_path):
            for line in open(_env_path):
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    _API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

_API_MODEL = "google/gemini-2.5-flash-lite:free"  # free on OpenRouter
_ENABLED = bool(_API_KEY)


def is_enabled() -> bool:
    return _ENABLED


def _call_llm(prompt: str, max_tokens: int = 500) -> str | None:
    """Call OpenRouter API. Returns response text or None on failure."""
    if not _ENABLED:
        return None

    import urllib.request

    body = json.dumps({
        "model": _API_MODEL,
        "messages": [
            {"role": "system", "content": "You are a career coach. Return only valid JSON, no markdown, no explanation outside the JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print(f"[LLM] API call failed: {e}")
        return None


def classify_resume(resume_text: str) -> dict | None:
    """LLM-based function classification — used when ensemble confidence is low."""
    prompt = f"""Classify this resume into one of these career functions.
Return JSON: {{"function": "...", "confidence": 0-100, "reasoning": "..."}}

Functions: technology, healthcare, education, finance, sales, food-service,
skilled-trade, design, marketing, ops, legal, arts-media, administrative,
logistics, hospitality, manufacturing, agriculture, science, social-service,
personal-care, protective-service, building-grounds, support

Resume:
{resume_text[:3000]}"""

    response = _call_llm(prompt, max_tokens=200)
    if not response:
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def extract_skill_evidence(resume_text: str, skills: list[str]) -> list[dict]:
    """For each skill, find the evidence span in the resume with proficiency."""
    if not skills:
        return []

    skills_list = ", ".join(skills[:20])
    prompt = f"""For each skill, find the exact sentence in the resume that proves it.
Rate proficiency as: proficient (strong evidence), developing (some evidence), or emerging (weak/minimal).

Return JSON: {{"evidence": [{{"skill": "...", "evidence": "exact sentence...", "proficiency": "proficient|developing|emerging"}}]}}

Skills to find: [{skills_list}]

Resume:
{resume_text[:3000]}"""

    response = _call_llm(prompt, max_tokens=600)
    if not response:
        return []
    try:
        result = json.loads(response)
        return result.get("evidence", [])
    except json.JSONDecodeError:
        return []


def explain_gaps(missing_skills: list[str], has_skills: list[str], function: str) -> str | None:
    """Explain why these gaps matter and how to close them."""
    if not missing_skills:
        return None

    prompt = f"""This person is pursuing a {function} career.
They HAVE these skills: {", ".join(has_skills[:10])}
They are MISSING these skills (from real job postings): {", ".join(missing_skills[:10])}

Write a 2-3 sentence explanation of which missing skills matter most and why.
Focus on the top 3-5 gaps. Be specific and actionable.
Return JSON: {{"explanation": "..."}}"""

    response = _call_llm(prompt, max_tokens=200)
    if not response:
        return None
    try:
        return json.loads(response).get("explanation", "")
    except json.JSONDecodeError:
        return None


def evaluate_jobs(resume_text: str, jobs: list[dict], skills: list[str]) -> list[dict]:
    """Rate each job for fit and label as ready/target. Adds zone context."""
    if not jobs:
        return []

    job_descriptions = "\n\n".join(
        f"JOB {i+1}: {j.get('title', '')} at {j.get('company', '')}"
        for i, j in enumerate(jobs[:5])
    )

    prompt = f"""For each job, rate how well this resume fits (0-100).
Label as "ready" (can apply now), "target" (growth path, needs some upskilling),
or "reach" (significant gaps).

Return JSON: {{"evaluations": [
  {{"job_index": 1, "fit": 75, "label": "ready", "why": "Your FastAPI and Python experience directly matches..."}},
  ...
]}}

Resume skills: {", ".join(skills[:15])}

{job_descriptions}"""

    response = _call_llm(prompt, max_tokens=500)
    if not response:
        return []
    try:
        result = json.loads(response)
        return result.get("evaluations", [])
    except json.JSONDecodeError:
        return []


def enhance(analysis: dict, resume_text: str) -> dict:
    """Add LLM enrichments to an existing pipeline result.

    Returns the enhanced dict — same structure, with additional fields.
    No-op if LLM is disabled.
    """
    if not _ENABLED:
        return analysis

    result = dict(analysis)
    result["llm_enhanced"] = False
    t0 = time.time()

    # 1. Fix classification if confidence is low
    if result.get("match_pct", 100) < 30:
        llm_class = classify_resume(resume_text)
        if llm_class and llm_class.get("function"):
            result["function"] = llm_class["function"]
            result["llm_reclassified"] = True
            result["llm_reasoning"] = llm_class.get("reasoning", "")

    # 2. Extract skill evidence
    evidence = extract_skill_evidence(resume_text, result.get("skills_extracted", [])[:15])
    if evidence:
        result["skill_evidence"] = evidence

    # 3. Explain gaps
    explanation = explain_gaps(
        result.get("missing_skills", []),
        result.get("skills_extracted", []),
        result["function"],
    )
    if explanation:
        result["gap_explanation"] = explanation

    # 4. Evaluate and label jobs
    evaluations = evaluate_jobs(
        resume_text,
        result.get("openings", []),
        result.get("skills_extracted", []),
    )
    if evaluations:
        result["job_evaluations"] = evaluations
        # Split into ready/target
        result["ready_jobs"] = []
        result["target_jobs"] = []
        for ev in evaluations:
            idx = ev.get("job_index", 1) - 1
            if idx < len(result.get("openings", [])):
                job = dict(result["openings"][idx])
                job["fit"] = ev.get("fit", 0)
                job["label"] = ev.get("label", "target")
                job["why"] = ev.get("why", "")
                if ev.get("label") == "ready":
                    result["ready_jobs"].append(job)
                else:
                    result["target_jobs"].append(job)

    result["llm_enhanced"] = True
    result["llm_timing_ms"] = round((time.time() - t0) * 1000)
    return result
