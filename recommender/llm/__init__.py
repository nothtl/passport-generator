"""
LLM enhancer — adds explanations, evidence, and ready/target job labeling.

Uses DeepSeek API. Falls back gracefully if no key — core pipeline works without it.
Cost: ~$0.001 per resume (DeepSeek pricing).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

_API_KEY = ""
# Load key from api file in project root
try:
    _api_path = os.path.join(os.path.dirname(__file__), "..", "..", "api")
    if os.path.exists(_api_path):
        _API_KEY = open(_api_path).read().strip()
except Exception:
    pass

if not _API_KEY:
    _API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"  # resolves to deepseek-v4-flash on their API
_ENABLED = bool(_API_KEY)


def is_enabled() -> bool:
    return _ENABLED


def _call_llm(prompt: str, max_tokens: int = 500, max_retries: int = 2) -> str | None:
    if not _ENABLED:
        return None

    import urllib.request

    body = json.dumps({
        "model": _DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a career coach. Return only valid JSON, no markdown, no explanation outside the JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                _DEEPSEEK_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            print(f"[LLM] Failed after {max_retries+1} attempts: {e}")
            return None
    return None


def classify_resume(resume_text: str) -> dict | None:
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
    if not skills:
        return []

    skills_list = ", ".join(skills[:20])
    prompt = f"""For each skill, find the exact sentence in the resume that proves it.
Rate proficiency: proficient (strong), developing (some), emerging (weak).

Return JSON: {{"evidence": [{{"skill": "...", "evidence": "exact sentence...", "proficiency": "proficient|developing|emerging"}}]}}

Skills: [{skills_list}]

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
    if not missing_skills:
        return None

    prompt = f"""This person is pursuing a {function} career.
They HAVE: {", ".join(has_skills[:10])}
They MISS (from real job postings): {", ".join(missing_skills[:10])}

Write 2-3 sentences on which gaps matter most and why. Be specific, actionable.
Return JSON: {{"explanation": "..."}}"""

    response = _call_llm(prompt, max_tokens=200)
    if not response:
        return None
    try:
        return json.loads(response).get("explanation", "")
    except json.JSONDecodeError:
        return None


def evaluate_jobs(resume_text: str, jobs: list[dict], skills: list[str]) -> list[dict]:
    if not jobs:
        return []

    job_descriptions = "\n\n".join(
        f"JOB {i+1}: {j.get('title', '')} at {j.get('company', '')}"
        for i, j in enumerate(jobs[:5])
    )

    prompt = f"""For each job, rate fit (0-100). Label: "ready" (apply now), "target" (growth path), or "reach" (major gaps).
Include WHY this job fits and what specific gaps remain.

Return JSON: {{"evaluations": [
  {{"job_index": 1, "fit": 75, "label": "ready", "why_fits": "...", "remaining_gaps": "..."}},
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


def infer_skills(resume_text: str, skills: list[str], missing: list[str]) -> list[str]:
    """LLM infers skills the person likely has based on related skills they DO show.

    E.g., someone using FastAPI + YOLO + PyTorch almost certainly knows Python,
    even if they didn't list it. Returns list of inferred skill names.
    """
    if not missing or not _ENABLED:
        return []

    prompt = f"""This person's resume shows these skills: {', '.join(skills[:20])}
These skills are listed as MISSING by our keyword scanner: {', '.join(missing[:15])}

Which of the MISSING skills does this person ALMOST CERTAINLY have, based on inference?
(e.g., someone who uses FastAPI, PyTorch, and ROS2 definitely knows Python.
Someone using Azure Functions definitely knows cloud deployment.)

Return JSON: {{"inferred": ["python", "cloud deployment", ...]}}
Only include skills you are highly confident about. Skip uncertain ones."""

    response = _call_llm(prompt, max_tokens=200)
    if not response:
        return []
    try:
        return json.loads(response).get("inferred", [])
    except json.JSONDecodeError:
        return []


def enhance(analysis: dict, resume_text: str) -> dict:
    """Add LLM enrichments. No-op if no API key."""
    if not _ENABLED:
        return analysis

    result = dict(analysis)
    result["llm_enhanced"] = False
    t0 = time.time()

    # 1. Fix classification if low confidence
    if result.get("match_pct", 100) < 30:
        llm_class = classify_resume(resume_text)
        if llm_class and llm_class.get("function"):
            result["function"] = llm_class["function"]
            result["llm_reclassified"] = True
            result["llm_reasoning"] = llm_class.get("reasoning", "")

    # 2. Infer missing skills that person likely has
    inferred = infer_skills(
        resume_text,
        result.get("skills_extracted", []),
        result.get("missing_skills", []),
    )
    if inferred:
        result["inferred_skills"] = inferred
        # Remove inferred from gaps
        result["missing_skills"] = [s for s in result.get("missing_skills", []) if s not in inferred]

    # 3. Skill evidence
    evidence = extract_skill_evidence(resume_text, result.get("skills_extracted", [])[:15])
    if evidence:
        result["skill_evidence"] = evidence

    # 4. Gap explanation
    explanation = explain_gaps(
        result.get("missing_skills", []),
        result.get("skills_extracted", []),
        result["function"],
    )
    if explanation:
        result["gap_explanation"] = explanation

    # 4. Job evaluations — include inferred skills for better analysis
    all_skills = list(set(result.get("skills_extracted", []) + result.get("inferred_skills", [])))
    evaluations = evaluate_jobs(
        resume_text,
        result.get("openings", []),
        all_skills,
    )
    if evaluations:
        result["job_evaluations"] = evaluations
        result["ready_jobs"] = []
        result["target_jobs"] = []
        for ev in evaluations:
            idx = ev.get("job_index", 1) - 1
            if idx < len(result.get("openings", [])):
                job = dict(result["openings"][idx])
                job["fit"] = ev.get("fit", 0)
                job["label"] = ev.get("label", "target")
                job["why_fits"] = ev.get("why_fits", ev.get("why", ""))
                job["remaining_gaps"] = ev.get("remaining_gaps", "")
                if ev.get("label") == "ready":
                    result["ready_jobs"].append(job)
                else:
                    result["target_jobs"].append(job)

    result["llm_enhanced"] = True
    result["llm_timing_ms"] = round((time.time() - t0) * 1000)
    return result
