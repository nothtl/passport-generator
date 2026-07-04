"""Generate cross-skill bridge summaries for ready-now jobs that differ from aspirational goals."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def generate_bridge(
    job_title: str,
    job_function: str,
    job_skills: list[str],
    student_skills: list[str],
    ideal_careers: list[str],
    aspirational_function: str,
) -> str | None:
    """Generate a 1-2 sentence bridge explaining how this job builds transferable skills
    toward the student's aspirational career goals.

    Returns None if LLM is unavailable or the job IS in the aspirational path.
    """
    if not DEEPSEEK_KEY:
        return None

    # No bridge needed if the job is already in the aspirational function
    if job_function == aspirational_function:
        return None

    prompt = (
        "A student's current best-fit job matches their existing skills but is NOT their dream career. "
        "Write 1-2 encouraging sentences explaining how this job builds TRANSFERABLE SKILLS "
        "that will help them reach their aspirational career. Be specific — name the skills. "
        "Tone: encouraging, practical, forward-looking.\n\n"
        f"Current job: {job_title} (field: {job_function})\n"
        f"Job's key skills: {', '.join(job_skills[:8])}\n"
        f"Student's existing skills: {', '.join(student_skills[:8])}\n"
        f"Aspirational career(s): {', '.join(ideal_careers)}\n\n"
        'Return EXACTLY this JSON (no markdown):\n'
        '{"bridge": "<1-2 sentence bridge summary>"}'
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": 0.4,
        "max_tokens": 150,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
        bridge = result.get("bridge", "").strip()
        return bridge or None
    except Exception:
        return None


def generate_skill_path(
    ideal_careers: list[str],
    current_function: str,
    core_gaps: list[str],
    bridge_gaps: list[str],
) -> str | None:
    """Generate a 2-3 sentence aspirational path summary explaining how to work toward
    the student's ideal careers from their current position.

    Returns None if LLM is unavailable.
    """
    if not DEEPSEEK_KEY or not ideal_careers:
        return None

    prompt = (
        "A student wants to work toward their dream career but has skill gaps. "
        "Write 2-3 encouraging, actionable sentences about how they can bridge the gap. "
        "Mention specific steps: which skills to build first, what kind of roles to look for, "
        "what experience to gain. Be realistic but optimistic.\n\n"
        f"Current field: {current_function}\n"
        f"Dream career(s): {', '.join(ideal_careers)}\n"
        f"Key skills they NEED to build: {', '.join(core_gaps[:5])}\n"
        f"Bridge skills to develop: {', '.join(bridge_gaps[:5])}\n\n"
        'Return EXACTLY this JSON (no markdown):\n'
        '{"path": "<2-3 sentence career path guidance>"}'
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": 0.4,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
        path = result.get("path", "").strip()
        return path or None
    except Exception:
        return None
