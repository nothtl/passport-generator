"""Generate skill progression trees showing the path from current skills to career goals."""
from __future__ import annotations

import json, os, urllib.request

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"


def generate_skill_tree(
    student_skills: list[str],
    implicit_skills: list[str],
    core_gaps: list[str],
    bridge_gaps: list[str],
    stretch_gaps: list[str],
    current_function: str,
    ideal_careers: list[str],
) -> dict:
    """Generate a skill progression tree showing current → bridge → goal.

    Returns a dict with 'tree' (text diagram) and 'steps' (list of actionable steps).
    """
    if not DEEPSEEK_KEY or not ideal_careers:
        return _default_tree(student_skills, core_gaps, ideal_careers)

    prompt = (
        "Create a skill progression tree for a student mapping their path from "
        "current skills to their dream career. Format it as a text tree diagram.\n\n"
        f"Current career direction: {current_function}\n"
        f"Dream career(s): {', '.join(ideal_careers)}\n"
        f"Skills they HAVE: {', '.join(student_skills[:10])}\n"
        f"Implicit skills (likely has): {', '.join(implicit_skills[:5])}\n"
        f"Core missing skills (needed now): {', '.join(core_gaps[:5])}\n"
        f"Bridge skills (developing): {', '.join(bridge_gaps[:5])}\n"
        f"Stretch skills (future): {', '.join(stretch_gaps[:5])}\n\n"
        "Return EXACTLY this JSON:\n"
        '{"tree": "<ASCII tree diagram showing skill progression from current → bridge → goal, with branches>", '
        '"steps": ["<step 1: what to learn first>", "<step 2>", "<step 3>", "<step 4>", "<step 5>"], '
        '"summary": "<2 sentence encouraging summary>"}\n\n'
        "Rules:\n"
        "- Tree should show: CURRENT SKILLS → BRIDGE SKILLS → GOAL SKILLS → CAREER\n"
        "- Use ASCII art: ├── └── │ symbols for branches\n"
        "- Steps should be actionable and specific (e.g., 'Take a free Codecademy Python course' not 'Learn Python')\n"
        "- Be encouraging and concrete"
    )

    payload = {
        "model": MODEL, "temperature": 0.4, "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        req = urllib.request.Request(
            DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return json.loads(body["choices"][0]["message"]["content"])
    except Exception:
        return _default_tree(student_skills, core_gaps, ideal_careers)


def _default_tree(skills: list[str], gaps: list[str], goals: list[str]) -> dict:
    """Fallback tree when LLM unavailable."""
    lines = ["CURRENT SKILLS"]
    for s in skills[:4]:
        lines.append(f"  ├── {s}")
    lines.append("  │")
    lines.append("BRIDGE SKILLS (build these)")
    for g in gaps[:3]:
        lines.append(f"  ├── {g}")
    lines.append("  │")
    lines.append(f"GOAL: {goals[0] if goals else 'your dream career'}")
    return {
        "tree": "\n".join(lines),
        "steps": [f"Build {g}" for g in gaps[:5]],
        "summary": f"Focus on building {', '.join(gaps[:3])} to reach your goal of becoming a {goals[0] if goals else 'professional'}.",
    }
