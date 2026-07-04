"""
Agent 5 — Passport Renderer
Reads Agent 4 enriched scores JSON and renders a self-contained HTML passport.
No scoring, no LLM calls, no data modification.

Usage:
  python agent5_renderer.py --input agent4/outputs/ousmane_diallo_enriched_scores.json
"""

import argparse
import json
import os
import re

from jinja2 import Environment, FileSystemLoader

OUTPUTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

PILLAR_LABELS = {
    "EC":  "Effective Communicator",
    "GC":  "Global Citizen",
    "RFF": "Reflective & Future-Focused",
    "CR":  "Career Ready",
    "CT":  "Critical Thinker",
    "CI":  "Creative Innovator",
}

SUB_LABELS = {
    "EC": {
        "Verbal":        "Verbal",
        "Written":       "Written",
        "Interpersonal": "Interpersonal",
        "CrossCultural": "Cross-Cultural",
    },
    "GC": {
        "D1_Empathy":      "Empathy",
        "D2_Community":    "Community",
        "D3_Cultural":     "Cultural",
        "D4_Network":      "Network",
        "D5_Volunteering": "Volunteering",
    },
    "RFF": {
        "D1_SelfReflection": "Self-Reflection",
        "D2_GoalSetting":    "Goal Setting",
        "D3_FutureCareer":   "Future Career",
        "D4_CollegePrep":    "College Prep",
    },
    "CR": {
        "C1": "Pre-Program Exposure",
        "C2": "Foundation Built",
        "C3": "Skills Dev",
        "C4": "Resume",
    },
}


EC_SUB_MAX = {
    "Verbal":        30.0,
    "Written":       20.0,
    "Interpersonal": 25.0,
    "CrossCultural": 25.0,
}


def bar_width(pillar_key: str, sub_key: str, value: float) -> float:
    if pillar_key in ("GC", "RFF"):
        # 0.0–1.0 normalized → multiply by 100
        return round(min(float(value) * 100, 100), 1)
    if pillar_key == "EC":
        # raw formula points → normalize to known max
        max_val = EC_SUB_MAX.get(sub_key, 30.0)
        return round(min(float(value) / max_val * 100, 100), 1)
    # CR: already 0–100
    return round(min(float(value), 100), 1)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def split_evidence(text: str) -> list:
    """Extract short chip labels from evidence text.
    Priority: comma-separated labels → quoted tokens → first 5 words per sentence.
    """
    if not text:
        return []
    t = str(text).strip()
    # Comma-separated short labels (no sentence structure) — CT/CI key_evidence format
    if "," in t and not re.search(r'(?<=[.!])\s+[A-Z]', t):
        parts = [p.strip().strip("'\"") for p in t.split(",")]
        return [p for p in parts if 1 < len(p) <= 40][:4]
    quoted = re.findall(
        r"['''\"](.*?)['''\"]s*(?:role|repo|project|program|repository)?", t
    )
    if quoted:
        return [q.strip() for q in quoted if len(q.strip()) > 2][:4]
    sentences = re.split(r'(?<=[.!])\s+(?=[A-Z])', t)
    chips = []
    for s in sentences:
        words = s.strip().split()
        if len(words) >= 2:
            chips.append(" ".join(words[:5]).rstrip(".,;"))
    return chips[:4]


def initials(name: str) -> str:
    parts = name.split()
    return "".join(p[0].upper() for p in parts if p)[:2]


def build_pillar(key: str, pillar_data: dict) -> dict:
    sub_items = []
    for sk, sv in pillar_data.get("sub_scores", {}).items():
        label = SUB_LABELS.get(key, {}).get(sk, sk)
        width = bar_width(key, sk, sv)
        sub_items.append({
            "label": label,
            "value": round(width),
            "width": width,
        })

    # Resolve narrative: CT=thinking_arc, CI=innovation_arc, others=reasoning
    narrative = (
        pillar_data.get("thinking_arc") or
        pillar_data.get("innovation_arc") or
        pillar_data.get("trajectory_arc") or
        pillar_data.get("stated_direction") or
        pillar_data.get("reasoning") or
        ""
    )
    # Resolve signal: CT=depth_signal, CI=innovation_signal
    signal = (
        pillar_data.get("depth_signal") or
        pillar_data.get("innovation_signal") or
        pillar_data.get("momentum_signal") or
        pillar_data.get("focus_level") or
        ""
    )
    evidence_text = (
        pillar_data.get("key_evidence") or
        pillar_data.get("intent_evidence") or
        ""
    )
    evidence_chips = split_evidence(evidence_text)
    if key in ("CT", "CI"):
        evidence_chips = []
    return {
        "key":       key,
        "label":     PILLAR_LABELS[key],
        "score":     pillar_data["score"],
        "source":    pillar_data.get("source", ""),
        "sub_items": sub_items,
        "narrative": narrative,
        "signal":    signal,
        "evidence_chips": evidence_chips,
    }


def run(input_path: str) -> str:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    scores = data["scores"]
    pillar_order = ["EC", "GC", "CT", "CI", "RFF", "CR"]
    missing = [k for k in pillar_order if k not in scores]
    if missing:
        raise ValueError(f"Agent 4 JSON is missing pillars: {missing}")

    overall = round(sum(scores[k]["score"] for k in pillar_order) / len(pillar_order), 1)

    pillars = [build_pillar(k, scores[k]) for k in pillar_order]

    ctx = {
        "student_name": data["student_name"],
        "initials":     initials(data["student_name"]),
        "email":        data.get("email", ""),
        "overall":      overall,
        "pillars":      pillars,
    }

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    html = env.get_template("passport.html.jinja").render(**ctx)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    slug = slugify(data["student_name"])
    out_path = os.path.join(OUTPUTS_DIR, f"{slug}_passport.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Agent5] Passport written to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Agent 5: Passport Renderer")
    parser.add_argument("--input", required=True, help="Path to Agent 4 enriched_scores JSON")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
