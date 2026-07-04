"""
EC (Effective Communicator) scorer.
Gemini LLM for text fields + formula for numeric fields.
Formulas from run_all_competencies.py; prompts via PROMPT_REGISTRY in ec_llm_prompts.py.
"""

import os
import sys

# Import PROMPT_REGISTRY from pipelines/ without copy-pasting
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pipelines'))
from ec_llm_prompts import PROMPT_REGISTRY

from .gemini_client import call_gemini, parse_json_resp, GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Normalisation helpers (from run_all_competencies.py)
# ---------------------------------------------------------------------------

ENGLISH_LEVEL_MAP = {
    'very comfortable':      5,
    'comfortable':           4,
    'somewhat comfortable':  3,
    'not very comfortable':  2,
    'not comfortable':       1,
}


def _english_level(val) -> int:
    if val is None:
        return 3
    v = str(val).strip().lower()
    return ENGLISH_LEVEL_MAP.get(v, 3)


def _norm17(val) -> float:
    """1-7 Likert → 0-1"""
    try:
        return max(0.0, min(1.0, (float(val) - 1) / 6))
    except (TypeError, ValueError):
        return 0.0


def _norm15(val) -> float:
    """1-5 scale → 0-1"""
    try:
        return max(0.0, min(1.0, (float(val) - 1) / 4))
    except (TypeError, ValueError):
        return 0.0


def _norm110(val) -> float:
    """1-10 scale → 0-1"""
    try:
        return max(0.0, min(1.0, (float(val) - 1) / 9))
    except (TypeError, ValueError):
        return 0.0


def _is_multilingual(val) -> int:
    if val is None:
        return 0
    langs = [x.strip() for x in str(val).split(',') if x.strip()]
    return 1 if len(langs) >= 2 else 0


def _to_binary(val) -> int:
    if val is None:
        return 0
    sv = str(val).strip().lower()
    if sv in ('1', 'true', 'yes', 'y', '1.0'):
        return 1
    if sv in ('0', 'false', 'no', 'n', '0.0'):
        return 0
    try:
        return 1 if float(val) > 0 else 0
    except (TypeError, ValueError):
        return 0


def _to_numeric_safe(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# LLM text scoring
# ---------------------------------------------------------------------------

def _score_text_field(text, prompt_key: str) -> float:
    """Score a text field using PROMPT_REGISTRY. Returns 1.0–5.0 (or 1.0 if missing)."""
    if text is None or str(text).strip() in ('', 'nan', '--', 'N/A', 'n/a'):
        return 1.0   # minimum; maps to 0.0 after norm15
    if not GEMINI_API_KEY:
        print("  Warning: GEMINI_API_KEY not set — EC text fields score at minimum")
        return 1.0
    prompt_template = PROMPT_REGISTRY[prompt_key]["prompt"]
    filled = prompt_template.replace("{text}", str(text))
    resp   = call_gemini(filled)
    parsed = parse_json_resp(resp)
    if parsed and "score" in parsed:
        return max(1.0, min(5.0, float(parsed["score"])))
    return 1.0


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_ec(fields: dict) -> dict:
    """
    fields: {canonical_name: value_or_None}
    Returns {"score": float, "sub_scores": {...}}
    """
    g = fields.get

    # ── Numeric inputs ───────────────────────────────────────────────────────
    eng_int   = _english_level(g("English - Spoken"))          # 1-5
    champ_rat = _norm110(g("Rate your Career Pathways Champion (1-10)"))
    comm_feel = _norm17(g("Community Feel (Quant)"))

    # ── LLM text scores (1-5 scale) ──────────────────────────────────────────
    sugg_llm  = _score_text_field(
        g("Any suggestions to make the Foundational Year a better experience"),
        "written_comm_quality"
    )
    touch_llm = _score_text_field(
        g("Did you find a way to stay in touch"),
        "written_comm_quality"
    )
    cross_llm = _score_text_field(
        g("Did you learn something about other careers from other Career Cohorts"),
        "written_comm_quality"
    )
    skill_llm = _score_text_field(
        g("What are three skills you have that will help you in your future career"),
        "written_comm_quality"
    )

    written_avg = (sugg_llm + touch_llm + cross_llm) / 3.0
    sugg_depth  = sugg_llm

    # ── V: Verbal Communication (max ~30) ────────────────────────────────────
    V = (
        _norm15(eng_int) * 10 +
        champ_rat        * 10 +
        comm_feel        *  5 +
        _norm15(sugg_depth) * 5
    )

    # ── W: Written Communication (max ~20) ───────────────────────────────────
    W = (
        _norm15(written_avg)        * 15 +
        ((skill_llm - 1.0) / 4.0)  *  5
    )

    # ── I_s: Interpersonal Skills (max ~25) ──────────────────────────────────
    conflict_val = g("Deal with conflicts - conflict management")
    listen_val   = g("Listen to others (post)")
    conflict_bonus = 1 if _to_numeric_safe(conflict_val) > 3 else 0
    I_s = (
        _norm17(listen_val)   * 10 +
        _norm17(conflict_val) * 10 +
        conflict_bonus        *  5
    )

    # ── C_s: Cross-Cultural Competence (max ~20) ─────────────────────────────
    C_s = (
        _norm17(g("Include others who are different - diversity and inclusion"))                     * 10 +
        _to_binary(g("After meeting Champions, I better understand people who are different from me")) *  5 +
        _is_multilingual(g("Languages"))                                                              *  3 +
        _norm15(eng_int)                                                                              *  2
    )

    ec_score = round(max(0.0, min(100.0, V + W + I_s + C_s)), 2)

    return {
        "score": ec_score,
        "sub_scores": {
            "Verbal":        round(V, 4),
            "Written":       round(W, 4),
            "Interpersonal": round(I_s, 4),
            "CrossCultural": round(C_s, 4),
        },
    }
