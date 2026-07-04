"""
GC (Global Citizen) scorer — pure formula, no LLM.
Formulas and weights taken verbatim from GlobalCitizen_Score_199_Students_final.ipynb.
"""


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

CULTURE_FEEL_MAP = {
    'i feel equally of both':                             1.00,
    'i feel more of the culture of my country of origin': 0.70,
    'i feel more american':                               0.60,
    'i am confused about my culture':                     0.25,
}


def _norm_1_7(val):
    try:
        return max(0.0, min(1.0, (float(val) - 1) / 6))
    except (TypeError, ValueError):
        return 0.0


def _norm_binary(val):
    if val is None:
        return 0.0
    sv = str(val).strip().upper()
    if sv in ('YES', 'TRUE', '1', '1.0'):
        return 1.0
    if sv in ('NO', 'FALSE', '0', '0.0'):
        return 0.0
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return 0.0


def _norm_hours_volunteered(val):
    if val is None:
        return 0.0
    v = str(val).strip()
    if v in ('', 'nan', 'None', 'False', '0', 'false'):
        return 0.0
    try:
        num = float(v)
    except ValueError:
        try:
            num = float(v.split('-')[0])
        except (ValueError, IndexError):
            return 0.0
    if num == 0:   return 0.0
    if num < 10:   return 0.1
    if num < 20:   return 0.3
    if num < 30:   return 0.5
    if num < 60:   return 0.7
    return 1.0


def _norm_culture_feel(val):
    if val is None:
        return 0.5
    v = str(val).strip().lower()
    if v in CULTURE_FEEL_MAP:
        return CULTURE_FEEL_MAP[v]
    try:
        num = float(val)
        return max(0.0, min(1.0, (num - 1) / 6))
    except (TypeError, ValueError):
        return 0.5


def _norm_career_network(val):
    """How many individuals do you know... — treat as count, cap at 10."""
    try:
        return min(float(val) / 10.0, 1.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_gc(fields: dict) -> dict:
    """
    fields: {canonical_name: value_or_None}
    Returns {"score": float, "sub_scores": {...}}
    """
    g = fields.get

    # ── D1 Empathy & Humility (weight 0.30) ─────────────────────────────────
    d1 = (
        _norm_1_7(g("Pre Empathy"))                                                      * 0.05 +
        _norm_1_7(g("Pre Humble"))                                                       * 0.05 +
        _norm_1_7(g("Pre Listen"))                                                       * 0.05 +
        _norm_1_7(g("Pre Include Others Who Are Different"))                             * 0.05 +
        _norm_1_7(g("Pre Deal with Conflicts"))                                          * 0.05 +
        _norm_1_7(g("Pre Lead With Authenticity"))                                       * 0.05 +
        _norm_1_7(g("Listen to others"))                                                 * 0.15 +
        _norm_1_7(g("Deal with conflicts with other people conflict management"))        * 0.15 +
        _norm_1_7(g("Include others who are different from you diversity and inclusion"))* 0.15 +
        _norm_1_7(g("Reflect if you have been in a similar situation as someone you are trying to help non positional leadership")) * 0.20
    )

    # ── D2 Community Feel (weight 0.20) ──────────────────────────────────────
    d2 = (
        _norm_1_7(g("Pre Community Connected")) * 0.15 +
        _norm_1_7(g("Community Feel (Quant)"))  * 0.40 +
        _norm_culture_feel(g("Culture Feel"))   * 0.45
    )

    # ── D3 Cultural Competency (weight 0.20) ─────────────────────────────────
    d3 = (
        _norm_1_7(g("I understand how my cultural values can shape my career choices"))  * 0.30 +
        _norm_binary(g("After meeting Champions and working with other peers in my SPEAKHIRE Internship Rounds I better understand people who are different from me")) * 0.30 +
        _norm_binary(g("Were you introduced to diverse career professionals who you can relate to during this Internship Round")) * 0.40
    )

    # ── D4 Network & Growth (weight 0.20) ────────────────────────────────────
    d4 = (
        _norm_binary(g("Meeting with my Champions during school helped me feel like I belong in school"))             * 0.15 +
        _norm_binary(g("This SPEAKHIRE Foundational Year helped me understand the value of building a strong network")) * 0.15 +
        _norm_binary(g("I feel more engaged in school and participate more than before"))                             * 0.15 +
        _norm_binary(g("I made new friends during the Foundational Year"))                                            * 0.15 +
        _norm_career_network(g("How many individuals do you know who work in the career you are interested in"))      * 0.40
    )

    # ── D5 Volunteering (weight 0.10) ────────────────────────────────────────
    d5 = (
        _norm_binary(g("FY1 Ever Volunteered"))            * 0.35 +
        _norm_hours_volunteered(g("FY1 Hours Volunteered")) * 0.65
    )

    gc_score = round(
        (d1 * 0.30 + d2 * 0.20 + d3 * 0.20 + d4 * 0.20 + d5 * 0.10) * 100,
        1
    )

    return {
        "score": gc_score,
        "sub_scores": {
            "D1_Empathy":      round(d1, 4),
            "D2_Community":    round(d2, 4),
            "D3_Cultural":     round(d3, 4),
            "D4_Network":      round(d4, 4),
            "D5_Volunteering": round(d5, 4),
        },
    }
