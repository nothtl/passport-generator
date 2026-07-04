"""
CR (Career Ready) scorer.
C1/C2 rule-based; C3/C4 via Gemini.
Formulas and prompts taken verbatim from PathCredits_CareerReady_Top200_finalest.ipynb.
"""

from .gemini_client import call_gemini, parse_json_resp, GEMINI_API_KEY

# ---------------------------------------------------------------------------
# C3 prompt (NACE competencies rubric, exact from CR notebook)
# ---------------------------------------------------------------------------

C3_PROMPT = """You are evaluating a student intern's professional skill development based on \
observations written by their Career Pathways Champion (a working professional mentor) \
across multiple mentorship sessions.

NACE CAREER READINESS FRAMEWORK (8 competencies):
1. Career and Self-Development: goal setting, career exploration, resume work, professional identity, understanding career pathways
2. Communication: verbal communication, written communication, professional language, presentation skills, active listening, email etiquette
3. Critical Thinking: problem solving, research skills, analytical thinking, decision making, evaluating information
4. Equity and Inclusion: cultural competency, working with diverse professionals, understanding different perspectives, inclusive mindset
5. Leadership: initiative, motivating others, project ownership, mentoring peers, taking responsibility
6. Professionalism: punctuality, work ethic, professional appearance, following through on commitments
7. Teamwork: collaboration, peer support, group dynamics, shared goals
8. Technology: digital tools, software, internet research, productivity apps

CHAMPION OBSERVATIONS (across all sessions):
{cpc_session_text}

Respond in this EXACT JSON format with no other text:
{{
  "c3_score": <integer 0-100>,
  "nace_competencies_addressed": [<list of competency names from the 8 above>],
  "score_rationale": "<one sentence explaining the score>",
  "dominant_skills": "<2-3 specific skills the Champion logged most>"
}}

SCORING RUBRIC:
- 80-100: Champion logged rich, specific skill development across 4+ NACE competencies.
- 60-79: Champion logged solid skill development across 2-3 NACE competencies.
- 40-59: Champion logged some skill coverage but limited specificity or depth.
- 20-39: Champion logged minimal skill content.
- 0-19: No meaningful skill content in Champion's observations.

ESL-AWARE: Focus on SUBSTANCE of skills covered, not writing style.
If Champion observations are too vague or minimal to assess, score 30."""

# ---------------------------------------------------------------------------
# C4 prompt (exact from CR notebook)
# ---------------------------------------------------------------------------

C4_PROMPT = """You are reviewing notes written by a Career Pathways Champion (a professional mentor) \
after each mentorship session with a high school intern in New York City.

The notes below are the Champion's responses to the question:
"We added the following to the Intern resume:"

These notes were collected across multiple sessions.

CHAMPION'S RESUME NOTES (across all sessions):
{cpc_resume_text}

YOUR TASK:
Decide whether the Champion genuinely worked on and built or improved this intern's resume \
during the program.

Answer TRUE if:
- The Champion added specific sections, content, or improvements to the resume \
(e.g. "added work experience", "updated skills section", "wrote objective statement", \
"added education section", "formatted resume", "added internship experience", \
"reviewed and edited resume", "personal branding section added")
- Even partial resume work counts — adding one section is enough for TRUE

Answer FALSE if:
- The Champion never worked on the resume at all
- All entries say things like "nothing", "N/A", "-", "no", "not yet", \
"will do next time", "ran out of time", "interns couldn't find resume"

These are immigrant youth interns writing in English as a second language, \
so Champions may write briefly — brief but substantive counts as TRUE

Respond in this EXACT JSON format with no other text:
{{
  "resume_built": <true or false>,
  "confidence": <"high", "medium", or "low">,
  "reason": "<one sentence explaining your decision>",
  "key_evidence": "<the specific text that most influenced your decision>"
}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hours(val) -> float | None:
    if val is None:
        return None
    v = str(val).strip()
    if v.lower() in ('', 'nan', 'none', 'false', '0'):
        return None
    try:
        return float(v)
    except ValueError:
        try:
            return float(v.split('-')[0])
        except (ValueError, IndexError):
            return None


def _to_bool(val) -> bool:
    if val is None:
        return False
    sv = str(val).strip().lower()
    if sv in ('true', 'yes', '1', '1.0'):
        return True
    if sv in ('false', 'no', '0', '0.0', ''):
        return False
    try:
        return float(val) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# C1 — Pre-Program Exposure
# ---------------------------------------------------------------------------

def _score_c1(fields: dict) -> int:
    ever_vol = _to_bool(fields.get("FY1 - Ever Volunteered?"))
    hours    = fields.get("FY1 - Hours Volunteered")
    had_int  = _to_bool(fields.get("FY1 - Had Internship?"))
    had_job  = _to_bool(fields.get("FY1 - Had/Have Job?"))

    h = _parse_hours(hours)
    if h is None and not ever_vol:  vol_score = 0
    elif h is None and ever_vol:    vol_score = 10
    elif h < 10:                    vol_score = 10
    elif h < 30:                    vol_score = 20
    elif h < 60:                    vol_score = 30
    else:                           vol_score = 40

    return vol_score + (30 if had_int else 0) + (30 if had_job else 0)


# ---------------------------------------------------------------------------
# C2 — Foundation Building
# ---------------------------------------------------------------------------

def _score_c2(fields: dict) -> float:
    try:
        attended = float(fields.get("Total Sessions Attended") or 0)
    except (TypeError, ValueError):
        attended = 0.0

    try:
        scheduled = float(fields.get("Total Sessions Scheduled") or 0)
    except (TypeError, ValueError):
        scheduled = 0.0

    sess_score = (min(attended / scheduled, 1.0) * 60) if scheduled > 0 else 0.0

    connected = fields.get("Connected Champions") or ""
    champ_count = len([x for x in str(connected).split(',') if x.strip()])
    champ_score = min(champ_count / 5, 1.0) * 40   # denominator = 5 per spec

    return round(sess_score + champ_score, 1)


# ---------------------------------------------------------------------------
# C3 — Skills Developed (Gemini)
# ---------------------------------------------------------------------------

def _score_c3(session_text: str) -> tuple[int, str]:
    """Returns (score 0-100, status)."""
    if not session_text or str(session_text).strip() in ('', 'nan', 'None'):
        return 0, "missing"
    if not GEMINI_API_KEY:
        print("  Warning: GEMINI_API_KEY not set — C3 score = 0")
        return 0, "missing"
    filled = C3_PROMPT.replace("{cpc_session_text}", str(session_text))
    resp   = call_gemini(filled)
    parsed = parse_json_resp(resp)
    if parsed and "c3_score" in parsed:
        return max(0, min(100, int(parsed["c3_score"]))), "scored"
    return 30, "scored"   # vague/unparseable → notebook default


# ---------------------------------------------------------------------------
# C4 — Resume Confirmation (Gemini)
# ---------------------------------------------------------------------------

def _score_c4(resume_text: str) -> tuple[int, str]:
    """Returns (score 0 or 100, status)."""
    if not resume_text or str(resume_text).strip() in ('', 'nan', 'None'):
        return 0, "missing"
    if not GEMINI_API_KEY:
        print("  Warning: GEMINI_API_KEY not set — C4 score = 0")
        return 0, "missing"
    filled = C4_PROMPT.replace("{cpc_resume_text}", str(resume_text))
    resp   = call_gemini(filled)
    parsed = parse_json_resp(resp)
    if parsed and "resume_built" in parsed:
        built = str(parsed["resume_built"]).lower() in ("true", "1", "yes")
        return (100 if built else 0), "scored"
    return 0, "scored"


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_cr(fields: dict) -> dict:
    """
    fields: {canonical_name: value_or_None}
    Returns {"score": float, "sub_scores": {...}}
    """
    c1 = _score_c1(fields)
    c2 = _score_c2(fields)
    c3, c3_status = _score_c3(fields.get("CPC All Session Text"))
    c4, c4_status = _score_c4(fields.get("CPC Resume Text"))

    # CR final = average of available (non-missing) components
    available = [c1, c2]   # C1 and C2 always scored
    if c3_status == "scored":
        available.append(c3)
    if c4_status == "scored":
        available.append(c4)

    cr_score = round(sum(available) / len(available), 1) if available else 0.0

    return {
        "score": cr_score,
        "sub_scores": {
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": c4,
        },
        "_c3_status": c3_status,
        "_c4_status": c4_status,
    }
