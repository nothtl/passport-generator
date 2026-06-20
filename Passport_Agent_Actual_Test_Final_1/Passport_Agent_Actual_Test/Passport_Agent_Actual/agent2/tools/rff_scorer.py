"""
RFF (Reflective & Future Focused) scorer.
Formula + Gemini for 5 text fields.
Formulas and prompts taken verbatim from RFF_Scoring_Pipeline_final_version.ipynb.
"""

import re
from .gemini_client import call_gemini, parse_json_resp, GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Normalisation helpers (exact from RFF notebook)
# ---------------------------------------------------------------------------

def _norm_1_7(val):
    try:
        return max(0.0, min(1.0, (float(val) - 1) / 6))
    except (TypeError, ValueError):
        return 0.0


def _norm_0_10(val):
    try:
        return max(0.0, min(1.0, float(val) / 10))
    except (TypeError, ValueError):
        return 0.0


def _norm_binary(val):
    try:
        v = float(val)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.0


KNOWN_HOPE_TAGS = [
    'gain skills i can use at a future job',
    'learn about careers',
    'get an internship',
    'learn about colleges',
    'make new friends',
    'get coaches and mentors',
    'get mentors',
    'mentoring', 'skills', 'careers', 'colleges', 'friends', 'internship',
]


def _norm_hope_tags(val):
    if val is None or str(val).strip() in ('', 'nan'):
        return 0.0
    text_lower = str(val).lower()
    matched = set()
    for tag in KNOWN_HOPE_TAGS:
        if tag in text_lower:
            if 'mentor' in tag:        matched.add('mentoring')
            elif 'skill' in tag:       matched.add('skills')
            elif 'career' in tag:      matched.add('careers')
            elif 'college' in tag:     matched.add('colleges')
            elif 'friend' in tag:      matched.add('friends')
            elif 'internship' in tag:  matched.add('internship')
            else:                      matched.add(tag)
    parts = re.split(r'[,\n]', str(val))
    for part in parts:
        p = part.strip().lower()
        if len(p) > 3 and p not in ('nan', 'y2'):
            matched.add(p[:30])
    return min(1.0, len(matched) / 6)


# ---------------------------------------------------------------------------
# LLM prompts (exact from RFF notebook)
# ---------------------------------------------------------------------------

PROMPT_ADJECTIVES = """You are evaluating a high school student self-reflection response.
The student was asked: What are three adjectives that describe the person you are and why?

These are immigrant youth in New York City. English may be their second language.
Evaluate SUBSTANCE and SELF-AWARENESS, not grammar or writing style.

STUDENT RESPONSE:
{text}

SCORING RUBRIC (return a score between 0.0 and 1.0):
- 0.0 to 0.1: Blank, single vague word (e.g. perfect), or clearly irrelevant answer
- 0.1 to 0.3: One or two generic adjectives (nice, good, kind) with no reasoning
- 0.3 to 0.5: Two or three adjectives, mostly generic, little or no reasoning
- 0.5 to 0.7: Three adjectives, at least one specific to the person, some reasoning present
- 0.7 to 0.9: Three meaningful adjectives, most specific to the person, clear reasoning for at least two
- 0.9 to 1.0: Three specific thoughtful adjectives with clear career-connected or personally meaningful reasoning for each

EXAMPLES:
- Deferential, prestigious, imminent (no reasoning) -> around 0.35
- Curious (Because I want to learn more about video game development) -> around 0.80
- I do not have any or N/A -> 0.0
- kind, nice, smart with no context -> around 0.25

Respond in this EXACT JSON format with no other text:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the score>"
}"""

PROMPT_SKILLS = """You are evaluating a high school student response about their skills.
The student was asked: What are three skills you have that will help you in your future career?

These are immigrant youth in New York City. English may be their second language.
Evaluate SUBSTANCE and CAREER-AWARENESS, not grammar or writing style.

STUDENT RESPONSE:
{text}

SCORING RUBRIC (return a score between 0.0 and 1.0):
- 0.0 to 0.1: I do not have any, blank, or a non-answer
- 0.1 to 0.3: Vague non-skills or soft personal traits not linked to any career
- 0.3 to 0.5: Names one or two real skills but does not connect them to any career
- 0.5 to 0.7: Names two or three real skills, at least one is specific and career-relevant
- 0.7 to 0.9: Names three clear career-relevant skills with some explanation or context
- 0.9 to 1.0: Names three specific well-articulated skills with strong connection to a future career

EXAMPLES:
- I do not have any as of now -> 0.05
- resilience, kindness, communication -> 0.50 (real skills, no career link)
- program management skills, Adaptability to change, Business mentality -> 0.80
- I am good at math and good drawing -> 0.45

Respond in this EXACT JSON format with no other text:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the score>"
}"""

PROMPT_SMART_GOAL = """You are evaluating a high school student SMART goal response.
The student was asked to set a SMART goal (Specific, Measurable, Achievable, Relevant, Time-bound).

These are immigrant youth in New York City. English may be their second language.
Evaluate whether this is a genuine structured goal, not grammar or writing quality.

STUDENT RESPONSE:
{text}

SCORING RUBRIC (return a score between 0.0 and 1.0):
- 0.0 to 0.1: Blank, --, or completely irrelevant
- 0.1 to 0.3: Vague aspiration with no SMART elements (e.g. I want to do well, Work hard)
- 0.3 to 0.5: Some specificity but missing most SMART elements, a wish rather than a plan
- 0.5 to 0.7: Specific and has one or two SMART elements (e.g. specific + measurable target)
- 0.7 to 0.9: Has three or more SMART elements, clearly connected to career or education
- 0.9 to 1.0: Genuinely SMART with specific, measurable, time-bound, relevant elements

EXAMPLES:
- Work Hard to achieve my goal because I know that good things take time -> 0.20
- I will improve my average to a overall 85 -> 0.55 (specific + measurable, partial SMART)
- My goal is to improve my entire grade (Specific), by 10% (Measurable) -> 0.85
- I will start my senior year with at least an 80 in all classes -> 0.65
- -- or N/A -> 0.0

Respond in this EXACT JSON format with no other text:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the score>"
}"""

PROMPT_IDEAL_JOB = """You are evaluating a high school student response about their ideal future career.
The student was asked: If you do not have a job, what is your ideal future career job?

These are immigrant youth in New York City. English may be their second language.
Evaluate SPECIFICITY and CAREER-AWARENESS, not grammar.

STUDENT RESPONSE:
{text}

SCORING RUBRIC (return a score between 0.0 and 1.0):
- 0.0 to 0.1: Non-answer: Not yet, Unsure, I do not know, N/A, blank, or placeholder
- 0.1 to 0.3: Extremely vague: a good job, something that pays well, I want to help people
- 0.3 to 0.5: Broad career field named but very general (business, science, art)
- 0.5 to 0.7: Named career area with some specificity (medicine, technology, law)
- 0.7 to 0.9: Specific named career role (Doctor, Lawyer, Software Engineer, Nurse)
- 0.9 to 1.0: Highly specific career role with specialisation or clear reasoning

EXAMPLES:
- Not yet -> 0.05
- Doctor or nursing -> 0.70
- Video Game Developer -> 0.90
- Lawyer -> 0.80
- A crime scene investigator -> 0.88
- something in business -> 0.30

Respond in this EXACT JSON format with no other text:
{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the score>"
}"""


def _score_text(text, prompt_template) -> float:
    """Call Gemini with the given prompt; return 0.0–1.0 score."""
    if text is None or str(text).strip() in ('', 'nan', '--', 'N/A', 'n/a'):
        return 0.0
    if not GEMINI_API_KEY:
        print("  Warning: GEMINI_API_KEY not set — skipping LLM scoring (score=0.0)")
        return 0.0
    filled = prompt_template.replace("{text}", str(text))
    resp = call_gemini(filled)
    parsed = parse_json_resp(resp)
    if parsed and "score" in parsed:
        return max(0.0, min(1.0, float(parsed["score"])))
    return 0.0


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_rff(fields: dict) -> dict:
    """
    fields: {canonical_name: value_or_None}
    Returns {"score": float, "sub_scores": {...}}
    """
    g = fields.get

    # ── LLM-scored text fields ───────────────────────────────────────────────
    n_adj    = _score_text(g("What are three adjectives that describe the person you are and why"), PROMPT_ADJECTIVES)
    n_skills = _score_text(g("What are three skills you have that will help you in your future career"), PROMPT_SKILLS)

    smart_text = g("SMART GOAL")
    n_smart  = _score_text(smart_text, PROMPT_SMART_GOAL)
    n_smart3 = _score_text(g("Remember the SMART Goal you set - next round"), PROMPT_SMART_GOAL)

    n_ideal  = _score_text(g("If you do not have a job, what is your ideal future career job"), PROMPT_IDEAL_JOB)

    # ── Formula-scored fields ────────────────────────────────────────────────
    n_hope1     = _norm_hope_tags(g("Hope to Gain"))
    n_hope2     = _norm_hope_tags(g("What do you hope to gain by going through this program"))
    n_pursue    = _norm_1_7(g("Know How To Pursue Careers"))
    n_prepared  = _norm_binary(g("I feel more prepared for my future career"))
    n_ready_col = _norm_1_7(g("I feel ready and prepared for college"))
    n_fy1_ready = _norm_1_7(g("FY1 Feel College Ready and Prepped"))
    n_stronger  = _norm_0_10(g("I feel I am now a stronger candidate for college and careers"))
    n_more_prep = _norm_0_10(g("I feel I am now more prepared for college"))
    n_connect   = _norm_binary(g("FY helped realize doing well connects to my career goals"))

    n_spk_insp  = _norm_0_10(g("The Speaker inspired me to think more about my future career"))
    n_spk_path  = _norm_0_10(g("The Speaker helped me think about my future career pathway"))
    n_spk_model = _norm_0_10(g("The Speaker was a relatable role model"))
    n_top_insp  = _norm_0_10(g("The topic inspired me to think more about my future career"))
    n_top_path  = _norm_0_10(g("The topic helped me think about my future career pathway"))

    # ── Dimensions ───────────────────────────────────────────────────────────
    d1 = n_adj * 0.30 + n_skills * 0.30 + n_hope1 * 0.20 + n_hope2 * 0.20
    d2 = n_smart * 0.70 + n_smart3 * 0.30
    d3 = n_pursue * 0.40 + n_prepared * 0.30 + n_ideal * 0.30
    d4 = (n_ready_col * 0.25 + n_fy1_ready * 0.20 + n_stronger * 0.20 +
          n_more_prep * 0.15 + n_connect * 0.20)
    d5 = (n_spk_insp * 0.25 + n_spk_path * 0.20 + n_spk_model * 0.20 +
          n_top_insp * 0.20 + n_top_path * 0.15)

    # D5 weight = 0 per notebook
    rff_score = round(
        (d1 * 0.25 + d2 * 0.25 + d3 * 0.25 + d4 * 0.25 + d5 * 0) * 100,
        1
    )

    return {
        "score": rff_score,
        "sub_scores": {
            "D1_SelfReflection": round(d1, 4),
            "D2_GoalSetting":    round(d2, 4),
            "D3_FutureCareer":   round(d3, 4),
            "D4_CollegePrep":    round(d4, 4),
        },
    }
