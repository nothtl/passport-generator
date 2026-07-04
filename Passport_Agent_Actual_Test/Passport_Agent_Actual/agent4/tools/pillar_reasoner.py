"""
Pillar Reasoner — generates 3-sentence LLM reasoning for EC, GC, RFF, and CR.
CT and CI already produce their own narrative fields (thinking_arc / innovation_arc).
"""
import json
import re

from .gemini_client import call_gemini, GEMINI_API_KEY

_PILLAR_DEFS = {
    "EC": (
        "Effective Communicator",
        "verbal, written, interpersonal, and cross-cultural communication skills",
        {"Verbal": "Verbal", "Written": "Written",
         "Interpersonal": "Interpersonal", "CrossCultural": "Cross-Cultural"},
    ),
    "GC": (
        "Global Citizen",
        "empathy, community engagement, cultural awareness, network building, and volunteering",
        {"D1_Empathy": "Empathy (0–1)", "D2_Community": "Community (0–1)",
         "D3_Cultural": "Cultural (0–1)", "D4_Network": "Network (0–1)",
         "D5_Volunteering": "Volunteering (0–1)"},
    ),
    "RFF": (
        "Reflective & Future-Focused",
        "self-reflection, goal setting, future career clarity, and college preparation",
        {"D1_SelfReflection": "Self-Reflection (0–1)", "D2_GoalSetting": "Goal Setting (0–1)",
         "D3_FutureCareer": "Future Career (0–1)", "D4_CollegePrep": "College Prep (0–1)"},
    ),
    "CR": (
        "Career Ready",
        "pre-program career exposure, foundational career skills built, technical skills development, and resume quality",
        {"C1": "Pre-Program Exposure (0–100)", "C2": "Foundation Built (0–100)",
         "C3": "Skills Dev (0–100)", "C4": "Resume (0–100)"},
    ),
}

_PILLAR_EVIDENCE_HINTS = {
    "EC": (
        "Communication signals: speaking to clients/customers, explaining or presenting ideas, "
        "writing or drafting documents, mentoring or coaching others verbally, interpreting "
        "between languages, facilitating meetings. Technical work only qualifies if it involved "
        "communicating with teammates, clients, or the public — not solo coding."
    ),
    "GC": (
        "Community and global-citizen signals: volunteering, civic engagement, cultural "
        "activities, outreach, cross-cultural collaboration, diversity work, faith-community "
        "involvement, supporting underserved populations. Prefer evidence of giving back or "
        "bridging cultural/social differences over general professional work."
    ),
    "RFF": (
        "Reflection and future-focus signals: a stated career goal or aspiration, a SMART "
        "goal, deliberate skill-building for a future career, self-described values or "
        "motivations, career exploration activities. Prefer survey responses about goals "
        "or aspirations over descriptions of daily job tasks."
    ),
    "CR": (
        "Career-readiness signals: a specific internship or job role with described "
        "responsibilities, a professional skill applied in a real work setting, "
        "NACE-aligned competencies demonstrated (communication, teamwork, leadership, "
        "critical thinking, professionalism, career development), or sessions and "
        "champion engagement that built career awareness."
    ),
}

PILLAR_SURVEY_FIELDS = {
    "EC": [
        "English - Spoken",
        "Languages",
        "Any suggestions to make the Foundational Year a better experience",
        "Did you find a way to stay in touch",
        "Did you learn something about other careers from other Career Cohorts",
        "What are three skills you have that will help you in your future career",
        "CPC What skill did you cover",
        "CPC What component skill did you cover",
    ],
    "GC": [
        "Culture Feel",
        "I understand how my cultural values can shape my career choices",
        "FY1 Ever Volunteered",
        "FY1 Hours Volunteered",
        "Pre Community Connected",
        "Community Feel (Quant)",
        "How many individuals do you know who work in the career you are interested in",
    ],
    "RFF": [
        "SMART GOAL",
        "Hope to Gain",
        "What do you hope to gain by going through this program",
        "Know How To Pursue Careers",
        "FY1 Feel College Ready and Prepped",
        "What are three adjectives that describe the person you are and why",
        "If you do not have a job, what is your ideal future career job",
        "CPC Discussion topics",
        "CPC Why",
    ],
    "CR": [
        "Total Sessions Attended",
        "Total Sessions Scheduled",
        "FY1 - Had Internship?",
        "FY1 - Had/Have Job?",
        "FY1 - Ever Volunteered?",
        "FY1 - Hours Volunteered",
        "Connected Champions",
        "CPC Session Count",
    ],
}


def _extract_survey_data(pillar_key: str, raw_fields: dict) -> str:
    """Extract found survey fields for this pillar from Agent 1 raw_data.
    raw_fields is the "fields" dict from agent1 output JSON:
    { "Field Name": { "value": ..., "status": "found"|"missing", "pillar": "EC" } }
    Returns a formatted string of found fields only.
    """
    relevant_keys = PILLAR_SURVEY_FIELDS.get(pillar_key, [])
    lines = []
    for key in relevant_keys:
        entry = raw_fields.get(key, {})
        if entry.get("status") == "found" and entry.get("value") is not None:
            val = str(entry["value"]).strip()
            if val and val.lower() not in ("nan", "none", "false", "0", ""):
                lines.append(f"  {key}: {val[:200]}")
    return "\n".join(lines) if lines else "  (no survey data found for this pillar)"


_REASON_PROMPT = """You are writing a 3-sentence professional profile note for a \
SPEAKHIRE PathCredits passport. The passport is read by the student, program staff, \
Champions (professional mentors), and potential employers. It should read like \
a professional profile summary, not a grade explanation.

Student name: {student_name}
Pillar: {pillar_name}
{pillar_name} measures: {pillar_def}

SUB-SCORES:
{sub_scores_text}

SURVEY DATA (self-reported by student, only found fields shown):
{survey_data}

STUDENT EVIDENCE FROM DOCUMENTS:
LinkedIn Experience:
{linkedin_experience}
Resume Experience:
{resume_experience}
Resume Skills: {resume_skills}
GitHub Projects:
{github_repos}

{prior_notes_block}EVIDENCE FOCUS FOR {pillar_name}:
{evidence_focus}

Write exactly 1 sentence in professional third person.

EVIDENCE SELECTION PRIORITY (follow this order strictly):
1. If SURVEY DATA contains a specific text response (a sentence, phrase, or stated goal —
   not just a number), use it. Reference the student's actual words or the specific
   activity they described. Do not paraphrase it into something generic.
2. If no useful survey text exists, use a named role from LinkedIn or Resume with a
   described activity — not just a job title.
3. Only fall back to a Likert/numeric survey score if no text evidence exists for
   this pillar.

- Go deeper than a title: reference what was actually done or described, not just
  what the student is called.
- State what this evidence demonstrates about {student_name}'s {pillar_name} qualities.
- Do not mention any score number, percentage, or pillar name.
- Do not mention gaps, missing evidence, or anything negative.
- STRICT ACCURACY RULE: Only state what is explicitly documented in the evidence above.
  Do not infer scale, frequency, or achievement beyond what is written. If the data says
  "1-10 hours" do not imply "extensive" volunteering. If the data says 1 professional
  contact, do not imply a broad professional network.
- Maximum 25 words.

Respond in this EXACT JSON format with no other text:
{
  "reasoning": "<one sentence, max 25 words>"
}"""


def _parse_reasoning(text: str) -> str | None:
    if not text:
        return None
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
    try:
        d = json.loads(text)
        if "reasoning" in d:
            return str(d["reasoning"]).strip()
    except Exception:
        pass
    # brace-matched fallback
    for i, ch in enumerate(text):
        if ch != '{':
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[i:j + 1])
                        if "reasoning" in d:
                            return str(d["reasoning"]).strip()
                    except Exception:
                        pass
                    break
    return None


def generate_reasoning(pillar_key: str, pillar_result: dict, docs: dict,
                       student_name: str = "",
                       raw_fields: dict = None,
                       prior_notes: list = None) -> str:
    """Return a 3-sentence reasoning string for the given pillar, or a fallback."""
    if pillar_key not in _PILLAR_DEFS:
        return ""
    if not GEMINI_API_KEY:
        return f"Score derived from {pillar_result.get('source', 'survey+docs')}."

    pillar_name, pillar_def, sub_label_map = _PILLAR_DEFS[pillar_key]
    sub_scores = pillar_result.get("sub_scores", {})

    sub_scores_text = "\n".join(
        f"  {sub_label_map.get(k, k)}: {round(float(v), 3)}"
        for k, v in sub_scores.items()
    )

    li = docs.get("linkedin") or {}
    github = docs.get("github") or {}
    resume = docs.get("resume") or {}

    linkedin_experience = "\n".join(
        f"  - {e.get('title', '?')} at {e.get('company', '?')} ({e.get('duration', '')}): "
        f"{(e.get('description') or '')[:200]}"
        for e in (li.get("experience") or [])
    ) or "  (none)"
    resume_experience = ((resume.get("sections") or {}).get("experience") or "").strip() or "(none)"
    resume_skills = ((resume.get("sections") or {}).get("skills") or "").strip() or "(none)"
    github_repos = "\n".join(
        f"  - {r.get('name', '?')}: {(r.get('description') or r.get('readme') or '')[:150]}"
        for r in (github.get("repos") or [])
    ) or "  (none)"

    survey_data = _extract_survey_data(pillar_key, raw_fields or {})
    evidence_focus = _PILLAR_EVIDENCE_HINTS.get(pillar_key, "")

    active_notes = [n for n in (prior_notes or []) if n and n.strip()]
    if active_notes:
        prior_notes_block = (
            "ALREADY CITED IN OTHER NOTES (do not repeat this evidence — "
            "choose a different role, activity, or survey response):\n"
            + "\n".join(f"  - {n}" for n in active_notes)
            + "\n\n"
        )
    else:
        prior_notes_block = ""

    prompt = (
        _REASON_PROMPT
        .replace("{student_name}",        student_name or "the student")
        .replace("{pillar_name}",         pillar_name)
        .replace("{pillar_def}",          pillar_def)
        .replace("{sub_scores_text}",     sub_scores_text)
        .replace("{survey_data}",         survey_data)
        .replace("{linkedin_experience}", linkedin_experience)
        .replace("{resume_experience}",   resume_experience[:600])
        .replace("{resume_skills}",       resume_skills)
        .replace("{github_repos}",        github_repos)
        .replace("{prior_notes_block}",   prior_notes_block)
        .replace("{evidence_focus}",      evidence_focus)
    )

    resp = call_gemini(prompt)
    result = _parse_reasoning(resp)
    if result:
        return result
    return f"Score derived from {pillar_result.get('source', 'survey+docs')}."
