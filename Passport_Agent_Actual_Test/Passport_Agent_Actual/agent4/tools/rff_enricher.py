"""
RFF (Reflective & Future Focused) enricher.
Fills missing RFF-relevant fields from LinkedIn and resume documents.
All signal detection uses Gemini — no keyword lists.
The skills field is intentionally NOT set here — ec_enricher owns it.
"""
import json
import re

from .gemini_client import call_gemini, GEMINI_API_KEY

_RFF_SIGNALS_PROMPT = """You are analyzing a student's LinkedIn experience and Champion
Peer Coach (CPC) session log to detect career and college readiness signals.

LINKEDIN EXPERIENCE (company names and any descriptions):
{linkedin_experience}

CPC SESSION OBSERVATIONS (notes from professional mentor/coach sessions):
{cpc_session_text}

Answer each question based strictly on what is written — not assumptions.

1. has_speakhire_on_linkedin — Does any LinkedIn company name or job description contain
   "SPEAKHIRE" or "Speak Hire"? Answer true or false.

2. has_goal_discussion — Does the CPC session text describe discussing a career goal,
   SMART goal, career plan, life objective, or future aspirations with the student?
   Answer true or false.

3. goal_text — If has_goal_discussion is true, extract or summarize the specific goal
   discussed in 150 characters or fewer. If false, return null.

4. has_career_discussion — Does the CPC session text describe career exploration, job
   search, internship opportunities, professional development, or industry knowledge?
   Answer true or false.

5. has_college_discussion — Does the CPC session text describe college applications,
   GPA, transcripts, academic majors, college preparation, or university readiness?
   Answer true or false.

Respond in this EXACT JSON format with no other text:
{
  "has_speakhire_on_linkedin": <true or false>,
  "has_goal_discussion": <true or false>,
  "goal_text": <"string up to 150 chars" or null>,
  "has_career_discussion": <true or false>,
  "has_college_discussion": <true or false>
}"""


def _infer_rff_signals(docs: dict, cpc_session_text: str = "") -> dict | None:
    """Gemini inference for RFF-relevant signals from LinkedIn and CPC session text.
    Returns dict with 5 keys, or None on failure or no content to analyze.
    """
    if not GEMINI_API_KEY:
        return None

    li = docs.get("linkedin") or {}
    experience = li.get("experience") or []
    linkedin_experience = "\n".join(
        f"- {e.get('title', '?')} at {e.get('company', '?')}: "
        f"{(e.get('description') or '')[:200]}"
        for e in experience
    )

    if not linkedin_experience and not cpc_session_text:
        return None

    prompt = (
        _RFF_SIGNALS_PROMPT
        .replace("{linkedin_experience}", linkedin_experience or "(none)")
        .replace("{cpc_session_text}",    cpc_session_text    or "(none)")
    )

    resp = call_gemini(prompt)
    if not resp:
        return None
    resp = re.sub(r'```(?:json)?\s*', '', resp).replace('```', '').strip()
    try:
        d = json.loads(resp)
        return {
            "has_speakhire_on_linkedin": bool(d.get("has_speakhire_on_linkedin", False)),
            "has_goal_discussion":       bool(d.get("has_goal_discussion",       False)),
            "goal_text":                 d.get("goal_text"),
            "has_career_discussion":     bool(d.get("has_career_discussion",     False)),
            "has_college_discussion":    bool(d.get("has_college_discussion",    False)),
        }
    except Exception:
        return None


def enrich_rff(docs: dict, missing: set, cpc_session_text: str = "") -> dict:
    enriched = {}
    li = docs.get("linkedin") or {}
    resume = docs.get("resume") or {}
    sections = resume.get("sections") or {}

    headline = (li.get("headline") or "").strip()
    about    = (li.get("about") or "").strip()
    experience    = li.get("experience") or []
    certifications = li.get("certifications") or []

    # ── Infer RFF signals from LinkedIn + CPC in one Gemini call ─────────────
    rff_fields_needing_inference = {
        "FY helped realize doing well connects to my career goals",
        "SMART GOAL",
        "Know How To Pursue Careers",
        "I feel more prepared for my future career",
        "I feel ready and prepared for college",
        "I feel I am now more prepared for college",
    }
    if rff_fields_needing_inference & missing:
        signals = _infer_rff_signals(docs, cpc_session_text)
    else:
        signals = None

    # ── D1: Self-Reflection text fields ──────────────────────────────────────
    if "What are three adjectives that describe the person you are and why" in missing:
        adj_proxy = (headline + ". " + about).strip()
        if adj_proxy.strip(".").strip():
            enriched["What are three adjectives that describe the person you are and why"] = adj_proxy

    # ── D2: SMART Goal ───────────────────────────────────────────────────────
    most_recent_title = experience[0].get("title", "") if experience else ""
    smart_proxy = None
    if headline or most_recent_title:
        parts = []
        if headline:
            parts.append(f"Career goal: {headline}.")
        if most_recent_title:
            parts.append(f"Current role: {most_recent_title}.")
        smart_proxy = " ".join(parts)

    if "SMART GOAL" in missing and smart_proxy:
        enriched["SMART GOAL"] = smart_proxy

    if "Remember the SMART Goal you set - next round" in missing and smart_proxy:
        enriched["Remember the SMART Goal you set - next round"] = smart_proxy

    # ── D3: Future Career ────────────────────────────────────────────────────
    if "If you do not have a job, what is your ideal future career job" in missing and headline:
        enriched["If you do not have a job, what is your ideal future career job"] = headline

    if "I feel more prepared for my future career" in missing:
        if len(certifications) >= 5 or len(experience) >= 4:
            enriched["I feel more prepared for my future career"] = "True"

    # ── D3: FY SPEAKHIRE connection ───────────────────────────────────────────
    if signals and signals["has_speakhire_on_linkedin"]:
        if "FY helped realize doing well connects to my career goals" in missing:
            enriched["FY helped realize doing well connects to my career goals"] = "True"

    # ── D3: Know How To Pursue Careers ───────────────────────────────────────
    if "Know How To Pursue Careers" in missing:
        score = min(4 + len(certifications) // 5, 7)
        if len(certifications) > 0 or len(experience) >= 3:
            enriched["Know How To Pursue Careers"] = score

    # ── D4: College Prep — numeric fields ────────────────────────────────────
    education_text = (sections.get("education") or "").strip()
    n_exp   = len(experience)
    n_certs = len(certifications)

    if "I feel ready and prepared for college" in missing:
        if education_text and (n_certs >= 3 or n_exp >= 3):
            enriched["I feel ready and prepared for college"] = 4
        elif education_text:
            enriched["I feel ready and prepared for college"] = 3

    if "I feel I am now a stronger candidate for college and careers" in missing:
        strength_score = min(4 + n_certs // 3 + n_exp // 4, 9)
        if n_certs > 0 or n_exp >= 2:
            enriched["I feel I am now a stronger candidate for college and careers"] = strength_score

    if "I feel I am now more prepared for college" in missing:
        prep_score = min(3 + n_certs // 3 + n_exp // 5, 8)
        if n_certs > 0 or n_exp >= 2:
            enriched["I feel I am now more prepared for college"] = prep_score

    # ── D2/D3: boost from CPC discussion topics (via Gemini inference) ────────
    if signals:
        if signals["has_goal_discussion"]:
            if "SMART GOAL" in missing and "SMART GOAL" not in enriched:
                goal_desc = signals.get("goal_text") or f"Career development discussed with Champion: {cpc_session_text[:200]}"
                enriched["SMART GOAL"] = goal_desc
            if "Know How To Pursue Careers" in missing and "Know How To Pursue Careers" not in enriched:
                enriched["Know How To Pursue Careers"] = 5

        if signals["has_career_discussion"]:
            if "I feel more prepared for my future career" in missing:
                enriched["I feel more prepared for my future career"] = "True"

        if signals["has_college_discussion"]:
            if "I feel ready and prepared for college" in missing and "I feel ready and prepared for college" not in enriched:
                enriched["I feel ready and prepared for college"] = 4
            if "I feel I am now more prepared for college" in missing and "I feel I am now more prepared for college" not in enriched:
                enriched["I feel I am now more prepared for college"] = 6

    return enriched
