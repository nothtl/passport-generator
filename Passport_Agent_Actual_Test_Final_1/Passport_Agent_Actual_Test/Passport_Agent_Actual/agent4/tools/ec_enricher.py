"""
EC (Effective Communicator) enricher.
Fills missing EC-relevant fields from LinkedIn and resume documents.
All signal detection uses Gemini — no keyword lists.
Two focused calls: one for interpersonal scores, one for language signals.
"""
import json
import re

from .gemini_client import call_gemini, GEMINI_API_KEY

_SKIP = {
    "Listen to others",
    "Deal with conflicts with other people conflict management",
    "Include others who are different from you diversity and inclusion",
    "Reflect if you have been in a similar situation as someone you are trying to help non positional leadership",
    "After meeting Champions, I better understand people who are different from me",
    "After meeting Champions and working with other peers in my SPEAKHIRE Internship Rounds I better understand people who are different from me",
    "Rate your Career Pathways Champion (1-10)",
    "Community Feel (Quant)",
}

_INTERP_INFER_PROMPT = """You are estimating behavioral scores for a student's interpersonal
communication skills based on their documented activities.

The three dimensions to estimate are each on a 1-7 scale:
  1 = no evidence at all
  2 = one vague indirect signal (e.g. a team role with no interpersonal description)
  3 = one clear indirect signal (e.g. "worked with diverse peers")
  4 = one direct behavioral example described in the docs
  5 = multiple direct examples from one role or context
  6 = consistent pattern across multiple roles or contexts
  7 = sustained, central theme across the student's entire documented history

STUDENT EVIDENCE:
LinkedIn experience: {linkedin_experience}
LinkedIn soft skills: {soft_skills}
Resume leadership: {resume_leadership}
Resume experience: {resume_experience}
CPC mentor session observations (skills observed by professional mentor): {cpc_session_text}

DIMENSIONS TO ESTIMATE:
1. "listen" — evidence of roles where listening, facilitating, or responding to
   others was part of the role (mentoring, championing, advising, community facilitation,
   interpreting/translating, peer tutoring, counseling, coaching).
2. "conflict" — evidence of navigating disagreements, managing diverse groups,
   or mediating situations (student government, managing a team, cross-org coordination).
3. "include" — evidence of deliberately reaching across differences
   (multilingual community work, bilingual services, diversity-focused ambassador roles,
   outreach to underrepresented groups, interpreter/translator work).

Rules:
- Score from described behaviors only, not assumed personality.
- If a dimension has no relevant evidence, return 1. Do not inflate.
- Interpreting or translating between languages IS strong evidence for both listen (4+)
  and include (5+) — it requires active listening and bridges cultural/linguistic gaps.
- Cite the specific role or description behind each non-1 score in the reasoning field.

Respond in this EXACT JSON format with no other text:
{
  "listen": <integer 1-7>,
  "conflict": <integer 1-7>,
  "include": <integer 1-7>,
  "reasoning": "<one sentence citing the strongest interpersonal signal in the docs>"
}"""

_LANG_INFER_PROMPT = """You are detecting language signals from a student's resume and profile.

Resume skills section: {resume_skills}
LinkedIn about/headline: {linkedin_about}
Resume experience section: {resume_experience}

1. english_level — assess the student's English proficiency from ALL evidence. Choose
   exactly one of these values:
     "Very Comfortable"     — native speaker or fully fluent professional English
     "Comfortable"          — proficient, works effectively in English
     "Somewhat Comfortable" — intermediate or conversational level
     "Not Very Comfortable" — basic, limited English
     "Not Comfortable"      — elementary or no English described
   Default to "Comfortable" if no language information is present.

2. languages — list every language the student speaks as found anywhere in the evidence
   (skills section, about, experience descriptions). Include English if they communicate
   professionally in English. Return as a JSON array of strings.
   Return [] if no languages are mentioned.

Respond in this EXACT JSON format with no other text:
{
  "english_level": "<one of the 5 exact values>",
  "languages": ["<language1>", "<language2>"]
}"""


def _infer_interpersonal(docs: dict, cpc_session_text: str = "") -> dict | None:
    """Gemini inference for EC interpersonal scores (listen, conflict, include).
    No keyword gate — always calls when API key available and docs contain content.
    Returns {"listen": int, "conflict": int, "include": int} or None.
    """
    if not GEMINI_API_KEY:
        return None

    li = docs.get("linkedin") or {}
    resume = docs.get("resume") or {}
    sections = resume.get("sections") or {}

    experience = li.get("experience") or []
    linkedin_experience = "\n".join(
        f"- {e.get('title', '?')} at {e.get('company', '?')}: "
        f"{(e.get('description') or '')[:200]}"
        for e in experience
    )
    soft_skills = ", ".join((li.get("skills") or {}).get("soft") or [])
    resume_leadership = (sections.get("leadership") or "").strip()
    resume_experience = (sections.get("experience") or "").strip()

    if not any([linkedin_experience, soft_skills, resume_leadership, resume_experience, cpc_session_text]):
        return None

    prompt = (
        _INTERP_INFER_PROMPT
        .replace("{linkedin_experience}", linkedin_experience or "(none)")
        .replace("{soft_skills}",         soft_skills         or "(none)")
        .replace("{resume_leadership}",   resume_leadership   or "(none)")
        .replace("{resume_experience}",   resume_experience   or "(none)")
        .replace("{cpc_session_text}",    cpc_session_text    or "(none)")
    )

    resp = call_gemini(prompt)
    if not resp:
        return None
    resp = re.sub(r'```(?:json)?\s*', '', resp).replace('```', '').strip()
    try:
        d = json.loads(resp)
        return {
            "listen":   max(1, min(7, int(d.get("listen",   1)))),
            "conflict": max(1, min(7, int(d.get("conflict", 1)))),
            "include":  max(1, min(7, int(d.get("include",  1)))),
        }
    except Exception:
        return None


def _infer_language_signals(docs: dict) -> dict | None:
    """Gemini inference for English level and spoken languages.
    No keyword gate — always calls when API key available and docs contain content.
    Returns {"english_level": str, "languages": str} or None.
    """
    if not GEMINI_API_KEY:
        return None

    li = docs.get("linkedin") or {}
    resume = docs.get("resume") or {}
    sections = resume.get("sections") or {}

    headline = (li.get("headline") or "").strip()
    about = (li.get("about") or "").strip()
    linkedin_about = f"{headline}\n{about}".strip()
    resume_skills = (sections.get("skills") or "").strip()
    resume_experience = (sections.get("experience") or "").strip()

    if not any([linkedin_about, resume_skills, resume_experience]):
        return None

    prompt = (
        _LANG_INFER_PROMPT
        .replace("{resume_skills}",    resume_skills    or "(none)")
        .replace("{linkedin_about}",   linkedin_about   or "(none)")
        .replace("{resume_experience}", resume_experience or "(none)")
    )

    resp = call_gemini(prompt)
    if not resp:
        return None
    resp = re.sub(r'```(?:json)?\s*', '', resp).replace('```', '').strip()
    try:
        d = json.loads(resp)
        langs_raw = d.get("languages") or []
        if isinstance(langs_raw, list):
            languages = ", ".join(str(l).strip() for l in langs_raw if str(l).strip())
        else:
            languages = str(langs_raw).strip()
        return {
            "english_level": str(d.get("english_level", "Comfortable")).strip(),
            "languages":     languages,
        }
    except Exception:
        return None


def enrich_ec(docs: dict, missing: set, cpc_session_text: str = "") -> dict:
    enriched = {}
    li = docs.get("linkedin") or {}
    resume = docs.get("resume") or {}

    # ── Interpersonal behavioral scores ──────────────────────────────────────
    interp_missing = {
        "Listen to others (post)",
        "Deal with conflicts - conflict management",
        "Include others who are different - diversity and inclusion",
    }
    if interp_missing & missing:
        interp = _infer_interpersonal(docs, cpc_session_text)
        if interp:
            if "Listen to others (post)" in missing:
                enriched["Listen to others (post)"] = interp["listen"]
            if "Deal with conflicts - conflict management" in missing:
                enriched["Deal with conflicts - conflict management"] = interp["conflict"]
            if "Include others who are different - diversity and inclusion" in missing:
                enriched["Include others who are different - diversity and inclusion"] = interp["include"]

    # ── English level and languages ───────────────────────────────────────────
    lang_missing = {"English - Spoken", "Languages"}
    if lang_missing & missing:
        lang = _infer_language_signals(docs)
        if lang:
            if "English - Spoken" in missing and lang.get("english_level"):
                enriched["English - Spoken"] = lang["english_level"]
            if "Languages" in missing and lang.get("languages"):
                enriched["Languages"] = lang["languages"]

    # ── Proxy text fields (LinkedIn headline + about → writing quality signal) ─
    about    = (li.get("about") or "").strip()
    headline = (li.get("headline") or "").strip()
    profile_text = (about + " " + headline).strip()

    if "Any suggestions to make the Foundational Year a better experience" in missing and profile_text:
        enriched["Any suggestions to make the Foundational Year a better experience"] = profile_text

    if "Did you find a way to stay in touch" in missing and profile_text:
        enriched["Did you find a way to stay in touch"] = profile_text

    # ── Cross-career learning proxy (experience breadth) ─────────────────────
    experience = li.get("experience") or []
    exp_text = ", ".join(
        f"{e.get('title', '?')} at {e.get('company', '?')}"
        for e in experience
        if e.get("title") or e.get("company")
    )
    if "Did you learn something about other careers from other Career Cohorts" in missing and exp_text:
        enriched["Did you learn something about other careers from other Career Cohorts"] = exp_text

    # ── Skills proxy (LinkedIn technical skills + resume skills section) ──────
    tech = (li.get("skills") or {}).get("technical") or []
    skills_raw = (resume.get("sections") or {}).get("skills") or ""
    skills_proxy = (", ".join(tech) + " " + skills_raw).strip()
    if "What are three skills you have that will help you in your future career" in missing and skills_proxy:
        enriched["What are three skills you have that will help you in your future career"] = skills_proxy

    return {k: v for k, v in enriched.items() if k not in _SKIP}
