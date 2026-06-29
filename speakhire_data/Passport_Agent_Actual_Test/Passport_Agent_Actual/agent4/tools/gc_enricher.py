"""
GC (Global Citizen) enricher.
Fills missing GC-relevant fields from LinkedIn and resume documents.
All signal detection uses Gemini — no keyword lists.
"""
import json
import re

from .gemini_client import call_gemini, GEMINI_API_KEY

_SKIP = set()

_GC_INFER_PROMPT = """You are evaluating a student's community engagement, empathy, and
global-citizen signals for a workforce development program called SPEAKHIRE.

STUDENT EVIDENCE:
LinkedIn headline/about: {linkedin_about}
LinkedIn experience: {linkedin_experience}
LinkedIn soft skills: {soft_skills}
Resume leadership section: {resume_leadership}
Resume experience section: {resume_experience}
Resume skills section: {resume_skills}

Return scores for all 11 empathy/community dimensions AND answer the 6 detection questions.

─── EMPATHY & COMMUNITY SCORES (1-7 scale) ──────────────────────────────────────────────

  1 = no evidence at all
  2 = one vague indirect signal
  3 = one clear indirect signal
  4 = one direct behavioral example described in the docs
  5 = multiple direct examples from one role or context
  6 = consistent pattern across multiple roles or contexts
  7 = sustained, central theme across the student's entire documented history

Pre-program baseline (who the student appeared to be before the program):
  1. pre_empathy — general orientation toward others' feelings (mentoring background,
     support roles, family/community service references)
  2. pre_humble — openness to learning and others' perspectives (language like "learning",
     "curious", service-oriented roles)
  3. pre_listen — prior listening/facilitation behaviors (peer tutoring, counseling,
     community facilitation)
  4. pre_include — prior inclusion behaviors (multilingual background, cross-cultural
     community involvement)
  5. pre_conflict — prior conflict navigation (team sports, religious/cultural community
     leadership, peer mediation)
  6. pre_lead_auth — authentic leadership signals (founding or initiating something,
     self-described values in headline/about)

Post-program behavioral (from described roles and activities):
  7. listen — active listening in roles: mentoring, coaching, championing, facilitating
  8. conflict — navigating group tensions: student government, cross-org coordination,
     managing diverse stakeholders
  9. include — reaching across differences: bilingual outreach, diversity ambassador,
     underrepresented community programs
  10. reflect — using own experience to help others: peer mentoring, shared-experience
      coaching, community advocacy based on personal background

Community connectedness:
  11. pre_community_connected — sense of being embedded in a community before the program:
      civic memberships, religious community, cultural associations, neighborhood orgs

─── DETECTION QUESTIONS ─────────────────────────────────────────────────────────────────

Answer each based strictly on what is written in the evidence — not assumptions.

  12. has_volunteer — Did the student engage in any volunteering, community service,
      nonprofit work, food pantry, shelter, faith community service, mutual aid, informal
      community helping, or unpaid service? Look at ALL sections including leadership and
      experience descriptions — do not rely on job titles alone. Answer true or false.
  13. volunteer_hours_estimate — If has_volunteer is true, estimate the commitment level.
      Return exactly one of: "1-10", "10-20", "20-30", "30+". If false, return "0".
  14. community_roles_count — Count distinct roles that serve the community, social causes,
      underserved populations, or cultural engagement. Include unpaid, faith, and campus
      service roles. Return integer 0-7.
  15. has_speakhire — Does any LinkedIn company name or job description mention "SPEAKHIRE"
      or "Speak Hire"? Answer true or false.
  16. has_campus_role — Has the student held any campus-based role: student leadership,
      peer tutoring, academic mentoring, student government, campus ambassador, residence
      advisor, or similar? Answer true or false.
  17. network_estimate — Based on internships, professional jobs, campus roles, and
      documented professional interactions: how many professionals in their career interest
      area does this student likely know? Return integer 0-10. (0 = none documented;
      3-4 = a few internship contacts; 7+ = extensive multi-role professional history)

CALIBRATED SCORING RULES:
- Score from described evidence only — not assumed personality.
- Return 1 for any score dimension with no relevant evidence. Do not inflate.
- For has_volunteer: look carefully at leadership sections — unpaid community work is
  often listed there rather than in experience titles.
- Cite the single strongest signal for each non-1 score in the reasoning field.

Respond in this EXACT JSON format with no other text:
{
  "pre_empathy": <integer 1-7>,
  "pre_humble": <integer 1-7>,
  "pre_listen": <integer 1-7>,
  "pre_include": <integer 1-7>,
  "pre_conflict": <integer 1-7>,
  "pre_lead_auth": <integer 1-7>,
  "listen": <integer 1-7>,
  "conflict": <integer 1-7>,
  "include": <integer 1-7>,
  "reflect": <integer 1-7>,
  "pre_community_connected": <integer 1-7>,
  "has_volunteer": <true or false>,
  "volunteer_hours_estimate": "<1-10 or 10-20 or 20-30 or 30+ or 0>",
  "community_roles_count": <integer 0-7>,
  "has_speakhire": <true or false>,
  "has_campus_role": <true or false>,
  "network_estimate": <integer 0-10>,
  "reasoning": "<2 sentences citing the strongest community/empathy signals in the docs>"
}"""


def _infer_gc_signals(docs: dict) -> dict | None:
    """Gemini inference for all GC empathy/community scores and document detection signals.
    No keyword gates — always calls when API key is available and docs contain content.
    Returns a flat dict with 17 keys, or None on failure.
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
    headline = (li.get("headline") or "").strip()
    about = (li.get("about") or "").strip()
    linkedin_about = f"{headline}\n{about}".strip()
    soft_skills = ", ".join((li.get("skills") or {}).get("soft") or [])
    resume_leadership = (sections.get("leadership") or "").strip()
    resume_experience = (sections.get("experience") or "").strip()
    resume_skills = (sections.get("skills") or "").strip()

    if not any([linkedin_about, linkedin_experience, resume_leadership, resume_experience, resume_skills]):
        return None

    prompt = (
        _GC_INFER_PROMPT
        .replace("{linkedin_about}",      linkedin_about      or "(none)")
        .replace("{linkedin_experience}", linkedin_experience  or "(none)")
        .replace("{soft_skills}",         soft_skills         or "(none)")
        .replace("{resume_leadership}",   resume_leadership   or "(none)")
        .replace("{resume_experience}",   resume_experience   or "(none)")
        .replace("{resume_skills}",       resume_skills       or "(none)")
    )

    resp = call_gemini(prompt)
    if not resp:
        return None
    resp = re.sub(r'```(?:json)?\s*', '', resp).replace('```', '').strip()
    try:
        d = json.loads(resp)
        return {
            "pre_empathy":             max(1, min(7, int(d.get("pre_empathy",             1)))),
            "pre_humble":              max(1, min(7, int(d.get("pre_humble",              1)))),
            "pre_listen":              max(1, min(7, int(d.get("pre_listen",              1)))),
            "pre_include":             max(1, min(7, int(d.get("pre_include",             1)))),
            "pre_conflict":            max(1, min(7, int(d.get("pre_conflict",            1)))),
            "pre_lead_auth":           max(1, min(7, int(d.get("pre_lead_auth",           1)))),
            "listen":                  max(1, min(7, int(d.get("listen",                  1)))),
            "conflict":                max(1, min(7, int(d.get("conflict",                1)))),
            "include":                 max(1, min(7, int(d.get("include",                 1)))),
            "reflect":                 max(1, min(7, int(d.get("reflect",                 1)))),
            "pre_community_connected": max(1, min(7, int(d.get("pre_community_connected", 1)))),
            "has_volunteer":           bool(d.get("has_volunteer",           False)),
            "volunteer_hours_estimate": str(d.get("volunteer_hours_estimate", "0")),
            "community_roles_count":   max(0, min(7, int(d.get("community_roles_count",   0)))),
            "has_speakhire":           bool(d.get("has_speakhire",           False)),
            "has_campus_role":         bool(d.get("has_campus_role",         False)),
            "network_estimate":        max(0, min(10, int(d.get("network_estimate",        0)))),
        }
    except Exception:
        return None


def enrich_gc(docs: dict, missing: set) -> dict:
    enriched = {}
    li = docs.get("linkedin") or {}
    experience = li.get("experience") or []

    inferred = _infer_gc_signals(docs)

    # ── D5: Volunteering ──────────────────────────────────────────────────────
    # Always override survey False when docs confirm True — students underreport.
    if inferred:
        has_vol = inferred["has_volunteer"]
        if "FY1 Ever Volunteered" in missing:
            enriched["FY1 Ever Volunteered"] = "True" if has_vol else "False"
        elif has_vol:
            enriched["FY1 Ever Volunteered"] = "True"

        hours_est = inferred["volunteer_hours_estimate"]
        if has_vol and hours_est and hours_est != "0":
            enriched["FY1 Hours Volunteered"] = hours_est

    # ── D2: Community Feel (Quant) ────────────────────────────────────────────
    if inferred and "Community Feel (Quant)" in missing:
        n_community = inferred["community_roles_count"]
        if n_community > 0:
            enriched["Community Feel (Quant)"] = min(n_community + 2, 7)

    # ── D3: Cultural competency binary fields ─────────────────────────────────
    distinct_companies = {
        e.get("company", "").strip()
        for e in experience
        if e.get("company") and e.get("company").strip()
    }
    if "After meeting Champions and working with other peers in my SPEAKHIRE Internship Rounds I better understand people who are different from me" in missing:
        if len(distinct_companies) >= 4:
            enriched["After meeting Champions and working with other peers in my SPEAKHIRE Internship Rounds I better understand people who are different from me"] = "True"

    if "Were you introduced to diverse career professionals who you can relate to during this Internship Round" in missing:
        if len(distinct_companies) >= 4:
            enriched["Were you introduced to diverse career professionals who you can relate to during this Internship Round"] = "True"

    # ── D4: Network & Growth ─────────────────────────────────────────────────
    if inferred:
        net_estimate = inferred["network_estimate"]
        if net_estimate > 0:
            enriched["How many individuals do you know who work in the career you are interested in"] = net_estimate

        if inferred["has_speakhire"]:
            if "I made new friends during the Foundational Year" in missing:
                enriched["I made new friends during the Foundational Year"] = "True"

        if inferred["has_campus_role"]:
            if "Meeting with my Champions during school helped me feel like I belong in school" in missing:
                enriched["Meeting with my Champions during school helped me feel like I belong in school"] = "True"
            if "I feel more engaged in school and participate more than before" in missing:
                enriched["I feel more engaged in school and participate more than before"] = "True"

        if inferred["has_speakhire"] or inferred["network_estimate"] >= 5:
            if "This SPEAKHIRE Foundational Year helped me understand the value of building a strong network" in missing:
                enriched["This SPEAKHIRE Foundational Year helped me understand the value of building a strong network"] = "True"

    # ── D1 Empathy + D2 Pre Community: from inferred dict ────────────────────
    if inferred:
        field_map = {
            "Pre Empathy":             "pre_empathy",
            "Pre Humble":              "pre_humble",
            "Pre Listen":              "pre_listen",
            "Pre Include Others Who Are Different": "pre_include",
            "Pre Deal with Conflicts": "pre_conflict",
            "Pre Lead With Authenticity": "pre_lead_auth",
            "Listen to others":        "listen",
            "Deal with conflicts with other people conflict management": "conflict",
            "Include others who are different from you diversity and inclusion": "include",
            "Reflect if you have been in a similar situation as someone you are trying to help non positional leadership": "reflect",
            "Pre Community Connected": "pre_community_connected",
        }
        for field_name, infer_key in field_map.items():
            if field_name in missing:
                enriched[field_name] = inferred[infer_key]

    return {k: v for k, v in enriched.items() if k not in _SKIP}
