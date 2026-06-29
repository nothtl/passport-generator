"""
CT (Critical Thinking) scorer. Entirely from documents.
Scores 0-100 based on evidence of analytical depth, research, and systematic reasoning.
"""
import json
import re

from .gemini_client import call_gemini, GEMINI_API_KEY

_CT_PROMPT = """You are evaluating a student's critical thinking ability for a workforce \
development program. These students are high school and early college students in their \
first professional experiences — most will not have technical projects or GitHub repos. \
Critical thinking for this population manifests through how analytically they approach \
roles, problems, and learning — not through software artifacts.

STUDENT EVIDENCE:

RESUME EXPERIENCE (roles and described work):
{resume_experience}

LINKEDIN EXPERIENCE (titles, companies, and any descriptions):
{linkedin_experience}

LINKEDIN PROJECTS:
{linkedin_projects}

LINKEDIN CERTIFICATIONS:
{linkedin_certs}

RESUME LEADERSHIP:
{resume_leadership}

RESUME EDUCATION (coursework, GPA, academic achievements):
{resume_education}

GITHUB REPOS (if present — bonus signal, not expected):
{github_repos_full}

Score this student's critical thinking from 0 to 100.

WHAT CRITICAL THINKING LOOKS LIKE FOR THIS POPULATION:
- Teaching or tutoring roles: required breaking down material, assessing learner needs, \
  adapting approach — this is active analytical work regardless of the subject.
- Translation/interpretation: requires nuanced analysis of meaning across languages and \
  cultural contexts simultaneously.
- Coordination and program management: organizing multi-step workflows, making decisions \
  under constraints, synthesising information from multiple sources.
- Health, advocacy, or social service roles with described responsibilities — understanding \
  systems, navigating policies, making judgment calls.
- Academic rigour: AP/honors coursework, research papers, science projects, debate, \
  or any described academic challenge.
- Certifications in substantive areas (health, legal, technical, professional) signal \
  deliberate structured learning.
- Any role where the student describes HOW they did something, not just WHAT they did.

CALIBRATED SCORING RUBRIC (for early-career students):
80-100: Strong documented pattern of analytical initiative. Multiple roles or activities \
  where the student describes reasoning, decision-making, or problem-solving. Teaching with \
  described differentiation, research involvement, program design, or complex coordination \
  with documented approach. Evidence shows the student thinks, not just executes.
60-79: At least one clear analytical role or activity beyond simple task execution. A \
  teaching/tutoring role, a coordination role with described responsibilities, translation \
  work, a health/advocacy role requiring judgment, or academic distinction showing \
  analytical depth. One strong analytical signal is sufficient for this band.
40-59: Some analytical signals alongside standard service or execution roles. A mix of \
  described responsibilities that imply structured thinking (multi-step coordination, \
  assessing needs, synthesising information) with more passive participation roles.
20-39: Mostly execution roles with minimal described analytical component. Standard \
  service jobs, basic volunteering, or activities with no described reasoning process. \
  Shows work ethic and reliability but limited documented thinking.
0-19: No analytical signal apparent. Only a job title with no description, or entirely \
  empty evidence fields.

Calibration notes:
- Do not penalise for absence of GitHub or technical projects — they are not expected.
- A teaching aide describing how they support students is stronger CT evidence than a \
  software intern with no described work.
- Certifications in professional domains (CPR, global health, legal, language) signal \
  deliberate analytical investment — not nothing.
- Score what is there, not what is missing.

Before outputting the score: identify the 1-2 strongest analytical signals in the \
evidence, determine the rubric band, then score within the band based on how fully \
the evidence satisfies that band. A student with one clear teaching/translation role \
and thin other evidence belongs in the lower 60-79 band, not below 40.

For the thinking_arc field, write exactly 1 sentence in professional third person.
- Start with "{student_name}'s [specific role, work, or activity] demonstrates..."
- Name the exact domain or context (e.g., "multilingual interpretation for community \
  services", "student support and instructional differentiation at a public school").
- State the specific analytical skill this demonstrates.
- Do not mention any score, percentage, or pillar name.
- Do not mention gaps, missing evidence, or anything negative.
- 25-30 words maximum.

Respond in this EXACT JSON format with no other text:
{
  "ct_score": <integer 0-100>,
  "thinking_arc": "<1 sentence, 25-30 words, starts with '{student_name}s [specific role/work/activity]...', names exact domain, states analytical capability>",
  "key_evidence": "<comma-separated short labels, e.g. 'Teaching aide, CPR cert, Program coordination' — role/cert/activity names, no descriptions, max 4 words each>",
  "depth_signal": "<deep | developing | surface>"
}"""


def score_ct(docs: dict, student_name: str = "") -> dict:
    li = docs.get("linkedin") or {}
    github = docs.get("github") or {}
    resume = docs.get("resume") or {}

    linkedin_experience = "\n".join(
        f"- {e.get('title','?')} at {e.get('company','?')} ({e.get('duration','')})"
        for e in (li.get("experience") or [])
    )

    # Include README content for deeper project analysis
    repos_lines = []
    for r in (github.get("repos") or []):
        name = r.get("name", "?")
        desc = r.get("description") or ""
        readme = (r.get("readme") or "")[:400]
        entry = f"- Repo '{name}'"
        if desc:
            entry += f": {desc}"
        if readme:
            entry += f"\n  README: {readme}"
        repos_lines.append(entry)
    github_repos_full = "\n".join(repos_lines)

    resume_experience = ((resume.get("sections") or {}).get("experience") or "").strip()
    resume_leadership = ((resume.get("sections") or {}).get("leadership") or "").strip()
    resume_education  = ((resume.get("sections") or {}).get("education")  or "").strip()

    certifications = li.get("certifications") or []
    linkedin_certs = "\n".join(
        f"- {c.get('name', '?')} ({c.get('issuer', '')})"
        for c in certifications[:10]
    ) or "(none)"

    li_projects = li.get("projects") or []
    linkedin_projects = "\n".join(
        f"- '{p.get('name', '?')}': {(p.get('description') or '')[:300]}"
        for p in li_projects
    ) or "(none)"

    _has_docs = any([
        linkedin_experience, github_repos_full, resume_experience,
        resume_leadership,
        linkedin_projects not in ("", "(none)"),
        linkedin_certs not in ("", "(none)"),
        resume_education,
    ])
    if not _has_docs:
        return {
            "score": 0,
            "thinking_arc": "Insufficient document evidence to evaluate critical thinking.",
            "key_evidence": "No LinkedIn experience, GitHub projects, or resume experience found.",
            "depth_signal": "surface",
            "source": "insufficient_docs",
        }

    if not GEMINI_API_KEY:
        return {
            "score": 0,
            "thinking_arc": "GEMINI_API_KEY not set.",
            "key_evidence": "",
            "depth_signal": "surface",
            "source": "docs",
        }

    prompt = (
        _CT_PROMPT
        .replace("{student_name}",        student_name or "The student")
        .replace("{linkedin_experience}", linkedin_experience or "(none)")
        .replace("{linkedin_certs}",      linkedin_certs)
        .replace("{linkedin_projects}",   linkedin_projects)
        .replace("{github_repos_full}",   github_repos_full  or "(none)")
        .replace("{resume_experience}",   resume_experience  or "(none)")
        .replace("{resume_leadership}",   resume_leadership  or "(none)")
        .replace("{resume_education}",    resume_education   or "(none)")
    )
    resp   = call_gemini(prompt)
    parsed = _parse_scored(resp, "ct_score")

    if parsed and "ct_score" in parsed:
        return {
            "score":        max(0, min(100, int(parsed["ct_score"]))),
            "thinking_arc": parsed.get("thinking_arc", ""),
            "key_evidence": parsed.get("key_evidence", ""),
            "depth_signal": parsed.get("depth_signal", "developing"),
            "source":       "docs",
        }

    return {
        "score": 0,
        "thinking_arc": "Could not parse Gemini response for CT scoring.",
        "key_evidence": "",
        "depth_signal": "surface",
        "source": "docs",
    }


def _parse_scored(text: str, score_key: str) -> dict | None:
    """Brace-matched JSON extractor — finds the first JSON object containing score_key."""
    if not text:
        return None
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
    try:
        data = json.loads(text)
        if score_key in data:
            return data
    except Exception:
        pass
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
                    candidate = text[i:j + 1]
                    try:
                        data = json.loads(candidate)
                        if score_key in data:
                            return data
                    except Exception:
                        pass
                    break
    return None
