"""
CI (Creative Innovator) scorer. Entirely from documents.
Scores 0-100 based on evidence of original thinking, novel project ideas, and
entrepreneurial or design-driven approaches.
"""
import json
import re

from .gemini_client import call_gemini, GEMINI_API_KEY

_CI_PROMPT = """You are evaluating a student's creative innovation ability for a workforce \
development program. These students are high school and early college students in their \
first professional experiences — most will not have GitHub repos or tech products. \
Creative innovation for this population means: do they create original work, take \
self-directed initiative, find new approaches to real problems, or produce something \
that goes beyond simply following existing instructions?

STUDENT EVIDENCE:

RESUME EXPERIENCE (roles and described work):
{resume_experience}

LINKEDIN ABOUT / HEADLINE:
{linkedin_about}

LINKEDIN EXPERIENCE (titles, companies, and any descriptions):
{linkedin_experience}

LINKEDIN PROJECTS:
{linkedin_projects}

LINKEDIN SOFT SKILLS:
{soft_skills}

RESUME LEADERSHIP:
{resume_leadership}

GITHUB REPOS (if present — bonus signal, not expected):
{github_repos_full}

Score this student's creative innovation from 0 to 100.

WHAT CREATIVE INNOVATION LOOKS LIKE FOR THIS POPULATION:
- Content creation roles: graphic design, social media, digital media, photography, \
  video — if the student is producing original material, not just scheduling posts.
- Performing arts: acting, modeling, music, dance — creative expression with a \
  described portfolio, production, or original work.
- Founding or co-founding: starting a club, program, initiative, or community effort \
  from scratch — shows the student identified a gap and created a solution.
- Event design and production: designing and running an original event (not just attending) \
  — stage management, community events with described creative elements.
- Advocacy or community organizing with described strategy: creating persuasive materials, \
  designing outreach campaigns, writing original content for a cause.
- Any role with a described self-directed creative contribution — "I designed...", \
  "I created...", "I launched...", "I built..." signals agency over the output.
- Self-initiated side projects in any domain — entrepreneurial, artistic, or social.
- Applying existing skills in an unexpected or personal domain beyond the job requirement.

CALIBRATED SCORING RUBRIC (for early-career students):
80-100: Multiple documented instances of original creative initiative. Could be: a sustained \
  content creation role producing original material, founding an organization or program, \
  professional performing arts work, or multiple self-directed creative projects. Creative \
  agency is the dominant pattern across their documented history.
60-79: At least one clearly self-initiated or original creative act. A content creation role \
  with described original work, founding something, a performing arts role with described \
  productions, or an advocacy role where they designed original materials. One strong \
  creative signal is sufficient for this band.
40-59: Some creative elements within otherwise standard roles. Participating in creative \
  productions without leading, a graphic design or social media role with limited description, \
  or creative contributions within a structured program. Some originality but mostly following \
  others' creative direction.
20-39: Standard participation and execution roles. Volunteering, retail, administrative work \
  with no described creative component. Shows initiative but not creative originality.
0-19: No creative signal apparent from available evidence.

Calibration notes:
- Do not penalise for absence of GitHub or tech products — they are not expected.
- A graphic designer creating original content for a nonprofit scores higher on CI than \
  a software intern who executed assigned tasks.
- Performing arts with described work (auditions, productions, portfolios) is strong CI \
  evidence — it is professional creative work.
- "Content Creator" or "Graphic Design" in a role title WITH any description is a real \
  signal. Without any description it is weak but not zero.
- Score what is there, not what is missing.

Before outputting the score: identify the 1-2 most original pieces of evidence, determine \
which rubric band they place the student in, and score within that band based on how \
completely those signals satisfy the band criteria. One genuine novel project with limited \
additional originality sits in the lower half of the 60-79 band, not at 79.

CRITICAL THINKING PROFILE NOTE (already written for this student):
"{ct_arc}"

The innovation_arc MUST reference different evidence than the CT note above. \
Choose the strongest creative signal from the docs that is NOT already described \
in the CT note. This rule applies to every student.

CI evidence must show ORIGINAL CREATION — producing, founding, designing, building, \
or performing. Community service, bilingual support, and general career readiness skills \
belong to GC/EC/CR pillars unless a distinctive creative element is explicitly described \
in the evidence.
{forbidden_block}

For the innovation_arc field, write exactly 1 sentence in professional third person.
- Start with "{student_name}'s [specific project, initiative, role, or work] demonstrates..."
- The evidence can be ANY signal of creative initiative or innovation: a novel project, \
  a self-initiated product, founding or launching something, a creative application of \
  technology in an unexpected domain, design work, or an entrepreneurial role — not limited \
  to GitHub repos.
- Name the exact application domain or what was built/created (e.g., "real-time compensation \
  data aggregation for medical professionals", "a student tech community from scratch").
- State what creative initiative or originality this demonstrates.
- Do not mention any score, percentage, or pillar name.
- Do not mention gaps, missing evidence, or anything negative.
- 25-30 words maximum.

Respond in this EXACT JSON format with no other text:
{
  "ci_score": <integer 0-100>,
  "innovation_arc": "<1 sentence, 25-30 words, starts with '{student_name}'s [specific work/project/initiative]...', names exact domain/what was created, states creative quality>",
  "key_evidence": "<comma-separated short labels only, e.g. 'AGGRATE, ml-from-scratch-od, solutionschallenge' — project/repo names, no descriptions, max 4 words each>",
  "innovation_signal": "<pioneering | developing | conventional>"
}"""


def score_ci(docs: dict, student_name: str = "", ct_arc: str = "", forbidden: str = "") -> dict:
    li = docs.get("linkedin") or {}
    github = docs.get("github") or {}
    resume = docs.get("resume") or {}

    linkedin_experience = "\n".join(
        f"- {e.get('title','?')} at {e.get('company','?')} ({e.get('duration','')})"
        for e in (li.get("experience") or [])
    )
    headline = (li.get("headline") or "").strip()
    about    = (li.get("about") or "").strip()
    linkedin_about = f"{headline}\n{about}".strip()

    # Include README content to assess originality of projects
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

    li_projects = li.get("projects") or []
    linkedin_projects = "\n".join(
        f"- '{p.get('name', '?')}': {(p.get('description') or '')[:300]}"
        for p in li_projects
    ) or "(none)"

    skills = li.get("skills") or {}
    soft_skills = ", ".join((skills.get("soft") or [])[:10]) or "(none)"

    _has_docs = any([
        linkedin_about, github_repos_full, resume_experience,
        resume_leadership,
        linkedin_projects not in ("", "(none)"),
        linkedin_experience,
        soft_skills not in ("", "(none)"),
    ])
    if not _has_docs:
        return {
            "score": 0,
            "innovation_arc": "Insufficient document evidence to evaluate creative innovation.",
            "key_evidence": "No LinkedIn, GitHub projects, or resume experience found.",
            "innovation_signal": "conventional",
            "source": "insufficient_docs",
        }

    if not GEMINI_API_KEY:
        return {
            "score": 0,
            "innovation_arc": "GEMINI_API_KEY not set.",
            "key_evidence": "",
            "innovation_signal": "conventional",
            "source": "docs",
        }

    forbidden_block = (
        f"STRICTLY FORBIDDEN — the following terms must NOT appear in innovation_arc: {forbidden}\n"
        f"Select a completely different project, role, or initiative for the innovation_arc."
        if forbidden else ""
    )
    prompt = (
        _CI_PROMPT
        .replace("{student_name}",        student_name or "The student")
        .replace("{ct_arc}",              ct_arc or "(none)")
        .replace("{forbidden_block}",     forbidden_block)
        .replace("{linkedin_about}",      linkedin_about     or "(none)")
        .replace("{linkedin_experience}", linkedin_experience or "(none)")
        .replace("{soft_skills}",         soft_skills)
        .replace("{linkedin_projects}",   linkedin_projects)
        .replace("{github_repos_full}",   github_repos_full  or "(none)")
        .replace("{resume_experience}",   resume_experience  or "(none)")
        .replace("{resume_leadership}",   resume_leadership  or "(none)")
    )
    resp   = call_gemini(prompt)
    parsed = _parse_scored(resp, "ci_score")

    if parsed and "ci_score" in parsed:
        return {
            "score":            max(0, min(100, int(parsed["ci_score"]))),
            "innovation_arc":   parsed.get("innovation_arc", ""),
            "key_evidence":     parsed.get("key_evidence", ""),
            "innovation_signal": parsed.get("innovation_signal", "developing"),
            "source":           "docs",
        }

    return {
        "score": 0,
        "innovation_arc": "Could not parse Gemini response for CI scoring.",
        "key_evidence": "",
        "innovation_signal": "conventional",
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
