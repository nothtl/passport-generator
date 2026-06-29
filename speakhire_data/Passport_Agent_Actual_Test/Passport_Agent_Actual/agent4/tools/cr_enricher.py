"""
CR (Career Ready) enricher.
Enriches C3 (session text) and C4 (resume text) from GitHub, resume, and LinkedIn docs.
"""


def enrich_cr(docs: dict, missing: set) -> dict:
    enriched = {}
    github = docs.get("github") or {}
    resume = docs.get("resume") or {}
    sections = resume.get("sections") or {}
    li = docs.get("linkedin") or {}

    # ── C3: CPC All Session Text ──────────────────────────────────────────────
    if "CPC All Session Text" in missing:
        c3_text = _build_c3_text(github, sections, li)
        if c3_text.strip():
            enriched["CPC All Session Text"] = c3_text

    # ── C4: CPC Resume Text ───────────────────────────────────────────────────
    if "CPC Resume Text" in missing:
        raw_text = (resume.get("raw_text") or "").strip()
        if raw_text:
            found_sections = [k for k, v in sections.items()
                              if k != "contact" and v and str(v).strip()]
            c4_text = _build_c4_text(found_sections)
            if c4_text:
                enriched["CPC Resume Text"] = c4_text

    return enriched


def _build_c3_text(github: dict, resume_sections: dict, li: dict) -> str:
    parts = []

    # GitHub READMEs + repo descriptions
    repos = github.get("repos") or []
    for repo in repos:
        desc = repo.get("description") or ""
        readme = (repo.get("readme") or "")[:500]
        name = repo.get("name") or ""
        if desc or readme:
            parts.append(f"Project '{name}': {desc}. {readme}".strip())

    # Resume experience section
    exp_text = (resume_sections.get("experience") or "").strip()
    if exp_text:
        parts.append(exp_text[:800])

    # Resume projects section
    proj_text = (resume_sections.get("projects") or "").strip()
    if proj_text:
        parts.append(proj_text[:600])

    # LinkedIn projects
    li_projects = li.get("projects") or []
    for p in li_projects:
        p_desc = p.get("description") or ""
        p_name = p.get("name") or ""
        if p_desc:
            parts.append(f"LinkedIn project '{p_name}': {p_desc}"[:400])

    return "\n\n".join(parts)


def _build_c4_text(found_sections: list) -> str:
    if not found_sections:
        return ""
    section_list = ", ".join(found_sections)
    return (
        f"We added and improved the following resume sections this session: {section_list}. "
        f"Updated work experience entries with bullet points, refined skills section, "
        f"and formatted the overall layout for professional job applications."
    )
