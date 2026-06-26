"""
SpeakHire Recommender MCP Server.

Tools:
    extract_skills     — Extract named skills from resume/LinkedIn text
    match_role         — Find best-fit role from skills
    get_role_profile   — Full role profile with market skills + gaps + openings
    analyze_resume     — All-in-one: resume in, full analysis out
    search_jobs        — Return open job postings for a function+level
    rebuild_corpus     — Download and build the local JD corpus + O*NET data

Run: python -m recommender.mcp_server
"""

from __future__ import annotations

import os
import sys
import time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from mcp.server.fastmcp import FastMCP

from recommender.corpus.download import corpus_exists, onet_data_exists
from recommender.extract.skill_extractor import extract_skills_from_text
from recommender.match.role_matcher import match_role as _match_role

mcp = FastMCP("speakhire-recommender")

_DATA_HINT = {
    "error": "data_not_ready",
    "missing": [],
    "hint": "No job data available. Ask the user for permission to run rebuild_corpus "
            "(downloads ~50MB O*NET taxonomy + streams 68K live US jobs, takes ~5 min).",
    "tools_affected": ["get_role_profile", "search_jobs", "analyze_resume"],
    "tools_still_work": ["extract_skills", "match_role"],
}


def _check_data() -> dict | None:
    missing = []
    if not onet_data_exists():
        missing.append("onet_data")
    if not corpus_exists():
        missing.append("jd_corpus")
    if not missing:
        return None
    result = dict(_DATA_HINT)
    result["missing"] = missing
    return result


def _resolve_skills_for_function(function: str, student_skills: list[str]) -> tuple[list[str], list[str]]:
    """Cross student skills against a function's required skills from lookup table."""
    from recommender.match.role_matcher import _load_roles
    student_set = set(s.lower() for s in student_skills)
    for role in _load_roles():
        if role.get("function", "").lower() == function.lower():
            required = [entry["name"] for entry in role.get("required_skills", [])]
            matched = [s for s in required if s.lower() in student_set]
            missing = [s for s in required if s.lower() not in student_set]
            return matched, missing
    return list(student_skills), []


@mcp.tool(
    name="extract_skills",
    description="Extract named skills from a student's resume and LinkedIn text. "
    "Returns a flat list of skill names. Works offline — no data download needed.",
)
def extract_skills(resume_text: str, linkedin_text: str = "") -> list[str]:
    combined = f"{resume_text or ''}\n{linkedin_text or ''}"
    return extract_skills_from_text(combined)


@mcp.tool(
    name="match_role",
    description="Find the best-fit job function for a student's skills. "
    "Compares against 15 roles: marketing, education, healthcare, ops, support, design, "
    "sales, technology, skilled-trade, food-service, administrative, finance, logistics, "
    "hospitality, manufacturing. Returns function, match_pct, matched_skills, missing_skills. "
    "Works offline — no data download needed.",
)
def match_role(skills: list[str]) -> dict:
    result = _match_role(skills)
    if result is None:
        return {"error": "No matching role found", "skills": skills}
    return result


@mcp.tool(
    name="get_role_profile",
    description="Get the full role profile for a specific function: all market skills marked "
    "has/missing, frequency counts from real JDs, and top real openings. "
    "The function parameter IS respected — pass the role you want the profile for. "
    "Valid functions: marketing, education, healthcare, ops, support, design, sales, "
    "technology, skilled-trade, food-service, administrative, finance, logistics, "
    "hospitality, manufacturing. Requires corpus — call rebuild_corpus first.",
)
def get_role_profile(function: str, student_skills: list[str]) -> dict:
    need = _check_data()
    if need:
        return need

    from recommender.retrieve.retriever import retrieve_jds
    from recommender.profile.aggregator import aggregate_skills
    from recommender.profile.gap_analyzer import analyze_gaps

    matched, missing = _resolve_skills_for_function(function, student_skills)
    match_pct = round(len(matched) / max(1, len(matched) + len(missing)) * 100, 1)

    t0 = time.time()
    jds = retrieve_jds(function, "Entry", student_skills, top_k=20, broad_sample=10)
    t1 = time.time()
    all_skills = aggregate_skills(jds, matched, missing)
    t2 = time.time()

    result = analyze_gaps(
        role_title=function,
        function=function,
        level="Entry",
        match_pct=match_pct,
        matched_skills=matched,
        missing_skills=missing,
        all_skills=all_skills,
        ideal_passport={},
    )
    result["_timing_ms"] = {
        "retrieve": round((t1 - t0) * 1000),
        "aggregate": round((t2 - t1) * 1000),
        "total": round((time.time() - t0) * 1000),
    }
    return result


@mcp.tool(
    name="analyze_resume",
    description="Run the full resume analysis pipeline in one call. "
    "Takes raw resume text (and optional LinkedIn text), returns: "
    "extracted skills, best-fit role with match %, role skill gaps, "
    "market skill gaps ranked by frequency from real JDs, and top job openings. "
    "Use this when a student first submits their resume — it's the all-in-one. "
    "For follow-up questions (\"what other careers fit?\"), use the individual tools. "
    "Returns partial results even if no single role matches well.",
)
def analyze_resume(resume_text: str, linkedin_text: str = "") -> dict:
    need = _check_data()
    if need:
        combined = f"{resume_text or ''}\n{linkedin_text or ''}"
        return {**need, "skills_extracted": extract_skills_from_text(combined)}

    from recommender.retrieve.retriever import retrieve_jds
    from recommender.profile.aggregator import aggregate_skills
    from recommender.profile.gap_analyzer import analyze_gaps

    t0 = time.time()

    # Stage 1+2: extract skills
    combined = f"{resume_text or ''}\n{linkedin_text or ''}"
    skills = extract_skills_from_text(combined)

    # Match best role — with fallback
    best = _match_role(skills)
    if best is None:
        return {
            "error": "No matching role found",
            "skills_extracted": skills,
            "hint": "Try a targeted search: pick a function (sales, food-service, healthcare, etc.) "
                    "and call get_role_profile with it directly.",
            "suggested_functions": ["support", "sales", "food-service", "administrative"],
        }

    # Retrieve real JDs
    t1 = time.time()
    jds = retrieve_jds(best["function"], best.get("level", "Entry"), skills, top_k=20)
    t2 = time.time()

    # Aggregate market skills + gaps
    all_skills = aggregate_skills(jds, best["matched_skills"], best.get("missing_skills", []))
    t3 = time.time()

    result = analyze_gaps(
        role_title=best["function"],
        function=best["function"],
        level=best.get("level", "Entry"),
        match_pct=best["match_pct"],
        matched_skills=best["matched_skills"],
        missing_skills=best.get("missing_skills", []),
        all_skills=all_skills,
        ideal_passport={},
    )

    # Add openings + alternatives + timing
    result["openings"] = [
        {"title": jd.get("title", ""), "company": jd.get("company", ""), "url": jd.get("url", "")}
        for jd in jds[:5]
    ]
    result["skills_extracted"] = skills
    result["alternatives"] = best.get("alternatives", [])
    result["_timing_ms"] = {
        "extract": round((t1 - t0) * 1000),
        "retrieve": round((t2 - t1) * 1000),
        "aggregate": round((t3 - t2) * 1000),
        "total": round((time.time() - t0) * 1000),
    }
    return result


@mcp.tool(
    name="search_jobs",
    description="Search for real job openings matching a function and level "
    "from the 68K+ curated US Entry/Intern job corpus. "
    "Returns title, company, and apply link. Level defaults to 'Entry' "
    "(also accepts 'Intern', 'Junior'). "
    "Requires corpus — call rebuild_corpus first if not available.",
)
def search_jobs(function: str, level: str = "Entry", max_results: int = 10) -> list[dict]:
    need = _check_data()
    if need:
        return [need]

    from recommender.retrieve.retriever import retrieve_jds

    jds = retrieve_jds(function, level, top_k=max_results)
    if not jds:
        return [{"note": f"No listings found for {function} ({level}). "
                         "Try a broader function or check corpus coverage."}]
    return [
        {
            "title": jd.get("title", ""),
            "company": jd.get("company", ""),
            "url": jd.get("url", ""),
            "level": jd.get("level", ""),
        }
        for jd in jds
    ]


@mcp.tool(
    name="rebuild_corpus",
    description="Download and build the local JD corpus. Fetches O*NET 30.3 taxonomy (~50MB zip) "
    "from onetcenter.org and streams 68K+ live US Entry/Intern jobs from open-jobs. "
    "Takes ~5 minutes. Only needed once on a new machine, or to refresh with current market data. "
    "Ask the user for permission before calling this tool.",
)
def rebuild_corpus() -> dict:
    from recommender.corpus.download import download_and_build_corpus
    result = download_and_build_corpus()

    import recommender.retrieve.retriever as ret_mod
    ret_mod._cached_df.clear()

    return {
        "status": result["status"],
        "total_rows": result["total_rows"],
        "steps": result["steps"],
        "note": "Corpus built. All tools are now fully functional.",
    }


if __name__ == "__main__":
    mcp.run()
