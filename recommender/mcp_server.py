"""
SpeakHire Recommender MCP Server.

Tools:
    extract_skills     — Extract named skills from resume/LinkedIn text
    match_role         — Find best-fit role from skills
    analyze_resume     — All-in-one: resume in, full analysis out
    search_jobs        — Return open job postings for a function+level
    rebuild_corpus     — Download and build the local JD corpus

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

from recommender.corpus.download import corpus_exists
from recommender.extract.skill_extractor import extract_skills_from_text
from recommender.match.ensemble_matcher import match_role as _match_role

mcp = FastMCP("speakhire-recommender")

_DATA_HINT = {
    "error": "data_not_ready",
    "hint": "No job data available. Ask the user for permission to run rebuild_corpus.",
}


@mcp.tool(
    name="extract_skills",
    description="Extract named skills from a student's resume and LinkedIn text. "
    "Works offline — no data download needed.",
)
def extract_skills(resume_text: str, linkedin_text: str = "") -> list[str]:
    combined = f"{resume_text or ''}\n{linkedin_text or ''}"
    return extract_skills_from_text(combined)


@mcp.tool(
    name="match_role",
    description="Find the best-fit job function for a student's skills. "
    "Compares against 16 functions using ML classifier trained on 2,484 resumes. "
    "Works offline — no data download needed.",
)
def match_role(skills: list[str]) -> dict:
    return {"note": "Pass raw resume text to analyze_resume for full classification"}


@mcp.tool(
    name="analyze_resume",
    description="Run the full resume analysis pipeline. Takes raw resume text, "
    "returns extracted skills, best-fit function with confidence, skill gaps "
    "from real job market data, alternatives to explore, and top job openings.",
)
def analyze_resume(resume_text: str, linkedin_text: str = "") -> dict:
    from recommender.retrieve.retriever import retrieve_jds, get_jd_skill_vocabulary, _compute_idf
    import re, math

    combined = f"{resume_text or ''}\n{linkedin_text or ''}"
    if not combined.strip():
        return {"error": "No resume text provided"}

    t0 = time.time()

    # Classify
    best = _match_role(combined)
    if not best:
        return {"error": "Could not classify resume"}

    func = best["function"]

    # Extract skills
    skills = extract_skills_from_text(combined)

    # Market-relevant skills + gaps
    jd_vocab = get_jd_skill_vocabulary(func)
    _norm = lambda s: re.sub(r"[- ,/]", "", s.lower())
    market_skills = [s for s in skills if _norm(s) in jd_vocab]

    idf = _compute_idf(func)
    N = len(idf) or 1
    student_normed = {_norm(s) for s in skills}
    gaps = []
    for raw_skill in jd_vocab:
        normed = _norm(raw_skill)
        if normed not in student_normed:
            idf_val = idf.get(normed, 0)
            df = int(N / math.exp(idf_val)) if idf_val > 0 else 0
            if df >= 2:
                gaps.append((raw_skill, df))
    gaps.sort(key=lambda x: -x[1])

    # Jobs
    jds = retrieve_jds(func, "Entry", skills, top_k=5)

    return {
        "function": func,
        "level": "Entry",
        "match_pct": best["match_pct"],
        "skills_extracted": skills,
        "market_skills": market_skills,
        "market_gaps": [s for s, _ in gaps[:15]],
        "alternatives": best.get("alternatives", []),
        "all_probas": best.get("all_probas", {}),
        "openings": [
            {"title": jd.get("title", ""), "company": jd.get("company", ""), "url": jd.get("url", "")}
            for jd in jds[:5]
        ],
        "_timing_ms": round((time.time() - t0) * 1000),
    }


@mcp.tool(
    name="search_jobs",
    description="Search for real job openings matching a function. "
    "Returns title, company, and apply link.",
)
def search_jobs(function: str, level: str = "Entry", max_results: int = 10) -> list[dict]:
    from recommender.retrieve.retriever import retrieve_jds
    jds = retrieve_jds(function, level, top_k=max_results)
    return [
        {"title": jd.get("title", ""), "company": jd.get("company", ""), "url": jd.get("url", "")}
        for jd in jds
    ]


@mcp.tool(
    name="rebuild_corpus",
    description="Download and build the local JD corpus. Takes ~10 min. "
    "Ask the user for permission before calling this tool.",
)
def rebuild_corpus() -> dict:
    from recommender.corpus.download import download_and_build_corpus
    import recommender.retrieve.retriever as ret_mod
    ret_mod._cached_df.clear()
    ret_mod._cached_idf.clear()
    result = download_and_build_corpus()
    return {"status": result["status"], "total_rows": result["total_rows"]}


if __name__ == "__main__":
    mcp.run()
