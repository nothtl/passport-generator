"""
SpeakHire Recommender MCP Server.

Tools:
    extract_skills     — Extract named skills from resume/LinkedIn text
    match_role         — Find best-fit role from skills
    analyze_resume     — All-in-one: resume in, full analysis out with tiered jobs
    search_jobs        — Return open job postings for a function+level
    submit_feedback    — Submit student like/dislike on recommended jobs
    rebuild_corpus     — Download and build the local JD corpus

Run: python -m recommender.mcp_server
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from mcp.server.fastmcp import FastMCP

from recommender.corpus.download import corpus_exists
from recommender.extract.skill_extractor import extract_skills_from_text
from recommender.llm import LLMConfig
from recommender.match.ensemble_matcher import match_role as _match_role
from recommender.analyze import analyze as analyze_pipeline, _default_llm_mode

mcp = FastMCP("speakhire-recommender")

_DATA_HINT = {
    "error": "data_not_ready",
    "hint": "No job data available. Ask the user for permission to run rebuild_corpus.",
}

_FEEDBACK_PATH = Path(_PROJECT_DIR) / "recommender" / "data" / "feedback.json"


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
    """Classify skills into best-fit career function using the ML classifier."""
    combined = ", ".join(skills) if skills else ""
    if not combined.strip():
        return {"error": "No skills provided"}
    from recommender.match.ensemble_matcher import match_role as _match
    return _match(combined)


@mcp.tool(
    name="analyze_resume",
    description="Run the full resume analysis pipeline. Returns tiered job recommendations: "
    "ready_now (2 jobs matching current skills), aspirational (3 jobs aligned with goals). "
    "Provide ideal_careers (3-5 dream job titles) for best results.",
)
def analyze_resume(
    resume_text: str,
    linkedin_text: str = "",
    headline_text: str = "",
    career_goals_text: str = "",
    smart_goals_text: str = "",
    hope_to_gain_text: str = "",
    ideal_career_text: str = "",
    study_text: str = "",
    ideal_careers: list[str] | None = None,
    lane_mode: str = "",
    disable_embeddings: bool = False,
    disable_llm: bool = False,
    subdomain_debug: bool = False,
    llm_mode: str = "",
    llm_provider: str = "",
    llm_model_route: str = "",
    llm_model_evidence: str = "",
    llm_model_jobs: str = "",
    llm_max_calls: int = 0,
    llm_debug: bool = False,
) -> dict:
    combined = f"{resume_text or ''}\n{linkedin_text or ''}"
    if not combined.strip():
        return {"error": "No resume text provided"}

    llm_config = None
    if any([llm_mode, llm_provider, llm_model_route, llm_model_evidence, llm_model_jobs, llm_max_calls, llm_debug]):
        defaults = LLMConfig()
        llm_config = LLMConfig(
            mode=("off" if disable_llm else (llm_mode or _default_llm_mode())),
            provider_name=llm_provider or defaults.provider_name,
            route_model=llm_model_route or defaults.route_model,
            evidence_model=llm_model_evidence or defaults.evidence_model,
            jobs_model=llm_model_jobs or defaults.jobs_model,
            max_calls=llm_max_calls or defaults.max_calls,
            debug=llm_debug,
        )
    elif disable_llm:
        llm_config = LLMConfig(mode="off")

    previous_disable_embeddings = os.getenv("RECOMMENDER_DISABLE_EMBEDDINGS", "")
    if disable_embeddings:
        os.environ["RECOMMENDER_DISABLE_EMBEDDINGS"] = "1"

    t0 = time.time()
    result = analyze_pipeline(
        resume_text=resume_text,
        linkedin_text=linkedin_text,
        headline_text=headline_text,
        career_goals_text=career_goals_text,
        smart_goals_text=smart_goals_text,
        hope_to_gain_text=hope_to_gain_text,
        ideal_career_text=ideal_career_text,
        study_text=study_text,
        ideal_careers=ideal_careers,
        top_k=5,
        lane_mode=lane_mode or "hybrid",
        llm_config=llm_config,
    )
    if disable_embeddings:
        if previous_disable_embeddings:
            os.environ["RECOMMENDER_DISABLE_EMBEDDINGS"] = previous_disable_embeddings
        else:
            os.environ.pop("RECOMMENDER_DISABLE_EMBEDDINGS", None)
    result["_timing_ms"] = round((time.time() - t0) * 1000)
    if subdomain_debug:
        result["subdomain_debug"] = {
            "subdomain": result.get("subdomain", ""),
            "subdomain_confidence": result.get("subdomain_confidence", 0.0),
            "subdomain_alternatives": result.get("subdomain_alternatives", []),
            "candidate_functions": result.get("candidate_functions", []),
            "job_cluster": result.get("job_cluster", []),
        }
    return result


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


def _load_feedback() -> dict:
    """Load feedback store, initializing if needed."""
    if _FEEDBACK_PATH.exists():
        try:
            with open(_FEEDBACK_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"likes": {}, "dislikes": {}, "preferences": {}}


def _save_feedback(data: dict) -> None:
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@mcp.tool(
    name="submit_feedback",
    description="Submit student feedback on recommended jobs. Student likes or dislikes a job, "
    "and the recommender adjusts future preferences. Provide student_id (e.g., email), "
    "job title, and whether they liked it.",
)
def submit_feedback(
    student_id: str,
    job_title: str,
    liked: bool = True,
    ideal_careers: list[str] | None = None,
) -> dict:
    data = _load_feedback()
    key = student_id.strip().lower()
    if not key:
        return {"error": "student_id required"}

    job_key = job_title.strip()

    # Record like/dislike
    if liked:
        data["likes"].setdefault(key, []).append(job_key)
        # Remove from dislikes if present
        if key in data["dislikes"] and job_key in data["dislikes"][key]:
            data["dislikes"][key].remove(job_key)
    else:
        data["dislikes"].setdefault(key, []).append(job_key)
        if key in data["likes"] and job_key in data["likes"][key]:
            data["likes"][key].remove(job_key)

    # Update preferences from liked jobs
    if liked and ideal_careers:
        prefs = data["preferences"].setdefault(key, {})
        for career in ideal_careers:
            prefs[career.strip().lower()] = prefs.get(career.strip().lower(), 0) + 1

    # Build preference profile
    prefs = data["preferences"].get(key, {})
    preferred_functions = list(prefs.keys()) if prefs else []

    _save_feedback(data)

    return {
        "status": "ok",
        "student_id": key,
        "total_likes": len(data["likes"].get(key, [])),
        "total_dislikes": len(data["dislikes"].get(key, [])),
        "preferred_careers": preferred_functions[:5],
    }


if __name__ == "__main__":
    mcp.run()
