"""Analyze a resume — extract skills, match role, show skill gaps.

Usage:
    python recommender/analyze.py resume.txt
    python recommender/analyze.py "Pasted resume text here..."
    python recommender/analyze.py                      # reads from stdin

Architecture: ML classifier (2,484 real resumes) → data-driven skills → corpus jobs.
No hardcoded regex patterns, keyword maps, or function mappings.
"""
from __future__ import annotations

import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from recommender.extract.skill_extractor import extract_skills_from_text, extract_skills_for_function
from recommender.match.ensemble_matcher import match_role as _match_role
from recommender.retrieve.retriever import retrieve_jds


def analyze(resume_text: str, top_k: int = 10):
    """Run the full pipeline and return a dict."""
    # Step 1: Classify into function using ML model
    best = _match_role(resume_text)
    if not best:
        return {"error": "Could not classify resume", "skills_found": []}

    func = best["function"]

    # Step 2: Extract skills using that function's data-driven features
    skills = extract_skills_from_text(resume_text, function=func)
    has_skills, missing_skills = extract_skills_for_function(resume_text, func)

    # Step 3: Retrieve real job openings
    jds = retrieve_jds(func, "Entry", skills, top_k=top_k)

    return {
        "function": func,
        "level": "Entry",
        "match_pct": best["match_pct"],
        "skills_extracted": skills,
        "has_skills": has_skills,
        "missing_skills": missing_skills,
        "alternatives": best.get("alternatives", []),
        "all_probas": best.get("all_probas", {}),
        "openings": [
            {"title": jd.get("title", ""), "company": jd.get("company", ""), "url": jd.get("url", "")}
            for jd in jds[:5]
        ],
    }


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            with open(arg, encoding="utf-8") as f:
                resume_text = f.read()
        else:
            resume_text = arg
    else:
        print("Paste resume text (Ctrl+Z then Enter on Windows, Ctrl+D on Unix):")
        resume_text = sys.stdin.read()

    if not resume_text.strip():
        print("Error: no resume text provided.")
        sys.exit(1)

    result = analyze(resume_text)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("=" * 60)
    print("BEST MATCH")
    print("=" * 60)
    print(f"  Function:   {result['function']} ({result['level']})")
    print(f"  Confidence: {result['match_pct']}%")
    print()

    print("=" * 60)
    print("SKILLS (data-driven from ML model)")
    print("=" * 60)
    print(f"  Has ({len(result['has_skills'])}):    {', '.join(result['has_skills'][:15])}")
    print(f"  Missing ({len(result['missing_skills'])}): {', '.join(result['missing_skills'][:10])}")
    print()

    if result.get("all_probas"):
        print("=" * 60)
        print("ALL FUNCTION PROBABILITIES")
        print("=" * 60)
        for func, prob in sorted(result["all_probas"].items(), key=lambda x: -x[1]):
            if prob >= 3:
                bar = "#" * int(prob / 5) + "." * (20 - int(prob / 5))
                print(f"  {func:25s} [{bar}] {prob:5.1f}%")
        print()

    if result.get("alternatives"):
        print("=" * 60)
        print("ALSO CONSIDER")
        print("=" * 60)
        for alt in result["alternatives"][:3]:
            print(f"  {alt['function']:25s} {alt['match_pct']}%")
        print()

    if result.get("openings"):
        print("=" * 60)
        print("TOP JOB OPENINGS")
        print("=" * 60)
        for i, jd in enumerate(result["openings"], 1):
            title = str(jd.get("title", ""))[:70]
            company = str(jd.get("company", ""))[:40]
            print(f"  {i}. {title}")
            if company:
                print(f"     @ {company}")


if __name__ == "__main__":
    main()
