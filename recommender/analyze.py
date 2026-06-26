"""Analyze a resume — extract skills, match role, show skill gaps.

Usage:
    python recommender/analyze.py resume.txt
    python recommender/analyze.py "Pasted resume text here..."
    python recommender/analyze.py                      # reads from stdin

Output: skills, best-fit role, market skill gaps, and top job openings.
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on path
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from recommender.extract.skill_extractor import extract_skills_from_text
from recommender.match.role_matcher import match_role
from recommender.retrieve.retriever import retrieve_jds
from recommender.profile.aggregator import aggregate_skills
from recommender.profile.gap_analyzer import analyze_gaps


def analyze(resume_text: str, top_k: int = 10):
    """Run the full pipeline and return a dict."""
    # Stage 1+2: extract skills
    skills = extract_skills_from_text(resume_text, use_semantic=True)

    # Match best role
    best = match_role(skills)
    if not best:
        return {"error": "No matching role found", "skills_found": skills}

    # Retrieve real JDs
    jds = retrieve_jds(best["function"], best.get("level", "Entry"), skills, top_k=top_k)

    # Aggregate market skills + gaps
    all_skills = aggregate_skills(jds, best["matched_skills"], best.get("missing_skills", []))
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

    # Add job openings + alternatives
    result["openings"] = [
        {"title": jd.get("title", ""), "company": jd.get("company", ""), "url": jd.get("url", "")}
        for jd in jds[:5]
    ]
    result["skills_extracted"] = skills
    result["alternatives"] = best.get("alternatives", [])
    return result


def main():
    # Get resume text
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
        print(f"Skills found: {result['skills_found']}")
        print(f"Error: {result['error']}")
        return

    # ── Print results ──
    print("=" * 60)
    print("SKILLS EXTRACTED")
    print("=" * 60)
    for s in result["skills_extracted"]:
        print(f"  [Y] {s}")
    print()

    print("=" * 60)
    print("BEST MATCH")
    print("=" * 60)
    print(f"  Role:      {result['role']} ({result['level']})")
    print(f"  Match:     {result['match_pct']}%")
    print(f"  Has:       {', '.join(result['matched_skills'])}")
    print(f"  Missing:   {', '.join(result['missing_skills'])}")
    print()

    print("=" * 60)
    print(f"TOP SKILL GAPS (market demands these, you don't have them)")
    print("=" * 60)
    for g in result["top_gaps"]:
        print(f"  [ ] {g['skill']:30s} (in {g['frequency']} JDs)")
    print()

    print("=" * 60)
    print("MARKET SKILLS RANKED")
    print("=" * 60)
    for s in result["ranked_skills"][:15]:
        mark = "[Y]" if s["has"] else "[ ]"
        print(f"  {mark} {s['skill']:30s} freq={s['frequency']}")
    print()

    if result["openings"]:
        print("=" * 60)
        print("TOP JOB OPENINGS")
        print("=" * 60)
        for i, jd in enumerate(result["openings"], 1):
            print(f"  {i}. {jd['title'][:70]}")
            if jd["company"]:
                print(f"     @ {jd['company']}")


if __name__ == "__main__":
    main()
