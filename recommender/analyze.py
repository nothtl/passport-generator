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

from recommender.extract.skill_extractor import extract_skills_from_text as _extract_skills
from recommender.match.ensemble_matcher import match_role as _match_role
from recommender.retrieve.retriever import retrieve_jds, get_related_skills
from recommender.llm import enhance as llm_enhance, is_enabled as llm_enabled


def analyze(resume_text: str, top_k: int = 10):
    """Run the full pipeline and return a dict."""
    # Step 1: Classify into function
    best = _match_role(resume_text)
    if not best:
        return {"error": "Could not classify resume", "skills_found": []}

    func = best["function"]

    # Step 2: Extract skills using smart extractor (ESCO vocab + n-grams)
    skills = _extract_skills(resume_text)

    # Load function features for skill filtering
    import json
    cls_path = os.path.join(os.path.dirname(__file__), 'data', 'classifier_skills.json')
    with open(cls_path) as f:
        func_features = json.load(f)

    # Step 3: Get has/missing from classifier features
    from recommender.extract.skill_extractor import extract_skills_for_function
    has_skills, missing_skills = extract_skills_for_function(resume_text, func)

    # Step 4: Find related skills via PMI
    related_skills = get_related_skills(func, skills)

    # Step 5: Retrieve real job openings
    jds = retrieve_jds(func, "Entry", skills, top_k=top_k)

    # Step 5: Filter skills to JD vocabulary for meaningful gap display
    from recommender.retrieve.retriever import get_jd_skill_vocabulary, _compute_idf
    import re as _re
    jd_vocab = get_jd_skill_vocabulary(func)
    _norm = lambda s: _re.sub(r'[- ,/]', '', s.lower())
    market_skills = [s for s in skills if _norm(s) in jd_vocab]

    # Market gaps: skills that appear in MANY JDs but student doesn't have.
    # Sort by document frequency (how many JDs list this skill), not IDF.
    import math
    idf = _compute_idf(func)
    N = len(idf) or 1
    student_normed = {_norm(s) for s in skills}
    market_gaps = []
    for raw_skill in jd_vocab:
        normed = _norm(raw_skill)
        if normed not in student_normed:
            idf_val = idf.get(normed, 0)
            df = int(N / math.exp(idf_val)) if idf_val > 0 else 0
            if df >= 2:  # appears in at least 2 JDs
                market_gaps.append((raw_skill, df))
    market_gaps.sort(key=lambda x: -x[1])
    market_missing = [s for s, _ in market_gaps[:15]]

    result = {
        "function": func,
        "level": "Entry",
        "match_pct": best["match_pct"],
        "skills_extracted": skills,
        "related_skills": [(s, round(sc, 2)) for s, sc in related_skills[:8]],
        "market_skills": market_skills,
        "has_skills": has_skills,
        "missing_skills": market_missing,
        "alternatives": best.get("alternatives", []),
        "all_probas": best.get("all_probas", {}),
        "openings": [
            {"title": jd.get("title", ""), "company": jd.get("company", ""), "url": jd.get("url", "")}
            for jd in jds[:5]
        ],
    }
    return llm_enhance(result, resume_text)


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
    print("SKILLS FOUND")
    print("=" * 60)
    all_skills = result.get('skills_extracted', [])
    print(f"  All ({len(all_skills)}):")
    if all_skills:
        cols = 3
        for i in range(0, min(len(all_skills), 24), cols):
            row = all_skills[i:i+cols]
            print(f"    " + "  ".join(f"[Y] {s:<30s}" for s in row))
    print()
    related = result.get('related_skills', [])
    if related:
        print("=" * 60)
        print("RELATED SKILLS (co-occur with yours in real job postings)")
        print("=" * 60)
        items = ", ".join("{} ({:.1f})".format(s, sc) for s, sc in related)
        print("  " + items)
        print()
    market = result.get('market_skills', [])
    print(f"  Market-relevant ({len(market)}/{len(all_skills)} appear in job postings)")
    print()
    missing = result.get('missing_skills', [])
    if missing:
        print("=" * 60)
        print("MARKET GAPS (skills real jobs demand that you don't show)")
        print("=" * 60)
        print(f"  {', '.join(missing[:15])}")
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

    if result.get("gap_explanation"):
        print("=" * 60)
        print("COACH NOTES (AI)")
        print("=" * 60)
        print("  " + result["gap_explanation"])
        print()

    if result.get("ready_jobs") or result.get("target_jobs"):
        print("=" * 60)
        print("JOB FIT (AI)")
        print("=" * 60)
        if result.get("ready_jobs"):
            print("  READY (apply now):")
            for j in result["ready_jobs"][:3]:
                why = j.get("why", "")[:80]
                print("    [{}%] {} @ {}".format(j.get("fit",0), j.get("title","")[:50], j.get("company","")[:25]))
                if why:
                    print("          " + why)
        if result.get("target_jobs"):
            print("  TARGET (growth path):")
            for j in result["target_jobs"][:3]:
                why = j.get("why", "")[:80]
                print("    [{}%] {} @ {}".format(j.get("fit",0), j.get("title","")[:50], j.get("company","")[:25]))
                if why:
                    print("          " + why)
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
