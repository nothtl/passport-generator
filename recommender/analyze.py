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

    # Filter noise: keep multi-word skills + single words above median IDF
    # Threshold comes from the data (median IDF of this function's skills), not hardcoded
    from recommender.retrieve.retriever import _compute_idf
    try:
        idf = _compute_idf(func)
        import re as _re2
        _norm2 = lambda s: _re2.sub(r'[- ,/]', '', s.lower())
        multi = [s for s in skills if ' ' in s]
        single = [(s, idf.get(_norm2(s), 0)) for s in skills if ' ' not in s]
        if single:
            single.sort(key=lambda x: -x[1])
            median_idf = sorted([v for _, v in single])[len(single)//2]
            threshold = max(0.5, median_idf)  # at least 0.5
            top_single = [s for s, v in single if v >= threshold]
            skills = multi + top_single
    except Exception:
        pass

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
    result = llm_enhance(result, resume_text)

    # Rule-based coach fallback if LLM unavailable
    if not result.get("coach_notes") and result.get("missing_skills"):
        top_gaps = result["missing_skills"][:5]
        func = result["function"]
        result["coach_notes"] = (
            f"For a {func} career, focus on developing: {', '.join(top_gaps)}. "
            f"These skills appear most frequently in real job postings for this field."
        )

    # Compute skill weights (TF-IDF) for display
    import re as _re3, math as _math
    from collections import Counter as _Counter
    _norm_wt = lambda s: _re3.sub(r'[- ,/]', '', s.lower())
    _tf_counts = _Counter(_norm_wt(s) for s in skills)
    skill_weights = {}
    try:
        idf = _compute_idf(func)
        for s in skills:
            n = _norm_wt(s)
            tf = 1 + _math.log(_tf_counts.get(n, 1))
            skill_weights[s] = round(tf * idf.get(n, 1.0), 1)
    except Exception:
        pass

    # Simplify for output: keep only human-relevant fields
    def _clean_company(name):
        """Filter out UUID-looking company names."""
        if not name: return ""
        if _re3.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', name):
            return ""
        return name

    jobs = []
    for j in result.get("ready_jobs", []) + result.get("target_jobs", []):
        jobs.append({
            "title": j.get("title", ""),
            "company": _clean_company(j.get("company", "")),
            "url": j.get("url", ""),
            "fit": j.get("fit", 0),
            "label": j.get("label", ""),
            "why": j.get("why_fits", j.get("why", "")),
            "gaps": j.get("remaining_gaps", ""),
        })
    if not jobs:
        for jd in result.get("openings", []):
            jobs.append({
                "title": jd.get("title", ""),
                "company": _clean_company(jd.get("company", "")),
                "url": jd.get("url", ""),
                "fit": 0, "label": "", "why": "", "gaps": "",
            })

    return {
        "resume": resume_text.strip(),
        "function": result["function"],
        "confidence": result["match_pct"],
        "skills": result.get("skills_extracted", []),
        "skill_weights": {s: w for s, w in sorted(skill_weights.items(), key=lambda x: -x[1])[:15]},
        "market_relevant": result.get("market_skills", []),
        "inferred": result.get("inferred_skills", []),
        "gaps": result.get("missing_skills", []),
        "related": [{"skill": s, "pmi": round(sc, 2)} for s, sc in result.get("related_skills", [])],
        "alternatives": [{"function": a["function"], "pct": a["match_pct"]} for a in result.get("alternatives", [])[:5]],
        "coach_notes": result.get("gap_explanation", ""),
        "jobs": jobs,
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
    print(f"  Function:   {result['function']}")
    print(f"  Confidence: {result.get('confidence', result.get('match_pct', 0))}%")
    print()

    print("=" * 60)
    print("SKILLS FOUND")
    print("=" * 60)
    all_skills = result.get('skills', [])
    market = result.get('market_relevant', [])
    weights = result.get('skill_weights', {})
    print(f"  Total: {len(all_skills)} ({len(market)} appear in real job postings)")
    if weights:
        print(f"  Top by TF-IDF weight:")
        top_weighted = sorted(weights.items(), key=lambda x: -x[1])[:10]
        for s, w in top_weighted:
            bar = '#' * int(w/2) + '.' * (10 - int(w/2))
            print(f"    [{bar}] {s:<30s} (×{w:.1f})")
    else:
        cols = 3
        for i in range(0, min(len(all_skills), 24), cols):
            row = all_skills[i:i+cols]
            print(f"    " + "  ".join(f"[Y] {s:<30s}" for s in row))
    print()

    if result.get("inferred"):
        print("=" * 60)
        print("INFERRED SKILLS (AI)")
        print("=" * 60)
        print("  " + ", ".join(result["inferred"]))
        print()

    if result.get("related"):
        print("=" * 60)
        print("RELATED SKILLS (co-occur in job postings)")
        print("=" * 60)
        items = ", ".join("{} ({:.1f})".format(s["skill"], s["pmi"]) for s in result["related"])
        print("  " + items)
        print()

    if result.get("gaps"):
        print("=" * 60)
        print("MARKET GAPS")
        print("=" * 60)
        print(f"  {', '.join(result['gaps'][:15])}")
        print()

    if result.get("coach_notes"):
        print("=" * 60)
        print("COACH NOTES")
        print("=" * 60)
        print("  " + result["coach_notes"])
        print()

    if result.get("alternatives"):
        print("=" * 60)
        print("ALSO CONSIDER")
        print("=" * 60)
        for alt in result["alternatives"][:3]:
            print(f"  {alt['function']:25s} {alt['pct']}%")
        print()

    if result.get("jobs"):
        print("=" * 60)
        print("TOP JOB OPENINGS")
        print("=" * 60)
        for i, j in enumerate(result["jobs"], 1):
            title = j.get("title", "")[:60]
            company = j.get("company", "")
            fit = j.get("fit", 0)
            label = j.get("label", "")
            label_str = f" [{fit}% {label}]" if fit > 0 and label else ""
            print(f"  {i}. {title}{label_str}")
            if company:
                print(f"     @ {company}")
            if j.get("why"):
                print(f"     Why:  {j['why'][:120]}")
            if j.get("gaps"):
                print(f"     Gaps: {j['gaps'][:120]}")
            if j.get("url"):
                print(f"     {j['url'][:80]}")


if __name__ == "__main__":
    main()
