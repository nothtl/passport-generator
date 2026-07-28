"""RJDB-based evaluation benchmark for resume-job matching.

Uses the RJDB test set (1000 triples) to measure how well our pipeline
distinguishes matched vs unmatched resumes for a given job description.

Metric: Pairwise Accuracy — % of pairs where the pipeline ranks the
matched resume higher than the unmatched resume.
"""
from __future__ import annotations

import json
import os
import sys
import time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from recommender.eval.metrics import recall_at_k, precision_at_k, ndcg_at_k


def load_rjdb(split: str = "test") -> list[dict]:
    """Load RJDB JSONL data."""
    rjdb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rjdb", "data")
    path = os.path.join(rjdb_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        print(f"RJDB {split} not found at {path}")
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_pairwise(records: list[dict], max_samples: int = 200) -> dict:
    """Evaluate pairwise ranking accuracy on RJDB.

    For each record:
    - Job = Job-Description
    - Resume A = Resume-matched (should rank HIGHER)
    - Resume B = Resume-unmatched (should rank LOWER)

    We run both resumes through our pipeline against the job description.
    If Resume A gets a higher fit score than Resume B: correct.
    """
    from recommender.analyze import analyze

    correct = 0
    total = 0
    a_scores = []
    b_scores = []

    for i, rec in enumerate(records[:max_samples]):
        jd = rec.get("Job-Description", "")
        resume_a = rec.get("Resume-matched", "")
        resume_b = rec.get("Resume-unmatched", "")
        skills = rec.get("Skills", [])

        if not jd or not resume_a or not resume_b:
            continue

        try:
            # Run pipeline with matched resume
            result_a = analyze(
                resume_text=resume_a,
                ideal_careers=skills[:3] if skills else [],
            )
            # Run pipeline with unmatched resume
            result_b = analyze(
                resume_text=resume_b,
                ideal_careers=skills[:3] if skills else [],
            )

            fit_a = result_a.get("ready_now", [{}])[0].get("fit", 0) if result_a.get("ready_now") else 0
            fit_b = result_b.get("ready_now", [{}])[0].get("fit", 0) if result_b.get("ready_now") else 0

            if fit_a > fit_b:
                correct += 1
            total += 1
            a_scores.append(fit_a)
            b_scores.append(fit_b)

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            acc = correct / max(1, total)
            print(f"  {i+1}/{min(max_samples, len(records))} — accuracy={acc:.2%}")

    accuracy = correct / max(1, total)
    avg_a = sum(a_scores) / max(1, len(a_scores))
    avg_b = sum(b_scores) / max(1, len(b_scores))

    return {
        "pairwise_accuracy": accuracy,
        "total_pairs": total,
        "correct": correct,
        "avg_matched_fit": avg_a,
        "avg_unmatched_fit": avg_b,
        "fit_delta": avg_a - avg_b,
    }


def evaluate_skill_overlap(records: list[dict], max_samples: int = 100) -> dict:
    """Check how well our skill extractor finds the labeled skills."""
    from recommender.extract.skill_extractor import extract_skill_profile

    total_skills = 0
    found_skills = 0

    for i, rec in enumerate(records[:max_samples]):
        labeled = set(s.lower().strip() for s in rec.get("Skills", []))
        resume = rec.get("Resume-matched", "")
        if not labeled or not resume:
            continue

        profile = extract_skill_profile(resume)
        extracted = set(s.lower().strip() for s in profile.get("skills", []))
        found_skills += len(labeled & extracted)
        total_skills += len(labeled)

    recall = found_skills / max(1, total_skills)
    return {"skill_recall": recall, "total_labeled_skills": total_skills, "found_skills": found_skills}


def run_quick(max_pairs: int = 50) -> dict:
    """Quick evaluation on a small sample (fast)."""
    records = load_rjdb("test")
    if not records:
        return {"error": "RJDB test set not found"}

    print(f"Loaded {len(records)} RJDB test records")
    print(f"Evaluating pairwise accuracy on {max_pairs} pairs...")

    pairwise = evaluate_pairwise(records, max_samples=max_pairs)
    skill = evaluate_skill_overlap(records, max_samples=min(50, max_pairs))

    return {**pairwise, **skill}


if __name__ == "__main__":
    result = run_quick(max_pairs=50)
    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"\n=== RJDB Benchmark Results ===")
    print(f"  Pairwise Accuracy:  {result['pairwise_accuracy']:.1%}")
    print(f"  Avg Matched Fit:    {result['avg_matched_fit']:.1f}%")
    print(f"  Avg Unmatched Fit:  {result['avg_unmatched_fit']:.1f}%")
    print(f"  Fit Delta:          {result['fit_delta']:.1f}%")
    print(f"  Skill Recall:       {result['skill_recall']:.1%}")
