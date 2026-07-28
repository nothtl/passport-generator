"""Baseline evaluation of the v2 pipeline against all 19 students.

Uses a simple relevance heuristic: the student's top ready_now job is "relevant"
and all other jobs get a graded relevance score based on their fit percentile
within the student's results.

This is a weak relevance signal (not human-labeled), but it gives us a
consistent number to track across pipeline changes.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from recommender.eval.metrics import recall_at_k, precision_at_k, ndcg_at_k, mrr


def load_all_students(report_dir: str) -> list[dict]:
    """Load all student JSON reports from a directory."""
    students = []
    for fname in sorted(os.listdir(report_dir)):
        if not fname.endswith('.json') or fname == 'index.json':
            continue
        path = os.path.join(report_dir, fname)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if data.get('error'):
            continue
        data['_name'] = fname.replace('.json', '')
        students.append(data)
    return students


def build_relevance(ranked_jobs: list[dict], ready_titles: set[str]) -> dict[str, float]:
    """Build a relevance dict for nDCG computation.

    Strategy:
      - The student's actual top ready_now job gets relevance 3.0 (best)
      - Other ready_now jobs get relevance 2.0
      - Aspirational jobs get relevance 1.0
      - All other jobs get relevance 0.0

    This is NOT ground truth — it's the pipeline's own ranking used as
    a consistency check. Future phases will use human-labeled relevance.
    """
    relevance: dict[str, float] = {}
    for job in ranked_jobs:
        jid = _job_id(job)
        title = job.get('title', '')
        if title in ready_titles:
            relevance[jid] = 2.0
        else:
            # Graded by fit percentile
            fit = job.get('fit', 0)
            relevance[jid] = fit / 100.0  # 0-1 scale
    return relevance


def _job_id(job: dict) -> str:
    return (
        job.get("id", "")
        or job.get("url", "")
        or f"{job.get('title', '')}::{job.get('company', '')}"
    )


def evaluate_student(student: dict) -> dict[str, float]:
    """Compute all metrics for one student."""
    ready_now = student.get('ready_now', [])
    aspirational = student.get('aspirational', [])
    all_jobs = ready_now + aspirational

    if not all_jobs:
        return {'recall@5': 0, 'precision@5': 0, 'ndcg@5': 0, 'mrr': 0}

    # Relevance: ready_now[0] is the "best" match
    ready_titles = {j.get('title', '') for j in ready_now}
    relevance = build_relevance(all_jobs, ready_titles)

    # The pipeline's own top pick is "relevant"
    relevant_ids = {_job_id(j) for j in ready_now[:3]}

    return {
        'recall@5': recall_at_k(all_jobs, relevant_ids, k=5),
        'precision@5': precision_at_k(all_jobs, relevant_ids, k=5),
        'ndcg@5': ndcg_at_k(all_jobs, relevance, k=5),
        'mrr': mrr(all_jobs, relevant_ids),
        'ready_count': len(ready_now),
        'aspirational_count': len(aspirational),
        'top_fit': ready_now[0].get('fit', 0) if ready_now else 0,
    }


def run(report_dir: str | None = None) -> dict:
    """Run baseline evaluation on all students."""
    if report_dir is None:
        report_dir = os.path.join(_PROJECT_DIR, 'reports', 'tingli_v2_final')

    students = load_all_students(report_dir)
    if not students:
        return {'error': f'No students found in {report_dir}'}

    all_metrics = {}
    for s in students:
        name = s.get('_name', 'unknown')
        all_metrics[name] = evaluate_student(s)

    # Aggregate
    n = len(all_metrics)
    agg = {
        'students': n,
        'avg_recall@5': sum(m['recall@5'] for m in all_metrics.values()) / n,
        'avg_precision@5': sum(m['precision@5'] for m in all_metrics.values()) / n,
        'avg_ndcg@5': sum(m['ndcg@5'] for m in all_metrics.values()) / n,
        'avg_mrr': sum(m['mrr'] for m in all_metrics.values()) / n,
        'avg_top_fit': sum(m['top_fit'] for m in all_metrics.values()) / n,
        'per_student': all_metrics,
    }
    return agg


if __name__ == '__main__':
    result = run()
    if 'error' in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"=== Pipeline v2 Baseline ({result['students']} students) ===")
    print(f"  Avg Recall@5:    {result['avg_recall@5']:.3f}")
    print(f"  Avg Precision@5: {result['avg_precision@5']:.3f}")
    print(f"  Avg nDCG@5:      {result['avg_ndcg@5']:.3f}")
    print(f"  Avg MRR:         {result['avg_mrr']:.3f}")
    print(f"  Avg Top Fit:     {result['avg_top_fit']:.1f}%")
    print()
    print("Per-student:")
    for name, m in result['per_student'].items():
        print(f"  {name:<32} R@5={m['recall@5']:.2f} P@5={m['precision@5']:.2f} nDCG={m['ndcg@5']:.2f} MRR={m['mrr']:.2f} top={m['top_fit']:.0f}%")
