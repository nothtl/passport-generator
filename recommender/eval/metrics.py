"""Standard IR evaluation metrics for job recommendation."""
from __future__ import annotations

import math


def recall_at_k(ranked_jobs: list[dict], relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of relevant jobs found in top-k results."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for j in ranked_jobs[:k] if _job_id(j) in relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(ranked_jobs: list[dict], relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    found = sum(1 for j in ranked_jobs[:k] if _job_id(j) in relevant_ids)
    return found / min(k, len(ranked_jobs))


def ndcg_at_k(ranked_jobs: list[dict], relevance: dict[str, float], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses graded relevance: higher score = more relevant.
    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = ideal DCG (sorted by relevance descending)
    nDCG = DCG / IDCG
    """
    if not relevance or k <= 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, job in enumerate(ranked_jobs[:k]):
        jid = _job_id(job)
        rel = relevance.get(jid, 0.0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0, we want log2(2)=1 at position 0

    # IDCG — ideal ordering by relevance descending
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(ranked_jobs: list[dict], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank — 1 / rank_of_first_relevant_job."""
    for i, job in enumerate(ranked_jobs):
        if _job_id(job) in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranked_jobs: list[dict], relevant_ids: set[str]) -> float:
    """Average Precision — mean of precision@k for each relevant result."""
    if not relevant_ids:
        return 0.0
    hits = 0
    total_precision = 0.0
    for i, job in enumerate(ranked_jobs):
        if _job_id(job) in relevant_ids:
            hits += 1
            total_precision += hits / (i + 1)
    return total_precision / len(relevant_ids)


def _job_id(job: dict) -> str:
    """Stable ID for a job across pipeline runs."""
    return (
        job.get("id", "")
        or job.get("url", "")
        or f"{job.get('title', '')}::{job.get('company', '')}"
    )
