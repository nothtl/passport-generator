"""Hybrid BM25 + Dense retrieval for job matching.

Phase 3 of the SOTA-inspired improvement plan.

BM25 provides lexical recall (catches exact term matches with proper TF
saturation). Dense embeddings provide semantic precision (catches
"PyTorch" → "deep learning").

Score via Reciprocal Rank Fusion (RRF, k=60) — research standard, no tuning needed.
RRF = Σ 1/(rank_i + k) for each retrieval method i. Robust to score scale differences.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter


# ── BM25 Implementation (pure Python, no dependencies) ────────────

class BM25:
    """BM25 Okapi scoring for job retrieval.

    f(qi, D) = term frequency of qi in document D
    IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    score = Σ IDF(qi) × f(qi,D) × (k1+1) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)

        # Tokenize all documents
        self.tokenized = [self._tokenize(d) for d in documents]

        # Document lengths
        self.doc_lens = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_lens) / max(1, self.N)

        # Document frequency per term
        self.df: dict[str, int] = {}
        for tokens in self.tokenized:
            seen = set()
            for token in tokens:
                if token not in seen:
                    self.df[token] = self.df.get(token, 0) + 1
                    seen.add(token)

        # IDF cache
        self.idf: dict[str, float] = {}
        for term, n in self.df.items():
            self.idf[term] = math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with stemming."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Keep tokens >= 2 chars, filter stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                     'with', 'from', 'by', 'as', 'this', 'that', 'it', 'its',
                     'we', 'you', 'they', 'he', 'she', 'will', 'can', 'may',
                     'has', 'have', 'had', 'do', 'does', 'did', 'not', 'no'}
        return [t for t in tokens if len(t) >= 2 and t not in stopwords]

    def score(self, query: str) -> list[float]:
        """Score all documents against a query. Returns list of scores."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [0.0] * self.N

        scores = [0.0] * self.N
        for qt in query_tokens:
            qt_idf = self.idf.get(qt, 0.0)
            if qt_idf == 0:
                continue
            for i, doc_tokens in enumerate(self.tokenized):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avgdl))
                scores[i] += qt_idf * numerator / max(1, denominator)

        return scores

    def get_top_n(self, query: str, n: int = 50) -> list[tuple[int, float]]:
        """Get top-N document indices with scores."""
        scores = self.score(query)
        indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed.sort(key=lambda x: -x[1])
        return indexed[:n]


# ── Hybrid Retriever ──────────────────────────────────────────────

def _load_function_docs(function: str, top_k: int = 500) -> tuple[list[dict], list[str]]:
    """Load all job documents for a function from the full 21GB parquet.

    Returns (job_dicts, document_texts) where document_texts[i] = title + description.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    from recommender.retrieve.retriever import _SUBSET_FUNCTION_MAP, _SECONDARY_FUNCTION_LABELS

    func_lower = function.lower()
    primary_label = _SUBSET_FUNCTION_MAP.get(func_lower, func_lower)
    secondary_labels = _SECONDARY_FUNCTION_LABELS.get(func_lower, [])
    all_labels = {primary_label} | set(secondary_labels)

    full_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "corpus", "_open_jobs_full.parquet"
    )

    pf = pq.ParquetFile(full_path)
    jobs: list[dict] = []
    docs: list[str] = []

    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=[
            "id", "company", "title", "url", "jd_markdown",
            "level", "function", "skills", "posted_at", "salary_min_k",
        ])

        # Function filter
        func_mask = pc.equal(pc.fill_null(table["function"], ""), primary_label)
        for lbl in secondary_labels:
            func_mask = pc.or_(func_mask, pc.equal(pc.fill_null(table["function"], ""), lbl))
        table = table.filter(func_mask)

        # Entry-level filter
        level_arr = pc.fill_null(table["level"], "")
        level_mask = pc.equal(level_arr, "")
        for lvl in ("intern", "entry", "junior", "unknown"):
            level_mask = pc.or_(level_mask, pc.equal(pc.utf8_lower(level_arr), lvl))
        table = table.filter(level_mask)

        if len(table) == 0:
            continue

        py = table.to_pydict()
        for j in range(len(table)):
            # Skip unpaid
            sal = py.get("salary_min_k", [None])[j]
            if sal is not None and sal == 0:
                continue

            title = str(py["title"][j] or "")
            jd = str(py["jd_markdown"][j] or "")[:2000]
            doc_text = f"{title}\n{jd}"

            jobs.append({
                "id": py["id"][j],
                "company": py["company"][j],
                "title": title,
                "url": py["url"][j],
                "jd_markdown": jd,
                "level": py["level"][j],
                "function": py["function"][j],
                "skills": py["skills"][j],
            })
            docs.append(doc_text)

        if len(jobs) >= top_k:
            break

    return jobs, docs


def hybrid_retrieve(
    function: str,
    resume_text: str,
    student_skills: list[str] | None = None,
    top_k: int = 30,
) -> list[dict]:
    """Two-stage hybrid retrieval: BM25 → dense rerank.

    1. BM25 over function-filtered jobs → top 50 candidates
    2. Dense embedding cosine similarity → rerank top 50
    3. Final score = 0.6 × BM25_norm + 0.4 × cosine_sim
    """
    # Stage 1: BM25 lexical retrieval
    jobs, docs = _load_function_docs(function, top_k=300)
    if not jobs:
        return []

    bm25 = BM25(docs)
    query = f"{resume_text[:2000]} {' '.join(student_skills or [])}"
    top_bm25 = bm25.get_top_n(query, n=min(50, len(jobs)))

    if not top_bm25:
        return []

    # Stage 2: Dense reranking
    from recommender.rank.job_ranker import _batch_embedding_similarity
    rerank_docs = [docs[idx] for idx, _ in top_bm25]
    cosine_scores = _batch_embedding_similarity(rerank_docs, resume_text)

    # Reciprocal Rank Fusion (RRF) — industry standard, scale-invariant.
    # RRF = Σ 1/(rank_i + k) for each method. k=60 is the standard value.
    # Ranks are 1-indexed within each method's result list.
    K = 60
    # BM25 ranks (1-indexed, by position in top_bm25 list)
    bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(top_bm25)}
    # Cosine ranks (1-indexed, by sorting cosine scores descending)
    cos_pairs = [(top_bm25[i][0], cosine_scores[i]) for i in range(len(top_bm25))]
    cos_pairs.sort(key=lambda x: -x[1])
    cos_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(cos_pairs)}

    # Compute RRF for each candidate
    all_indices = set(bm25_ranks.keys()) | set(cos_ranks.keys())
    fused = []
    for idx in all_indices:
        bm25_r = bm25_ranks.get(idx, len(top_bm25) + 1)
        cos_r = cos_ranks.get(idx, len(top_bm25) + 1)
        rrf = 1.0 / (bm25_r + K) + 1.0 / (cos_r + K)
        fused.append((idx, rrf, bm25_r, cos_r))

    fused.sort(key=lambda x: -x[1])

    # Return top-k jobs. Max 1 per company (not just per title) to prevent
    # "amazing-athletes-2" dominating with Magic/Chess/Instructor variants.
    result = []
    seen_titles = set()
    seen_companies = set()
    for idx, score, bm25, cosine in fused:
        job = jobs[idx]
        company = str(job.get("company", "")).lower()[:30]
        title_key = " ".join(str(job.get("title", "")).lower().split()[:4])
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        # Max 1 per company
        if company and company in seen_companies:
            continue
        if company:
            seen_companies.add(company)
        job["_hybrid_score"] = score
        job["_bm25_rank"] = bm25
        job["_cosine_rank"] = cosine
        result.append(job)
        if len(result) >= top_k:
            break

    return result
