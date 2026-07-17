"""3-Signal Ensemble Matcher - No hardcoded patterns, no LLM.

Signal 1: ML Classifier (TF-IDF + LinearSVC, 73.9% on 2,484 resumes)
Signal 2: O*NET Task Overlap (18,796 task statements via TF-IDF)
Signal 3: Sentence Embeddings (all-MiniLM-L6-v2, semantic similarity vs O*NET tasks)

Fusion: weighted voting with confidence thresholding.

Configuration: recommender/config.yaml -> ensemble section.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter
from typing import Any

import numpy as np

from recommender.config import get_ensemble

_CFG = get_ensemble()
_STOPS: set = set(_CFG.stops)
_OCC_TO_FUNC_MAP: dict = dict(_CFG.occ_to_func)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
_MODEL_PATH = os.path.join(_DATA, "resume_classifier.pkl")
_CLASSES_PATH = os.path.join(_DATA, "resume_classifier_classes.json")
_INDEX_PATH = os.path.join(_DATA, "onet_occ_index.json")

# ── Signal 1: ML Classifier ──
_model = None
_classes = None


def _load_classifier():
    global _model, _classes
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(_CLASSES_PATH) as f:
            _classes = json.load(f)
    return _model, _classes


def _classifier_probas(text: str) -> dict[str, float]:
    model, classes = _load_classifier()
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    probas = model.predict_proba([cleaned])[0]
    return {c: float(p) for c, p in zip(classes, probas)}


# ── Signal 2: O*NET Task Overlap ──
_onet_index = None


def _load_onet():
    global _onet_index
    if _onet_index is None:
        with open(_INDEX_PATH, encoding="utf-8") as f:
            _onet_index = json.load(f)
    return _onet_index


# Occupation-to-function mapping (loaded from config)
_OCC_TO_FUNC: dict[str, str] = {}
_func_set: set[str] = set()

def _build_occ_to_func():
    """Build occupation->function mapping from config."""
    global _OCC_TO_FUNC, _func_set
    if _OCC_TO_FUNC:
        return
    _build_from_config()


def _build_from_config():
    """Build occupation-to-function mapping from config YAML."""
    global _OCC_TO_FUNC, _func_set
    index = _load_onet()
    for occ in index["occs"]:
        title = occ["title"]
        func = "unmapped"
        for pattern, func_name in _OCC_TO_FUNC_MAP.items():
            if re.search(pattern, title):
                func = func_name
                break
        _OCC_TO_FUNC[title] = func
        _func_set.add(func)
    _func_set.discard("unmapped")


def _onet_probas(text: str) -> dict[str, float]:
    """Score functions by TF-IDF overlap with O*NET task statements."""
    index = _load_onet()
    occs = index["occs"]
    num_occs = index["num_occs"]
    keyword_df = index["keyword_df"]
    keyword_occs = index["keyword_occs"]

    words = [w for w in re.findall(r"\b[a-z][a-z]+\b", text.lower()) if w not in _STOPS]
    wc = Counter(words)

    # Score per occupation
    occ_scores = Counter()
    for w, count in wc.items():
        w_key = str(w)
        if w_key in keyword_occs and keyword_df[w_key] > 2:
            idf = math.log(num_occs / keyword_df[w_key])
            tf = 1 + math.log(count)
            for occ_idx in keyword_occs[w_key]:
                occ_scores[occ_idx] += tf * idf

    # Aggregate to functions
    _build_occ_to_func()
    func_scores: dict[str, float] = {}
    for occ_idx, score in occ_scores.items():
        if occ_idx < len(occs):
            func = _OCC_TO_FUNC.get(occs[occ_idx]["title"], "unmapped")
            if func != "unmapped":
                func_scores[func] = func_scores.get(func, 0) + score

    # Normalize to probabilities
    total = sum(func_scores.values()) or 1
    return {f: s / total for f, s in func_scores.items()}


# ── Signal 3: Sentence Embeddings ──
_embedder = None
_occ_embeddings = None
_occ_funcs = None


def _load_embedder():
    global _embedder
    if _embedder is None:
        if os.getenv("RECOMMENDER_DISABLE_EMBEDDINGS", "").strip() == "1":
            raise RuntimeError("Embeddings disabled by environment")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _embedder


def _load_occ_embeddings():
    global _occ_embeddings, _occ_funcs
    if _occ_embeddings is None:
        index = _load_onet()
        _build_occ_to_func()

        # Build text per occupation: title + task statements for richer embedding
        # Load actual task text for each occupation
        import pandas as pd
        onet = os.path.join(_HERE, "..", "data", "onet")
        tasks_df = pd.read_excel(os.path.join(onet, "Task Statements.xlsx"))
        occ_task_text = {}
        for (_, title), group in tasks_df.groupby(["O*NET-SOC Code", "Title"]):
            tasks = " ".join(group["Task"].dropna().tolist()[:10])  # first 10 tasks
            occ_task_text[title] = tasks

        occ_texts = []
        occ_titles = []
        for occ in index["occs"]:
            title = occ["title"]
            tasks = occ_task_text.get(title, title)
            occ_texts.append(f"{title}: {tasks[:800]}")  # title + tasks
            occ_titles.append(title)

        model = _load_embedder()
        _occ_embeddings = model.encode(occ_texts, normalize_embeddings=True)
        _occ_funcs = [_OCC_TO_FUNC.get(t, "unmapped") for t in occ_titles]
    return _occ_embeddings, _occ_funcs


def _extract_job_title(text: str) -> str:
    """Extract the most recent job title from resume text."""
    # Common patterns: "Title at Company" or "Title, Company" or "Title | Company"
    patterns = [
        r'(?:^|\n)([^,\n]{5,60})\s+(?:at|@|[–|-])\s+[A-Z]',  # Title at Company
        r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})(?:\s*\||\s*–|\s*-|\s*,\s*)',  # Title | or Title,
    ]
    for pat in patterns:
        match = re.search(pat, text[:500])  # search first 500 chars
        if match:
            title = match.group(1).strip()
            if len(title) > 5:
                return title
    return ""


def _embedding_probas(text: str) -> dict[str, float]:
    """Score functions by cosine similarity of resume-vs-occupation embeddings.
    Job title gets 2x weight — it carries disproportionate signal about function.
    """
    model = _load_embedder()
    occ_embs, occ_funcs = _load_occ_embeddings()

    # Extract and weight job title more heavily
    title = _extract_job_title(text)

    # Chunk full text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    if not chunks:
        chunks = [text]

    texts_to_embed = chunks.copy()
    weights = [1.0] * len(chunks)

    # Add job title with 2x weight
    if title and len(title) > 5:
        texts_to_embed.append(title)
        weights.append(2.0)

    embeddings = model.encode(texts_to_embed, normalize_embeddings=True)
    resume_emb = np.average(embeddings, axis=0, weights=weights)

    # Cosine similarity against all occupations
    similarities = np.dot(resume_emb, occ_embs.T)

    # Aggregate to functions: max similarity per function
    func_sims: dict[str, float] = {}
    for i, sim in enumerate(similarities):
        func = occ_funcs[i]
        if func != "unmapped":
            func_sims[func] = max(func_sims.get(func, 0), float(sim))

    # Normalize
    total = sum(func_sims.values()) or 1
    return {f: s / total for f, s in func_sims.items()}


# ── Fusion Layer ──

def match_role(
    text: str,
    aspiration_signal: dict[str, float] | None = None,
    study_signal: dict[str, float] | None = None,
    experience_signal: dict[str, float] | None = None,
) -> dict | None:
    """3-Signal ensemble: classifier + O*NET + embeddings → weighted fusion.

    Weights tuned on 2,484 resumes:
      - Classifier: 0.50 (best individual accuracy)
      - O*NET:      0.15 (niche occupation vocabulary)
      - Embeddings:  0.35 (semantic bridge)
    """
    has_intent_signal = bool(aspiration_signal or study_signal or experience_signal)

    # Get signals — embedding may fail if sentence_transformers unavailable
    s1 = _classifier_probas(text)
    s2 = _onet_probas(text)
    try:
        s3 = _embedding_probas(text)
    except Exception:
        s3 = {}

    DW = _CFG.default_weights
    signals: list[tuple[str, dict[str, float], float]] = [
        ("classifier", s1, DW.classifier),
        ("onet", s2, DW.onet),
        ("embeddings", s3, DW.embeddings),
    ]
    if has_intent_signal:
        IW = _CFG.intent_weights
        SB = _CFG.study_boost
        classifier_max = max(s1.values()) if s1 else 0
        max_study = max(study_signal.values()) if study_signal else 0
        study_boost = SB.boost if (classifier_max < SB.classifier_threshold and max_study > SB.study_threshold) else 0.0
        signals = [
            ("aspiration", aspiration_signal or {}, IW.aspiration),
            ("study", study_signal or {}, IW.study + study_boost),
            ("classifier", s1, IW.classifier - study_boost * 0.5),
            ("onet", s2, IW.onet),
            ("embeddings", s3, IW.embeddings - study_boost * 0.5),
            ("experience", experience_signal or {}, IW.experience),
        ]

    all_funcs = set()
    for _, signal, _ in signals:
        all_funcs.update(signal.keys())
    if not all_funcs:
        return None

    fused: dict[str, float] = {}
    for func in all_funcs:
        fused[func] = 0.0
        for _, signal, weight in signals:
            fused[func] += weight * signal.get(func, 0.0)

    total = sum(fused.values()) or 1
    for func in fused:
        fused[func] = fused[func] / total * 100

    ranked = sorted(fused.items(), key=lambda x: -x[1])
    best_func, best_pct = ranked[0]
    candidate_functions = [best_func]
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= 12:
        candidate_functions.append(ranked[1][0])

    alternatives = []
    for func, pct in ranked[1:]:
        if pct >= 3:
            alternatives.append({"function": func, "match_pct": round(pct)})

    return {
        "function": best_func,
        "level": "Entry",
        "match_pct": round(best_pct),
        "alternatives": alternatives[:5],
        "candidate_functions": candidate_functions,
        "all_probas": {func: round(p, 1) for func, p in ranked[:10]},
        "signal_breakdown": {
            name: {func: round(p * 100) for func, p in sorted(signal.items(), key=lambda x: -x[1])[:3]}
            for name, signal, _ in signals
        },
    }
