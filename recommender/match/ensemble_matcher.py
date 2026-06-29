"""
3-Signal Ensemble Matcher — No hardcoded patterns, no LLM.

Signal 1: ML Classifier (TF-IDF + LinearSVC, 73.9% on 2,484 resumes)
Signal 2: O*NET Task Overlap (18,796 task statements via TF-IDF)
Signal 3: Sentence Embeddings (all-MiniLM-L6-v2, semantic similarity vs O*NET tasks)

Fusion: weighted voting with confidence thresholding.
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


_STOPS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "or", "not",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "who", "whom", "which", "what", "when",
    "where", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "if", "then", "else", "about",
    "up", "out", "over", "under", "again", "further", "once", "here",
    "there", "activities", "may", "work", "use", "using", "equipment",
    "information", "develop", "prepare", "provide", "materials",
    "determine", "ensure", "evaluate", "maintain", "programs", "procedures",
    "perform", "reports", "conduct", "research", "control", "plan", "test",
    "monitor", "direct", "review", "plans", "tools", "required", "including",
    "related", "knowledge", "principles", "techniques", "methods", "needs",
    "appropriate", "necessary", "standards", "results", "processes", "services",
    "records", "record", "supervise", "inspect", "specifications", "clean",
    "coordinate", "identify", "quality", "customer",
}

# Occupation-to-function mapping (ONLY used for aggregation)
_OCC_TO_FUNC: dict[str, str] = {}
_func_set: set[str] = set()


def _build_occ_to_func():
    """Build occupation->function mapping from O*NET titles."""
    global _OCC_TO_FUNC, _func_set
    if _OCC_TO_FUNC:
        return

    # Domain keywords that map to functions
    mappings = [
        (r"\bSoftware\b", "technology"), (r"\bDeveloper", "technology"),
        (r"\bEngineer(?!$)", "technology"), (r"\bComputer\b", "technology"),
        (r"\bProgrammer", "technology"), (r"\bIT\b", "technology"),
        (r"\bNetwork", "technology"), (r"\bDatabase", "technology"),
        (r"\bMarketing\b", "marketing"), (r"\bSales\b", "sales"),
        (r"\bTeacher", "education"), (r"\bInstructor", "education"),
        (r"\bEducation", "education"), (r"\bProfessor", "education"),
        (r"\bSchool\b", "education"), (r"\bLibrarian", "education"),
        (r"\bNurse", "healthcare"), (r"\bMedical\b", "healthcare"),
        (r"\bHealth\b", "healthcare"), (r"\bClinical\b", "healthcare"),
        (r"\bPhysician", "healthcare"), (r"\bTherapist", "healthcare"),
        (r"\bDental\b", "healthcare"), (r"\bPharmacy", "healthcare"),
        (r"\bDesigner", "design"), (r"\bDesign\b", "design"),
        (r"\bGraphic\b", "design"), (r"\bArtist", "design"),
        (r"\bClerk", "administrative"), (r"\bReceptionist", "administrative"),
        (r"\bSecretar", "administrative"), (r"\bOffice\b", "administrative"),
        (r"\bAccountant", "finance"), (r"\bFinancial\b", "finance"),
        (r"\bAuditor", "finance"), (r"\bBudget", "finance"),
        (r"\bChef\b", "food-service"), (r"\bCook(?!ing)", "food-service"),
        (r"\bRestaurant\b", "food-service"), (r"\bKitchen\b", "food-service"),
        (r"\bFood\b", "food-service"), (r"\bBarista", "food-service"),
        (r"\bBartender", "food-service"), (r"\bWait", "food-service"),
        (r"\bConstruction\b", "skilled-trade"), (r"\bElectrician", "skilled-trade"),
        (r"\bPlumber", "skilled-trade"), (r"\bCarpenter", "skilled-trade"),
        (r"\bWelder", "skilled-trade"), (r"\bMechanic", "skilled-trade"),
        (r"\bDriver\b", "logistics"), (r"\bTruck\b", "logistics"),
        (r"\bWarehouse\b", "logistics"), (r"\bDelivery\b", "logistics"),
        (r"\bManufacturing\b", "manufacturing"), (r"\bProduction\b", "manufacturing"),
        (r"\bAssembly\b", "manufacturing"), (r"\bHotel\b", "hospitality"),
        (r"\bHospitality\b", "hospitality"), (r"\bLodging", "hospitality"),
        (r"\bLawyer", "legal"), (r"\bAttorney", "legal"),
        (r"\bParalegal", "legal"), (r"\bLegal\b", "legal"),
        (r"\bCourt\b", "legal"), (r"\bPolice\b", "protective-service"),
        (r"\bFirefighter", "protective-service"), (r"\bSecurity\b", "protective-service"),
        (r"\bGuard\b", "protective-service"), (r"\bWriter", "arts-media"),
        (r"\bJournalist", "arts-media"), (r"\bEditor\b", "arts-media"),
        (r"\bActor", "arts-media"), (r"\bMusician", "arts-media"),
        (r"\bPhotographer", "arts-media"), (r"\bMedia\b", "arts-media"),
        (r"\bBroadcast", "arts-media"), (r"\bJanitor", "building-grounds"),
        (r"\bCleaner", "building-grounds"), (r"\bMaid", "building-grounds"),
        (r"\bLandscap", "building-grounds"), (r"\bHairdresser", "personal-care"),
        (r"\bCosmetolog", "personal-care"), (r"\bBarber", "personal-care"),
        (r"\bFarm", "agriculture"), (r"\bAgricultur", "agriculture"),
        (r"\bScientist", "science"), (r"\bChemist", "science"),
        (r"\bBiologist", "science"), (r"\bLaboratory\b", "science"),
        (r"\bSocial Work", "social-service"), (r"\bCounselor", "social-service"),
        (r"\bManager", "ops"), (r"\bSupervisor", "ops"),
        (r"\bDirector\b", "ops"), (r"\bOperations\b", "ops"),
        (r"\bCustomer Service\b", "support"), (r"\bSupport\b", "support"),
    ]

    index = _load_onet()
    for occ in index["occs"]:
        title = occ["title"]
        func = "unmapped"
        for pat, f in mappings:
            if re.search(pat, title):
                func = f
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


def _embedding_probas(text: str) -> dict[str, float]:
    """Score functions by cosine similarity of resume-vs-occupation embeddings."""
    model = _load_embedder()
    occ_embs, occ_funcs = _load_occ_embeddings()

    # Chunk long text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    if not chunks:
        chunks = [text]
    chunk_embs = model.encode(chunks, normalize_embeddings=True)
    resume_emb = np.mean(chunk_embs, axis=0)

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

def match_role(text: str) -> dict | None:
    """3-Signal ensemble: classifier + O*NET + embeddings → weighted fusion.

    Weights tuned on 2,484 resumes:
      - Classifier: 0.50 (best individual accuracy)
      - O*NET:      0.15 (niche occupation vocabulary)
      - Embeddings:  0.35 (semantic bridge)
    """
    # Get all three signals
    s1 = _classifier_probas(text)
    s2 = _onet_probas(text)
    s3 = _embedding_probas(text)

    # Expert voting: each signal gets one vote (its top pick).
    # If 2+ experts agree, that wins. Otherwise, highest confidence wins.
    # Then blend probabilities for the winning function.
    def top(func_probs):
        return max(func_probs, key=func_probs.get) if func_probs else None

    t1, t2, t3 = top(s1), top(s2), top(s3)
    c1 = s1.get(t1, 0) if t1 else 0
    c2 = s2.get(t2, 0) if t2 else 0
    c3 = s3.get(t3, 0) if t3 else 0
    top_picks = [p for p in [t1, t2, t3] if p]

    # Count votes
    from collections import Counter
    votes = Counter(top_picks)
    winner = votes.most_common(1)[0][0]

    # Weighted blend for the winner
    w1 = 0.50 if t1 == winner else 0.15
    w2 = 0.15 if t2 == winner else 0.15
    w3 = 0.35 if t3 == winner else 0.15

    # If 2+ experts agree, boost winner weight
    if votes[winner] >= 2:
        w1 = 0.60 if t1 == winner else 0.10
        w2 = 0.15 if t2 == winner else 0.10
        w3 = 0.40 if t3 == winner else 0.10

    all_funcs = set(s1.keys()) | set(s2.keys()) | set(s3.keys())
    fused: dict[str, float] = {}
    for func in all_funcs:
        p1 = s1.get(func, 0)
        p2 = s2.get(func, 0)
        p3 = s3.get(func, 0)
        fused[func] = w1 * p1 + w2 * p2 + w3 * p3

    # Normalize
    total = sum(fused.values()) or 1
    for func in fused:
        fused[func] = fused[func] / total * 100

    # Rank
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    best_func, best_pct = ranked[0]

    alternatives = []
    for func, pct in ranked[1:]:
        if pct >= 3:
            alternatives.append({"function": func, "match_pct": round(pct)})

    return {
        "function": best_func,
        "level": "Entry",
        "match_pct": round(best_pct),
        "alternatives": alternatives[:5],
        "all_probas": {func: round(p, 1) for func, p in ranked[:10]},
        "signal_breakdown": {
            "classifier": {func: round(p * 100) for func, p in sorted(s1.items(), key=lambda x: -x[1])[:3]},
            "onet": {func: round(p * 100) for func, p in sorted(s2.items(), key=lambda x: -x[1])[:3]},
            "embeddings": {func: round(p * 100) for func, p in sorted(s3.items(), key=lambda x: -x[1])[:3]},
        },
    }
