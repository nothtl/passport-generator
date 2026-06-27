"""
O*NET-based occupation matching via Task Statement keyword indexing.

Bridges formal O*NET language → informal resume language by using
18,796 O*NET Task Statements (actual job activities) as the vocabulary.

Build: 2-second index, <1ms per query. No models, no downloads, no API.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
_INDEX_PATH = os.path.join(_DATA, "onet_occ_index.json")

# Caches
_occ_index: dict[str, Any] | None = None  # {occs: [...], keywords: {...}, num_occs: N}

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
}

# Occupation title → function mapping (word-boundary aware regex)
_FUNC_MAP: list[tuple[str, str]] = [
    (r"\bSoftware\b", "technology"), (r"\bDeveloper", "technology"),
    (r"\bEngineer", "technology"), (r"\bComputer\b", "technology"),
    (r"\bData\b", "technology"), (r"\bNetwork", "technology"),
    (r"\bWeb\b", "technology"), (r"\bIT\b", "technology"),
    (r"\bDatabase", "technology"), (r"\bProgrammer", "technology"),
    (r"\bMarketing\b", "marketing"), (r"\bSales\b", "sales"),
    (r"\bTeacher", "education"), (r"\bInstructor", "education"),
    (r"\bEducation", "education"), (r"\bTraining\b", "education"),
    (r"\bProfessor", "education"), (r"\bLibrarian", "education"),
    (r"\bNurse", "healthcare"), (r"\bMedical\b", "healthcare"),
    (r"\bHealth\b", "healthcare"), (r"\bClinical\b", "healthcare"),
    (r"\bPhysician", "healthcare"), (r"\bTherapist", "healthcare"),
    (r"\bDental\b", "healthcare"), (r"\bCustomer Service\b", "support"),
    (r"\bSupport\b", "support"), (r"\bDesigner", "design"),
    (r"\bDesign\b", "design"), (r"\bGraphic\b", "design"),
    (r"\bManager", "ops"), (r"\bOperations\b", "ops"),
    (r"\bAdministrative\b", "administrative"), (r"\bOffice\b", "administrative"),
    (r"\bClerk", "administrative"), (r"\bReceptionist", "administrative"),
    (r"\bSecretar", "administrative"), (r"\bAccountant", "finance"),
    (r"\bFinancial\b", "finance"), (r"\bAuditor", "finance"),
    (r"\bBookkeeping\b", "finance"), (r"\bChef\b", "food-service"),
    (r"\bCook", "food-service"), (r"\bRestaurant\b", "food-service"),
    (r"\bKitchen\b", "food-service"), (r"\bFood\b", "food-service"),
    (r"\bBarista", "food-service"), (r"\bWait", "food-service"),
    (r"\bHost", "food-service"), (r"\bConstruction\b", "skilled-trade"),
    (r"\bElectrician", "skilled-trade"), (r"\bPlumber", "skilled-trade"),
    (r"\bCarpenter", "skilled-trade"), (r"\bWelder", "skilled-trade"),
    (r"\bHVAC\b", "skilled-trade"), (r"\bRoofer", "skilled-trade"),
    (r"\bDrywall\b", "skilled-trade"), (r"\bDriver\b", "logistics"),
    (r"\bTruck\b", "logistics"), (r"\bLogistics\b", "logistics"),
    (r"\bWarehouse\b", "logistics"), (r"\bDelivery\b", "logistics"),
    (r"\bManufacturing\b", "manufacturing"), (r"\bProduction\b", "manufacturing"),
    (r"\bAssembly\b", "manufacturing"), (r"\bHospitality\b", "hospitality"),
    (r"\bHotel\b", "hospitality"),
]

# Exclude "Systems" from technology when the occupation is clearly non-tech
_SYSTEMS_BLACKLIST = {
    "Transportation", "Vehicle", "Inspector", "Aircraft", "Aviation",
    "Power", "Water", "Wastewater", "Electrical", "Mechanical",
}


def _build_index() -> dict:
    """Build keyword→occupation index from O*NET Task Statements."""
    import pandas as pd

    onet = os.path.join(_HERE, "..", "data", "onet")
    tasks_df = pd.read_excel(os.path.join(onet, "Task Statements.xlsx"))

    occs = []
    keyword_df = Counter()
    keyword_occs: dict[str, set[int]] = {}

    for (soc, title), group in tasks_df.groupby(["O*NET-SOC Code", "Title"]):
        occ_idx = len(occs)
        occs.append({"title": title, "soc": soc})
        all_words = set()
        for t in group["Task"].dropna():
            all_words.update(
                w for w in re.findall(r"\b[a-z][a-z]+\b", t.lower())
                if w not in _STOPS
            )
        for w in all_words:
            keyword_df[w] += 1
            keyword_occs.setdefault(w, set()).add(occ_idx)

    return {
        "occs": occs,
        "num_occs": len(occs),
        "keyword_df": dict(keyword_df),
        "keyword_occs": {k: list(v) for k, v in keyword_occs.items()},
    }


def _load_index() -> dict:
    """Load or build the occupation index."""
    global _occ_index
    if _occ_index is None:
        if os.path.exists(_INDEX_PATH):
            with open(_INDEX_PATH, encoding="utf-8") as f:
                _occ_index = json.load(f)
        else:
            _occ_index = _build_index()
            with open(_INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump(_occ_index, f)
    return _occ_index


def _occ_to_function(title: str) -> str | None:
    """Map O*NET occupation title to function category."""
    # Check systems blacklist first
    for pattern, func in _FUNC_MAP:
        if re.search(pattern, title):
            if func == "technology" and pattern == r"\bSystems\b":
                if any(bad in title for bad in _SYSTEMS_BLACKLIST):
                    continue
            return func
    return None


def match_onet_occupations(
    text: str,
    top_k: int = 5,
) -> list[dict]:
    """Match resume text against 923 O*NET occupations via TF-IDF keyword overlap.

    Returns list of {title, soc, score} dicts.
    """
    index = _load_index()
    occs = index["occs"]
    num_occs = index["num_occs"]
    keyword_df = index["keyword_df"]
    keyword_occs = index["keyword_occs"]

    # Tokenize resume
    words = [w for w in re.findall(r"\b[a-z][a-z]+\b", text.lower()) if w not in _STOPS]
    wc = Counter(words)

    # Score each occupation by TF-IDF keyword overlap
    occ_scores = Counter()
    for w, count in wc.items():
        w_key = str(w)
        if w_key in keyword_occs and keyword_df[w_key] > 2:
            idf = math.log(num_occs / keyword_df[w_key])
            tf = 1 + math.log(count)
            weight = tf * idf
            for occ_idx in keyword_occs[w_key]:
                occ_scores[occ_idx] += weight

    # Rank and return top-k
    results = []
    for occ_idx, score in occ_scores.most_common(top_k):
        occ = occs[occ_idx]
        func = _occ_to_function(occ["title"])
        results.append({
            "title": occ["title"],
            "soc": occ["soc"],
            "score": round(score, 1),
            "function": func,
        })

    return results


def match_role_onet(text: str) -> dict | None:
    """Match resume text to best-fit function using O*NET occupation data.

    Returns dict compatible with the old match_role format:
    {function, match_pct, onet_title, alternatives, ...}
    """
    occs = match_onet_occupations(text, top_k=15)
    if not occs:
        return None

    # Score functions: best occupation + breadth bonus.
    # Breadth only matters as a tiebreaker when scores are close.
    func_best_score: dict[str, float] = {}
    func_best_title: dict[str, str] = {}
    func_breadth: dict[str, int] = {}
    for occ in occs:
        func = occ["function"]
        if func:
            if func not in func_best_score or occ["score"] > func_best_score[func]:
                func_best_score[func] = occ["score"]
                func_best_title[func] = occ["title"]
            # Count additional occupations with meaningful scores
            if occ["score"] > 8:
                func_breadth[func] = func_breadth.get(func, 0) + 1

    # Combine: best score + small breadth bonus (only for close calls)
    func_score = {}
    for func in func_best_score:
        base = func_best_score[func]
        breadth = func_breadth.get(func, 1)
        # Each additional occupation beyond the first adds 10% of its score
        func_score[func] = base * (1 + 0.1 * (breadth - 1))

    if not func_best_score:
        return None

    # Rank functions by score (best occupation + breadth bonus)
    ranked = sorted(func_score.items(), key=lambda x: -x[1])

    # Confidence: what fraction of total signal goes to each function?
    total_signal = sum(s for _, s in ranked)
    if total_signal == 0:
        total_signal = 1

    best_func, best_score_val = ranked[0]
    best_title = func_best_title[best_func]
    match_pct = round(best_score_val / total_signal * 100)

    alternatives = []
    for func, score_val in ranked[1:]:
        pct = round(score_val / total_signal * 100)
        if pct >= 5:
            alternatives.append({
                "function": func,
                "match_pct": pct,
                "onet_title": func_best_title[func],
            })

    return {
        "function": best_func,
        "level": "Entry",
        "match_pct": match_pct,
        "onet_title": best_title,
        "top_occupations": [
            {"title": o["title"], "score": o["score"], "function": o["function"]}
            for o in occs[:5]
        ],
        "alternatives": alternatives[:5],
    }
