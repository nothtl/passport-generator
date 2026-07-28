"""Skill Knowledge Graph for query-time skill expansion.

Bridges vocabulary gaps like "computer vision" → "AI" → "machine learning"
that flat synonym dictionaries (ESCO) can't handle.

Graph structure:
  - Nodes: normalized skill names
  - Edges: (skill_a, skill_b, weight) where weight ∈ [0, 1]
    - PMI co-occurrence edges (from JD corpus, weight = min(1, PMI/10))
    - ESCO synonym edges (weight = 0.85)
    - Same-function co-occurrence (weight = min(1, PMI/8))

Query-time: BFS 2 hops from student skills, collect reachable skills
with cumulative path weight > 0.3.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(HERE, "skill_graph.json")


def _norm(s: str) -> str:
    return re.sub(r"[- ,/]", "", str(s).lower())


def build() -> dict[str, dict[str, float]]:
    """Build the skill graph from PMI + ESCO data. Run once, cached to JSON."""
    if os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH, encoding="utf-8") as f:
            return json.load(f)

    print("Building skill graph (one-time, ~30s)...")
    graph: dict[str, dict[str, float]] = {}

    def _add_edge(a: str, b: str, weight: float):
        an, bn = _norm(a), _norm(b)
        if an == bn:
            return
        if an not in graph:
            graph[an] = {}
        if bn not in graph:
            graph[bn] = {}
        # Keep max weight if edge already exists
        graph[an][bn] = max(graph[an].get(bn, 0), weight)
        graph[bn][an] = max(graph[bn].get(an, 0), weight)

    # 1. ESCO synonym edges
    esco_path = os.path.join(HERE, "esco_synonyms.json")
    if os.path.exists(esco_path):
        with open(esco_path, encoding="utf-8") as f:
            esco = json.load(f)
        for canonical, aliases in esco.items():
            for alias in aliases[:5]:  # top 5 aliases per canonical
                _add_edge(canonical, alias, 0.85)
        print(f"  ESCO: {len(esco)} canonical skills with synonym edges")

    # 2. PMI co-occurrence edges from per-function corpora
    from recommender.retrieve.retriever import _compute_pmi, _CORPUS_DIR

    functions = ["technology", "healthcare", "education", "finance", "sales",
                 "design", "arts-media", "social-service", "ops", "support",
                 "engineering", "science"]
    pmi_edges = 0
    for func in functions:
        try:
            pmi = _compute_pmi(func)
        except Exception:
            continue
        for (a, b), score in pmi.items():
            weight = min(1.0, score / 10.0)  # normalize PMI to [0, 1]
            if weight > 0.2:  # only strong co-occurrences
                _add_edge(a, b, weight)
                pmi_edges += 1

    print(f"  PMI: {pmi_edges} co-occurrence edges across {len(functions)} functions")

    # 3. Hardcoded domain bridges (manual curation for common gaps)
    _DOMAIN_BRIDGES = {
        # AI/ML cluster
        ("computervision", "machinelearning"): 0.7,
        ("computervision", "deeplearning"): 0.7,
        ("computervision", "ai"): 0.6,
        ("machinelearning", "ai"): 0.8,
        ("machinelearning", "datascience"): 0.7,
        ("deeplearning", "neuralnetworks"): 0.8,
        ("nlp", "machinelearning"): 0.7,
        ("objectdetection", "computervision"): 0.9,
        ("yolo", "objectdetection"): 0.8,
        ("yolo", "computervision"): 0.7,
        ("fastapi", "backend"): 0.7,
        ("fastapi", "api"): 0.8,
        ("azure", "cloud"): 0.7,
        ("cosmosdb", "database"): 0.7,
        # Healthcare bridges
        ("bilingual", "patientservices"): 0.5,
        ("mentoring", "youthcounseling"): 0.6,
        ("communityoutreach", "socialwork"): 0.6,
        # Design bridges
        ("adobe", "graphicdesign"): 0.9,
        ("photoshop", "graphicdesign"): 0.9,
        ("illustrator", "graphicdesign"): 0.9,
    }
    for (a, b), weight in _DOMAIN_BRIDGES.items():
        _add_edge(a, b, weight)
    print(f"  Domain bridges: {len(_DOMAIN_BRIDGES)} hardcoded edges")

    # Serialize
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f)
    print(f"  Saved graph: {len(graph)} nodes, {sum(len(v) for v in graph.values())} edges")

    return graph


def expand_skills(student_skills: list[str], max_hops: int = 2, min_weight: float = 0.3, top_k: int = 30) -> list[str]:
    """Expand student skills by traversing the skill graph.

    Args:
        student_skills: Original extracted skills
        max_hops: Maximum BFS depth (1-2 recommended)
        min_weight: Minimum cumulative path weight to include a skill
        top_k: Maximum expanded skills to return

    Returns:
        Original skills + graph-expanded skills (deduplicated)
    """
    graph = build()
    if not graph:
        return list(student_skills)

    normalized = {_norm(s) for s in student_skills}
    visited: dict[str, float] = {}  # skill → best cumulative weight
    queue = deque()

    # Seed with student skills (weight = 1.0)
    for s in normalized:
        if s in graph:
            visited[s] = 1.0
            queue.append((s, 0, 1.0))  # (node, hops, cumulative_weight)

    # BFS
    while queue:
        node, hops, cum_weight = queue.popleft()
        if hops >= max_hops:
            continue
        if node not in graph:
            continue
        for neighbor, edge_weight in graph[node].items():
            new_weight = cum_weight * edge_weight
            if new_weight < min_weight:
                continue
            if neighbor in visited and visited[neighbor] >= new_weight:
                continue
            visited[neighbor] = new_weight
            queue.append((neighbor, hops + 1, new_weight))

    # Return original + expanded, sorted by weight
    expanded = [(s, w) for s, w in visited.items() if s not in normalized]
    expanded.sort(key=lambda x: -x[1])
    result = list(student_skills) + [s for s, _ in expanded[:top_k]]
    return result


if __name__ == "__main__":
    # One-time build
    g = build()
    print(f"\nGraph: {len(g)} nodes")

    # Test expansion
    test_skills = ["python", "fastapi", "computer vision", "bilingual"]
    expanded = expand_skills(test_skills, max_hops=2, top_k=15)
    print(f"\nTest expansion:")
    print(f"  Input:  {test_skills}")
    print(f"  Output: {expanded}")
