"""Quick smoke test — verifies the pipeline loads and runs."""
from __future__ import annotations

import sys
import os

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def test_extract():
    from recommender.extract.skill_extractor import extract_skills_from_text
    skills = extract_skills_from_text(
        "Software Engineer. Python, AWS, React. Built REST APIs. CS degree."
    )
    assert len(skills) > 0, "No skills extracted"


def test_match():
    from recommender.match.ensemble_matcher import match_role
    result = match_role(
        "Software Engineer. Python, AWS, Docker, Kubernetes. CS degree. Led development team."
    )
    assert result is not None
    assert "function" in result
    assert result["match_pct"] > 0


def test_retrieve():
    from recommender.retrieve.retriever import retrieve_jds
    jds = retrieve_jds("technology", "Entry", top_k=3)
    assert isinstance(jds, list)


if __name__ == "__main__":
    test_extract()
    test_match()
    test_retrieve()
    print("[PASS] All tests passed")
