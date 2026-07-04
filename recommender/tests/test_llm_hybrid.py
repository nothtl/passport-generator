from __future__ import annotations

import os
import sys

import pytest

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

os.environ.setdefault("RECOMMENDER_DISABLE_EMBEDDINGS", "1")
os.environ.setdefault("RECOMMENDER_DEFAULT_LLM_MODE", "off")


def test_route_judge_rejects_choice_outside_candidate_pool():
    from recommender.llm.provider import StaticLLMProvider
    from recommender.llm.route_judge import RouteIntentJudge

    provider = StaticLLMProvider(
        {
            "route": {
                "chosen_function": "arts-media",
                "secondary_function": "technology",
                "goal_domains": ["technology"],
                "study_domains": ["technology"],
                "goal_roles": ["intern"],
                "confidence": 0.92,
                "evidence_spans": [
                    {"section": "summary", "text": "seeking IT support internship"},
                    {"section": "experience", "text": "provided technical support"},
                ],
                "reason_codes": ["goal_alignment"],
            }
        }
    )
    judge = RouteIntentJudge(provider=provider, model="route-test")

    result = judge.judge(
        {
            "candidate_functions": ["technology", "support"],
            "headline_text": "Student seeking IT internship",
            "summary_text": "seeking IT support internship",
            "study_text": "High School Diploma in technical skills",
            "signal_breakdown": {"classifier": {"technology": 41, "support": 24}},
        }
    )

    assert result["accepted"] is False
    assert "invalid_chosen_function" in result["review_reasons"]


def test_evidence_normalizer_rejects_contact_artifacts_and_keeps_possible_non_scoring():
    from recommender.llm.evidence_judge import EvidenceNormalizerJudge
    from recommender.llm.provider import StaticLLMProvider

    provider = StaticLLMProvider(
        {
            "evidence": {
                "verified_skills": ["technical support", "hardware troubleshooting"],
                "possible_skills": ["python"],
                "rejected_skills": ["gmail", "pdf", "linkedin"],
                "implicit_skills": [
                    {"skill": "python", "reason": "FastAPI, ROS2", "confidence": "high"}
                ],
                "evidence_items": [
                    {
                        "canonical_skill": "technical support",
                        "source_section": "experience",
                        "exact_span": "Provided technical support and resolved device issues.",
                        "classification": "verified",
                    },
                    {
                        "canonical_skill": "hardware troubleshooting",
                        "source_section": "experience",
                        "exact_span": "Troubleshot hardware and software issues for staff.",
                        "classification": "verified",
                    },
                    {
                        "canonical_skill": "python",
                        "source_section": "projects",
                        "exact_span": "Built robotics experiments with FastAPI and ROS2.",
                        "classification": "possible",
                    },
                ],
            }
        }
    )
    judge = EvidenceNormalizerJudge(provider=provider, model="evidence-test")

    result = judge.judge(
        {
            "skills": ["gmail", "pdf", "linkedin", "technical support", "hardware troubleshooting"],
            "section_skills": {
                "summary": ["gmail", "pdf", "linkedin"],
                "experience": ["technical support", "hardware troubleshooting"],
                "projects": ["fastapi", "ros2"],
            },
            "allowed_sections": ["skills", "experience", "projects", "research", "leadership", "education"],
        }
    )

    assert result["accepted"] is True
    assert result["scored_skills"] == ["hardware troubleshooting", "technical support"]
    assert "gmail" not in result["scored_skills"]
    assert "python" not in result["scored_skills"]
    assert "python" in result["possible_skills"]
    assert result["implicit_skills"][0]["skill"] == "python"


def test_job_fit_gap_judge_blocks_licensed_roles_without_credentials():
    from recommender.llm.job_judge import JobFitGapJudge
    from recommender.llm.provider import StaticLLMProvider

    provider = StaticLLMProvider(
        {
            "jobs": {
                "jobs": [
                    {
                        "job_id": "physician-role",
                        "subdomain": "clinical-support",
                        "goal_fit_band": "medium",
                        "attainability_band": "blocked",
                        "hard_blockers": ["doctor_level", "licensure_required"],
                        "important_missing_skills": ["diagnosis", "patient assessment"],
                        "reason_codes": ["requires_license"],
                    },
                    {
                        "job_id": "medical-assistant-role",
                        "subdomain": "clinical-support",
                        "goal_fit_band": "high",
                        "attainability_band": "bridge",
                        "hard_blockers": [],
                        "important_missing_skills": ["patient intake", "documentation"],
                        "reason_codes": ["good_goal_fit"],
                    },
                ]
            }
        }
    )
    judge = JobFitGapJudge(provider=provider, model="jobs-test")
    base_jobs = [
        {
            "id": "physician-role",
            "title": "Hospitalist Physician",
            "fit": 48,
            "skills": ["diagnosis", "patient assessment"],
            "job_score_breakdown": {},
        },
        {
            "id": "medical-assistant-role",
            "title": "Medical Assistant",
            "fit": 42,
            "skills": ["patient intake", "documentation"],
            "job_score_breakdown": {},
        },
    ]

    decision = judge.judge(
        {
            "chosen_functions": ["healthcare"],
            "verified_skills": ["home health aide", "patient care"],
            "student_intent": {"goal_domains": ["healthcare"], "study_domains": ["healthcare"]},
            "jobs": base_jobs,
        }
    )
    ranked = judge.apply(base_jobs, decision)

    assert ranked[0]["title"] == "Medical Assistant"
    assert ranked[0]["job_score_breakdown"]["llm_adjustments"]["attainability_band"] == "bridge"
    assert ranked[1]["title"] == "Hospitalist Physician"
    assert ranked[1]["job_score_breakdown"]["llm_adjustments"]["hard_blockers"] == [
        "doctor_level",
        "licensure_required",
    ]


def test_llm_orchestrator_cache_replays_identical_stage_outputs(tmp_path: pytest.TempPathFactory):
    from recommender.llm.orchestrator import LLMConfig, LLMOrchestrator
    from recommender.llm.provider import StaticLLMProvider

    provider = StaticLLMProvider(
        {
            "route": {
                "chosen_function": "technology",
                "secondary_function": "support",
                "goal_domains": ["technology"],
                "study_domains": ["technology"],
                "goal_roles": ["intern"],
                "confidence": 0.82,
                "evidence_spans": [
                    {"section": "summary", "text": "seeking IT internship"},
                    {"section": "experience", "text": "provided technical support"},
                ],
                "reason_codes": ["goal_alignment"],
            }
        }
    )
    config = LLMConfig(mode="force", provider_name="static", max_calls=3, debug=True, cache_dir=str(tmp_path))
    orchestrator = LLMOrchestrator(config=config, provider=provider)
    context = {
        "route": {
            "candidate_functions": ["technology", "support"],
            "headline_text": "Seeking IT internship",
            "summary_text": "seeking IT internship",
            "study_text": "technical skills",
            "signal_breakdown": {"classifier": {"technology": 41, "support": 24}},
        }
    }

    first = orchestrator.run_stage("route", context["route"])
    second = orchestrator.run_stage("route", context["route"])

    assert first == second
    assert provider.call_count("route") == 1


def test_openrouter_provider_loads_key_from_dotenv(tmp_path, monkeypatch):
    from recommender.llm.provider import load_openrouter_api_key

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-openrouter-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert load_openrouter_api_key(search_roots=[tmp_path]) == "test-openrouter-key"
