from __future__ import annotations

import json
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

os.environ.setdefault("RECOMMENDER_DISABLE_EMBEDDINGS", "1")
os.environ.setdefault("RECOMMENDER_DEFAULT_LLM_MODE", "off")


def _read_fixture(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "fixtures", name)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_extract_filters_dash_and_noise_skills():
    from recommender.extract.skill_extractor import extract_skills_from_text

    skills = extract_skills_from_text(_read_fixture("leila_resume.txt"))

    assert "- resume" not in skills
    assert "activities" not in skills
    assert "students" not in skills
    assert "home health aide" in skills
    assert "mental health screening" in skills


def test_extract_skill_profile_ignores_contact_wrapper_artifacts():
    from recommender.extract.skill_extractor import extract_skill_profile

    profile = extract_skill_profile(_read_fixture("abigail_resume.txt"))

    assert "gmail" not in profile["skills"]
    assert "pdf" not in profile["skills"]
    assert "http" not in profile["skills"]
    assert "https" not in profile["skills"]


def test_parse_resume_sections_detects_summary_education_and_experience():
    from recommender.extract.section_parser import parse_resume_sections

    sections = parse_resume_sections(_read_fixture("abigail_linkedin.md"))

    assert "summary" in sections
    assert "experience" in sections
    assert "education" in sections
    assert "seeking education internship" in sections["summary"].lower()
    assert "program manager" in sections["experience"].lower()


def test_detect_student_intent_prefers_education_for_abigail():
    from recommender.match.student_intent import build_student_intent_profile

    profile = build_student_intent_profile(
        resume_text=_read_fixture("abigail_resume.txt"),
        linkedin_text=_read_fixture("abigail_linkedin.md"),
        headline_text="Student at City College seeking education Internship",
    )

    assert profile["has_student_intent"] is True
    assert "education" in profile["goal_domains"]
    assert profile["goal_signal"]["education"] > profile["goal_signal"].get("design", 0.0)


def test_detect_student_intent_extracts_goal_and_study_subdomains():
    from recommender.match.student_intent import build_student_intent_profile

    profile = build_student_intent_profile(
        resume_text="Student leader mentoring middle school students and coordinating after-school workshops.",
        headline_text="Aspiring youth mentor and future school counselor",
        career_goals_text="I want to work in after-school programs and student support.",
        study_text="Bachelor of Arts in Childhood Education",
    )

    assert "education" in profile["goal_domains"]
    assert "youth-programs" in profile["goal_subdomains"]
    assert "classroom-support" in profile["study_subdomains"]
    assert profile["study_level"] == "bachelor"


def test_match_role_keeps_professional_resume_without_intent():
    from recommender.match.ensemble_matcher import match_role

    result = match_role(
        "Software Engineer. Python, AWS, Docker, Kubernetes. Built REST APIs and led backend development."
    )

    assert result is not None
    assert result["function"] == "technology"
    assert result["match_pct"] > 0


def test_analyze_uses_goal_and_study_signals_for_student_profiles():
    from recommender.analyze import analyze

    result = analyze(
        _read_fixture("abigail_resume.txt"),
        linkedin_text=_read_fixture("abigail_linkedin.md"),
        headline_text="Student at City College seeking education Internship",
        top_k=5,
    )

    assert result["function"] in {"education", "social-service"}
    assert result["student_intent"]["has_student_intent"] is True
    assert result["lane_used"] in {"fast", "rescue"}
    assert result["subdomain"]
    assert "bridge_gaps" in result
    assert "stretch_gaps" in result
    assert "job_cluster" in result
    assert any(job["job_type"] in {"goal", "mixed"} for job in result["jobs"])
    assert any(
        any(token in job["title"].lower() for token in ["tutor", "teacher", "youth", "student", "education"])
        for job in result["jobs"][:5]
    )


def test_analyze_uses_study_signal_for_health_science_student():
    from recommender.analyze import analyze

    result = analyze(
        _read_fixture("leila_resume.txt"),
        linkedin_text=_read_fixture("leila_linkedin.md"),
        study_text="Bachelor of Science in Health Science",
        top_k=5,
    )

    assert result["function"] == "healthcare"
    assert "healthcare" in result["student_intent"]["study_domains"]
    assert any(
        any(token in job["title"].lower() for token in ["care", "health", "medical", "patient", "clinical"])
        for job in result["jobs"][:5]
    )


def test_implicit_skills_do_not_change_scored_skills_or_gaps():
    from recommender.analyze import analyze

    result = analyze(
        "Built backend services with FastAPI, PyTorch, and ROS2 for robotics experiments.",
        top_k=3,
    )

    implicit = {skill["skill"] for skill in result["implicit_skills"]}
    extracted = set(result["skills"])
    gaps = set(result["gaps"])

    assert "python" in implicit
    assert "python" not in extracted
    assert "python" in gaps or "python" in result["verify_gaps"]


def test_filter_job_quality_blanks_uuid_and_drops_junk():
    from recommender.retrieve.retriever import filter_job_records

    jobs = [
        {"title": "Tutor", "company": "123e4567-e89b-12d3-a456-426614174000", "url": "x"},
        {"title": "??", "company": "", "url": "y"},
        {"title": "Youth Program Assistant", "company": "school", "url": "z"},
    ]

    filtered = filter_job_records(jobs, candidate_function="education")

    assert filtered[0]["company"] == ""
    assert all(job["title"] != "??" for job in filtered)
    assert any(job["title"] == "Youth Program Assistant" for job in filtered)


def test_filter_job_quality_blocks_physician_roles_without_credentials():
    from recommender.retrieve.retriever import filter_job_records

    jobs = [
        {
            "title": "Physician, Early Career - Pasadena, CA - Onsite",
            "company": "health-system",
            "url": "x",
            "jd_markdown": "Licensed physician role requiring diagnosis, examinations, and medical license.",
            "skills": ["diagnosis", "examinations", "licensure"],
        },
        {
            "title": "Medical Assistant",
            "company": "clinic",
            "url": "y",
            "jd_markdown": "Support patients with intake, documentation, and clinical assistance.",
            "skills": ["patient intake", "documentation", "patient care"],
        },
    ]

    filtered = filter_job_records(
        jobs,
        candidate_function="healthcare",
        goal_domains=["healthcare"],
        study_domains=["healthcare"],
    )

    assert [job["title"] for job in filtered] == ["Medical Assistant"]


def test_score_jobs_prefers_goal_aligned_role():
    from recommender.match.student_intent import build_student_intent_profile
    from recommender.rank.job_ranker import rank_jobs

    intent = build_student_intent_profile(
        resume_text=_read_fixture("abigail_resume.txt"),
        linkedin_text=_read_fixture("abigail_linkedin.md"),
        headline_text="Student at City College seeking education Internship",
    )
    jobs = [
        {
            "id": "edu-1",
            "title": "After School Tutor",
            "company": "school",
            "url": "x",
            "jd_markdown": "Tutor students, support youth programs, and mentor learners.",
            "skills": ["tutoring", "mentoring", "youth engagement"],
        },
        {
            "id": "food-1",
            "title": "Pizza Maker",
            "company": "shop",
            "url": "y",
            "jd_markdown": "Prepare pizzas, handle food preparation, and maintain sanitation.",
            "skills": ["food preparation", "sanitation"],
        },
    ]

    ranked = rank_jobs(
        jobs=jobs,
        student_intent=intent,
        extracted_skills=["mentoring", "community outreach", "public speaking"],
        experience_skills=["mentoring", "community outreach"],
        top_k=2,
    )

    assert ranked[0]["title"] == "After School Tutor"
    assert ranked[0]["job_type"] in {"goal", "mixed"}
    assert "subdomain_alignment" in ranked[0]["job_score_breakdown"]
    assert ranked[0]["fit"] > ranked[1]["fit"]


def test_rank_jobs_penalizes_blocked_physician_role_without_credentials():
    from recommender.rank.job_ranker import rank_jobs

    intent = {
        "goal_domains": ["healthcare"],
        "goal_roles": [],
        "study_domains": ["healthcare"],
        "study_program": "Bachelor of Science in Health Science",
    }
    jobs = [
        {
            "id": "physician-role",
            "title": "Physician, Early Career - Pasadena, CA - Onsite",
            "company": "health-system",
            "url": "x",
            "jd_markdown": "Licensed physician role requiring diagnosis, examinations, and medical license.",
            "skills": ["diagnosis", "examinations", "licensure"],
        },
        {
            "id": "assistant-role",
            "title": "Medical Assistant",
            "company": "clinic",
            "url": "y",
            "jd_markdown": "Support patients with intake, documentation, and clinical assistance.",
            "skills": ["patient intake", "documentation", "patient care"],
        },
    ]

    ranked = rank_jobs(
        jobs=jobs,
        student_intent=intent,
        extracted_skills=["patient care", "home health aide"],
        experience_skills=["patient care"],
        top_k=2,
    )

    assert ranked[0]["title"] == "Medical Assistant"
    assert ranked[0]["fit"] > ranked[1]["fit"]


def test_analyze_marks_clear_technology_resume_as_fast_lane():
    from recommender.analyze import analyze

    result = analyze(
        "Software Engineer. Python, AWS, Docker, Kubernetes. Built REST APIs and led backend development.",
        top_k=3,
    )

    assert result["lane_used"] == "fast"
    assert result["subdomain"] in {"software", "engineering"}
    assert result["subdomain_confidence"] >= 0.5


def test_benchmark_config_loads():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "benchmark_expectations.json")
    data = json.loads(_read_fixture("benchmark_expectations.json"))

    assert os.path.exists(path)
    assert "Abigail Rodriguez" in data
    assert "required_title_keywords" in data["Abigail Rodriguez"]


def test_analyze_hybrid_route_override_keeps_benjamin_on_technology():
    from recommender.analyze import analyze
    from recommender.llm.orchestrator import LLMConfig
    from recommender.llm.provider import StaticLLMProvider

    resume_text = """
    Benjamin Medrano
    MOTIVATED HIGH SCHOOL GRADUATE | BILINGUAL CUSTOMER SERVICE & IT SUPPORT
    Provided technical support and organizational assistance in a professional internship setting.
    Troubleshot hardware and software issues, improving user experience.
    Assisted clients with technical problems, delivering clear and effective guidance.
    Supported school activities and provided translation services to facilitate communication.
    High School Diploma in Leadership, Bilingual Communication, and Technical Skills
    """
    provider = StaticLLMProvider(
        {
            "route": {
                "chosen_function": "technology",
                "secondary_function": "support",
                "goal_domains": ["technology"],
                "study_domains": ["technology"],
                "goal_subdomains": ["it-support"],
                "study_subdomains": ["it-support"],
                "goal_roles": ["intern"],
                "confidence": 0.81,
                "evidence_spans": [
                    {"section": "summary", "text": "IT SUPPORT"},
                    {"section": "experience", "text": "Provided technical support"},
                ],
                "reason_codes": ["experience_alignment"],
            }
        }
    )

    result = analyze(
        resume_text,
        top_k=3,
        llm_config=LLMConfig(mode="force", provider_name="static", debug=True, cache_dir=None),
        llm_provider=provider,
    )

    assert result["function"] == "technology"
    assert result["lane_used"] == "rescue"
    assert result["subdomain"] == "it-support"
    assert result["llm_stage_results"]["route"]["accepted"] is True
    assert result["review_reasons"] == []


def test_analyze_hybrid_evidence_cleanup_removes_contact_junk():
    from recommender.analyze import analyze
    from recommender.llm.orchestrator import LLMConfig
    from recommender.llm.provider import StaticLLMProvider

    provider = StaticLLMProvider(
        {
            "evidence": {
                "verified_skills": ["mentoring", "public speaking", "graphic design"],
                "possible_skills": ["javascript"],
                "rejected_skills": ["gmail", "pdf", "http", "https", "linkedin", "i am", "is a"],
                "implicit_skills": [
                    {"skill": "javascript", "reason": "website platform work", "confidence": "medium"}
                ],
                "evidence_items": [
                    {
                        "canonical_skill": "mentoring",
                        "source_section": "experience",
                        "exact_span": "I help students with their remote learning and mentor them to be successful.",
                        "classification": "verified",
                    },
                    {
                        "canonical_skill": "public speaking",
                        "source_section": "skills",
                        "exact_span": "I recently presented to a class of 25+ students.",
                        "classification": "verified",
                    },
                    {
                        "canonical_skill": "graphic design",
                        "source_section": "experience",
                        "exact_span": "Graphic Design & Content Creator for MetaBronx.",
                        "classification": "verified",
                    },
                    {
                        "canonical_skill": "javascript",
                        "source_section": "projects",
                        "exact_span": "platform designed for wineries that accelerates online ordering",
                        "classification": "possible",
                    },
                ],
            }
        }
    )

    result = analyze(
        _read_fixture("abigail_resume.txt"),
        linkedin_text=_read_fixture("abigail_linkedin.md"),
        headline_text="Student at City College seeking education Internship",
        top_k=5,
        llm_config=LLMConfig(mode="force", provider_name="static", debug=True, cache_dir=None),
        llm_provider=provider,
    )

    assert "gmail" not in result["skills"]
    assert "pdf" not in result["skills"]
    assert "graphic design" in result["skills"]
    assert "javascript" not in result["skills"]
    assert "javascript" in result["possible_skills"]
    assert result["llm_stage_results"]["evidence"]["accepted"] is True


def test_analyze_hybrid_job_judge_blocks_physician_for_leila():
    from recommender.analyze import analyze
    from recommender.llm.orchestrator import LLMConfig
    from recommender.llm.provider import StaticLLMProvider

    provider = StaticLLMProvider(
        {
                "jobs": {
                    "jobs": [
                        {
                            "job_id": "greenhouse:7731521003",
                            "subdomain": "clinical-support",
                            "goal_fit_band": "medium",
                            "attainability_band": "blocked",
                            "hard_blockers": ["doctor_level", "licensure_required"],
                            "important_missing_skills": ["diagnosis"],
                            "reason_codes": ["requires_license"],
                        },
                        {
                            "job_id": "greenhouse:4256834009",
                            "subdomain": "clinical-support",
                            "goal_fit_band": "high",
                            "attainability_band": "bridge",
                            "hard_blockers": [],
                            "important_missing_skills": ["documentation", "patient intake"],
                        "reason_codes": ["good_goal_fit"],
                    },
                ]
            }
        }
    )

    result = analyze(
        _read_fixture("leila_resume.txt"),
        linkedin_text=_read_fixture("leila_linkedin.md"),
        study_text="Bachelor of Science in Health Science",
        top_k=5,
        llm_config=LLMConfig(mode="force", provider_name="static", debug=True, cache_dir=None),
        llm_provider=provider,
    )

    titles = [job["title"] for job in result["jobs"][:5]]
    assistant_index = titles.index("Clinical Intern")

    assert "Physician, Early Career - Pasadena, CA - Onsite" not in titles[:2]
    assert result["jobs"][assistant_index]["job_score_breakdown"]["llm_adjustments"]["attainability_band"] == "bridge"


def test_analyze_falls_back_to_deterministic_when_llm_provider_errors():
    from recommender.analyze import analyze
    from recommender.llm.orchestrator import LLMConfig

    class ExplodingProvider:
        def complete_json(self, stage, system_prompt, user_prompt, model):
            raise RuntimeError("rate limit")

    result = analyze(
        _read_fixture("abigail_resume.txt"),
        linkedin_text=_read_fixture("abigail_linkedin.md"),
        headline_text="Student at City College seeking education Internship",
        top_k=3,
        llm_config=LLMConfig(mode="force", provider_name="static", debug=True, cache_dir=None),
        llm_provider=ExplodingProvider(),
    )

    assert result["function"] in {"education", "social-service"}
    assert result["llm_stage_results"]["route"]["accepted"] is False
    assert "llm_provider_error" in result["review_reasons"]
