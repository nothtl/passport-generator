from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RECOMMENDER_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_RECOMMENDER_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def test_full_pipeline_on_abigail() -> None:
    from recommender.extract.skill_extractor import extract_skills_from_text

    resume_text = """
    Abigail Rodriguez
    Graphic Design & Content Creator for MetaBronx 2021 - Present
    I write and create content on issues that affect the community.
    I share historical and global stories on current events as well as personal
    interests like country music, poems, family stories, and other creative ideas.
    I train younger students on how to navigate the internship application process
    and I'm part of the program manager.

    Volunteer for SPEAKHIRE 2021-2022
    I help students with their remote learning and mentor them to be successful.

    Volunteer for Teen Outreach Program (TOP) at Crotona High School
    I write and record original stories for children. I help with outreach activities
    in the community like cleaning up the local parks and creating and posting flyers
    about upcoming events for local teens.

    Intern at My Bodega Online
    Built a database of EBT stores/warehouses across NYC metro area using Google Maps.
    """

    linkedin_text = """
    Program Manager at MetaBronx
    Organization of information, analysis of data for decision making.

    General Assistant and Digital Historian Instructor at The Glass Files
    Publishing tool used to write stories covering culture, traditions, experiences.
    Trained as a mentor through Digital Historian workshops, helping students build
    confidence to share their stories.

    Peer Group Connection Volunteer at Crotona International High School
    Teacher-run volunteer program matching upper grades with younger students
    for networking and mentoring. Created interactive programs using Kahoot, Padlet.
    """

    skills = extract_skills_from_text(resume_text + "\n" + linkedin_text)
    print("Stage 1 — Extracted skills:", skills)
    assert len(skills) > 0, "No skills extracted"

    from recommender.match.role_matcher import match_role
    best_role = match_role(skills)
    print(f"Stage 2 — Best role: {best_role['function']} (match={best_role['match_pct']}%)")
    assert best_role is not None, "No role matched"
    assert best_role["match_pct"] > 0, "Zero match score"

    from recommender.retrieve.retriever import retrieve_jds
    jds = retrieve_jds(
        function=best_role["function"],
        level=best_role["level"],
        student_skills=skills,
        top_k=5,
    )
    print(f"Stage 3 — Retrieved {len(jds)} JDs")
    assert isinstance(jds, list)

    from recommender.profile.aggregator import aggregate_skills, aggregate_passport
    all_skills = aggregate_skills(
        jds,
        matched_skills=best_role["matched_skills"],
        missing_skills=best_role["missing_skills"],
    )
    print(f"Stage 4a — Aggregated {len(all_skills)} skills across JDs")
    for s in sorted(all_skills, key=lambda x: x.frequency, reverse=True):
        mark = "Y" if s.student_has else "N"
        print(f"  [{mark}] {s.skill} (freq={s.frequency})")

    passport = aggregate_passport(jds)
    print(f"Stage 4b — Ideal passport: {passport}")

    from recommender.profile.gap_analyzer import analyze_gaps
    result = analyze_gaps(
        role_title=best_role["function"],
        function=best_role["function"],
        level=best_role["level"],
        match_pct=best_role["match_pct"],
        matched_skills=best_role["matched_skills"],
        missing_skills=best_role["missing_skills"],
        all_skills=all_skills,
        ideal_passport=passport,
    )
    print(f"Stage 4c — Gap analysis complete")
    print(f"  Role: {result['role']} ({result['level']})")
    print(f"  Match: {result['match_pct']}%")
    print(f"  Has: {result['skills_summary']['has']}, Missing: {result['skills_summary']['missing']}")
    print(f"  Top gaps: {[g['skill'] for g in result['top_gaps']]}")
    assert result["match_pct"] > 0
    print("\n[PASS] All stages passed")


def test_no_corpus_fallback() -> None:
    from recommender.retrieve.retriever import retrieve_jds
    jds = retrieve_jds(function="Marketing & Communications", level="Entry")
    assert isinstance(jds, list)
    print("[PASS] retriever handles missing corpus gracefully")


def test_full_pipeline_on_leila() -> None:
    from recommender.extract.skill_extractor import extract_skills_from_text

    resume_text = """
    Leila Titikpina
    Education: Ithaca College, BS in Health Science, Graduated May 2026

    Level 4 CNYMRC — Volunteer
    Managed the Flu POD on campus with 520 students and staff vaccinated.

    Platinum Home Care — Home Health Aide
    Completed 75 hours of training focused on assisting patients at home.

    Woodhull Community Care — Case Management Intern
    Created training for case management positions including how to use Epic and VMware.
    """

    skills = extract_skills_from_text(resume_text)
    print("Leila — Extracted skills:", skills)

    from recommender.match.role_matcher import match_role
    best_role = match_role(skills)
    print(f"Leila — Best role: {best_role['function']} (match={best_role['match_pct']}%)")
    assert best_role is not None


if __name__ == "__main__":
    test_full_pipeline_on_abigail()
    print()
    test_no_corpus_fallback()
    print()
    test_full_pipeline_on_leila()
    print("[PASS] All tests passed")
