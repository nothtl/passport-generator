"""Batch run new pipeline on half of SpeakHire students."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OLD_REPORTS = ROOT / "reports" / "tingli_v2_final"
NEW_REPORTS = ROOT / "reports" / "phase123_batch"
STUDENT_DATA = ROOT / "Passport_Agent_Actual_Test" / "Passport_Agent_Actual" / "student_data"

# College Scorecard API key
os.environ["COLLEGE_SCORECARD_API_KEY"] = "Hb93OyvFH3g9M8tzSAkYDXZ8oEz3RzCZ1kGAoIKx"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_student_dir(name: str) -> Path | None:
    target = normalize_name(name)
    best = None
    for path in STUDENT_DATA.iterdir():
        if not path.is_dir():
            continue
        current = normalize_name(path.name)
        if current == target:
            return path
        if target in current or current in target:
            best = path
    return best


def run_student(name: str, resume_text: str, old_result: dict) -> dict:
    """Run full pipeline for one student."""
    from recommender.analyze import analyze
    from recommender.college.college_matcher import match_colleges
    from recommender.major_recommender import recommend_majors
    from recommender.skill_bridge import generate_skill_path_summary
    from recommender.lifecycle import detect_student_stage
    from recommender.extract.location_extractor import extract_location

    # Find LinkedIn data
    student_dir = find_student_dir(name)
    linkedin_text = ""
    headline_text = ""
    study_text = ""
    career_goals_text = ""
    smart_goals_text = ""
    hope_to_gain_text = ""
    ideal_career_text = ""

    if student_dir:
        linkedin_files = sorted(student_dir.glob("*LinkedIn*"))
        if linkedin_files:
            linkedin_text = linkedin_files[0].read_text(encoding="utf-8", errors="replace")

        # Try to read SMART goals / career goals
        for goal_file in student_dir.glob("*goal*"):
            career_goals_text = goal_file.read_text(encoding="utf-8", errors="replace")[:2000]
        for smart_file in student_dir.glob("*SMART*"):
            smart_goals_text = smart_file.read_text(encoding="utf-8", errors="replace")[:2000]
        for gain_file in student_dir.glob("*hope*gain*"):
            hope_to_gain_text = gain_file.read_text(encoding="utf-8", errors="replace")[:2000]
        for ideal_file in student_dir.glob("*ideal*career*"):
            ideal_career_text = ideal_file.read_text(encoding="utf-8", errors="replace")[:2000]

    # Extract headline
    for line in linkedin_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            headline_text = stripped.strip("*").strip()
            break

    # Extract study from linkedin or resume
    for text in [linkedin_text, resume_text]:
        m = re.search(r"(Bachelor|Master|Associate|High School)[^\n]{0,80}", text, re.IGNORECASE)
        if m:
            study_text = m.group(0).strip()
            break

    # Run main pipeline
    t0 = time.time()
    from recommender.llm import LLMConfig
    result = analyze(
        resume_text=resume_text,
        linkedin_text=linkedin_text,
        headline_text=headline_text,
        career_goals_text=career_goals_text,
        smart_goals_text=smart_goals_text,
        hope_to_gain_text=hope_to_gain_text,
        ideal_career_text=ideal_career_text,
        study_text=study_text,
        top_k=10,
        lane_mode="hybrid",
        llm_config=LLMConfig(
        mode="hybrid",
        route_model="google/gemini-2.0-flash-001",
        evidence_model="google/gemini-2.0-flash-001",
        jobs_model="google/gemini-2.0-flash-001",
    ),  # Cheap ~$0.02/student
    )
    elapsed_analyze = time.time() - t0
    location = extract_location(resume_text)
    student_state = location.get("state", "")

    # Detect student stage and add internship info
    study_level = result.get("student_intent", {}).get("study_level", "")
    try:
        stage_info = detect_student_stage(study_level)
        result["student_stage_info"] = stage_info
    except Exception as e:
        result["student_stage_info"] = {"error": str(e)}

    # College recommendations
    try:
        student_profile = {
            "goal_domains": result.get("student_intent", {}).get("goal_domains", [result.get("function", "")]),
            "study_domains": result.get("student_intent", {}).get("study_domains", []),
            "skills": result.get("skills", [])[:20],
            "study_level": study_level,
            "preferred_state": student_state,
            "ideal_careers": result.get("ideal_careers", []),
        }
        t0c = time.time()
        colleges = match_colleges(student_profile, top_k=6)
        result["colleges"] = colleges
        result["_timing_college_s"] = round(time.time() - t0c, 1)
    except Exception as e:
        result["colleges"] = {"error": str(e), "tiers": {}}

    # Major recommendations
    try:
        majors = recommend_majors(
            result.get("function", "education"),
            result.get("skills", []),
            result.get("ideal_careers"),
            top_n=5,
        )
        result["recommended_majors"] = majors
    except Exception as e:
        result["recommended_majors"] = {"error": str(e)}

    # Skill path summary
    try:
        path_summary = generate_skill_path_summary(
            result.get("skills", []),
            result.get("core_gaps", []),
            result.get("ideal_careers"),
        )
        result["skill_path_summary"] = path_summary
    except Exception as e:
        result["skill_path_summary"] = str(e)

    # Pad internships to 5 if possible (use ready_now jobs as fallback)
    internships = result.get("internships", [])
    if len(internships) < 5:
        ready = result.get("ready_now", [])
        seen_t = {i["title"] for i in internships}
        for j in ready:
            if len(internships) >= 5:
                break
            if j["title"] not in seen_t:
                if j.get("fit", 0) < 30:  # Fix 3: skip low-quality padding
                    continue
                j_copy = dict(j)
                j_copy["internship"] = True
                internships.append(j_copy)
                seen_t.add(j_copy["title"])
        result["internships"] = internships

    # Cap tiers to exactly 5  
    result["ready_now"] = result.get("ready_now", [])[:5]
    result["internships"] = result.get("internships", [])[:5]

    # Pad aspirational to 5 from ready_now overflow
    aspirational = list(result.get("aspirational", []))
    if len(aspirational) < 5:
        ready = result.get("ready_now", [])
        seen_t = {j["title"] for j in aspirational}
        for j in ready:
            if len(aspirational) >= 5:
                break
            if j["title"] not in seen_t:
                aspirational.append(dict(j))
                seen_t.add(j["title"])
    result["aspirational"] = aspirational[:5]

    # Pad colleges to 5 by picking top matches
    colleges = result.get("colleges", {})
    if isinstance(colleges, dict) and not colleges.get("error"):
        all_colleges = []
        for tier in ["match", "safety", "reach"]:
            all_colleges.extend(colleges.get(tier, []))
        result["top_colleges"] = all_colleges[:5]
    result["_timing_analyze_s"] = round(elapsed_analyze, 1)
    # Also time colleges
    return result


def main():
    NEW_REPORTS.mkdir(parents=True, exist_ok=True)

    report_files = sorted(OLD_REPORTS.glob("*.json"))
    # Take half the students
    half = len(report_files) // 2
    selected = report_files  # All students

    print(f"Running pipeline on {len(selected)}/{len(report_files)} students...")
    print(f"Output: {NEW_REPORTS}")
    print()

    results = []
    for i, path in enumerate(selected, 1):
        name = path.stem
        with open(path, encoding="utf-8") as f:
            old_result = json.load(f)

        resume_text = old_result.get("resume", "")
        if not resume_text.strip():
            print(f"  [{i}/{len(selected)}] {name}: SKIP (no resume)")
            continue

        print(f"  [{i}/{len(selected)}] {name}...", end=" ", flush=True)
        try:
            result = run_student(name, resume_text, old_result)
            elapsed = result.get("_timing_analyze_s", 0)

            # Save
            out_path = NEW_REPORTS / f"{name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            # Quick summary
            func = result.get("function", "?")
            internships = len(result.get("internships", []))
            goal_jobs = len(result.get("aspirational", []))
            top_colleges = len(result.get("top_colleges", []))
            stage_info = result.get("student_stage_info", {})
            if isinstance(stage_info, dict):
                stage_label = stage_info.get("label", "?")
            else:
                stage_label = str(stage_info) if stage_info else "?"

            print(f"OK ({elapsed:.0f}s) | {func} | {stage_label} | {internships} intern, {goal_jobs} goal, {top_colleges} college")
            results.append({"name": name, "status": "ok", "elapsed": elapsed})
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({"name": name, "status": "error", "error": str(e)})

    # Summary
    print()
    print("=" * 50)
    print(f"Done! {len([r for r in results if r['status']=='ok'])}/{len(selected)} successful")
    print(f"Reports saved to: {NEW_REPORTS}")
    with open(NEW_REPORTS / "_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()