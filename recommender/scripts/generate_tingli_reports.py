from __future__ import annotations

import json
import re
from pathlib import Path

from recommender.analyze import analyze

ROOT = Path(__file__).resolve().parents[2]
OLD_REPORTS = ROOT / "reports" / "tingli_reports"
NEW_REPORTS = ROOT / "reports" / "tingli_reports_reworked"
SUMMARY = ROOT / "reports" / "tingli_reports_reworked_summary.md"
BENCHMARKS = ROOT / "recommender" / "tests" / "fixtures" / "benchmark_expectations.json"
STUDENT_DATA = ROOT / "Passport_Agent_Actual_Test" / "Passport_Agent_Actual" / "student_data"


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_student_dir(name: str) -> Path | None:
    target = _normalize_name(name)
    best = None
    for path in STUDENT_DATA.iterdir():
        if not path.is_dir():
            continue
        current = _normalize_name(path.name)
        if current == target:
            return path
        if target in current or current in target:
            best = path
    return best


def _extract_headline(linkedin_text: str) -> str:
    for line in linkedin_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            return stripped.strip("*").strip()
    return ""


def _extract_study_text(linkedin_text: str, resume_text: str) -> str:
    match = re.search(r"(Bachelor|Master|Associate)[^\n]+", linkedin_text, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()
    match = re.search(r"(Bachelor|Master|Associate)[^\n]+", resume_text, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return ""


def _load_benchmarks() -> dict:
    with open(BENCHMARKS, encoding="utf-8") as handle:
        return json.load(handle)


def _evaluate_benchmark(name: str, old_result: dict, new_result: dict, benchmarks: dict) -> dict | None:
    if name not in benchmarks:
        return None
    expected = benchmarks[name]
    top_titles = [job.get("title", "").lower() for job in new_result.get("jobs", [])[:3]]
    function_ok = new_result.get("function") in expected.get("expected_functions", [])
    required_ok = any(
        any(keyword in title for keyword in expected.get("required_title_keywords", []))
        for title in top_titles
    )
    forbidden_hit = [
        keyword
        for keyword in expected.get("forbidden_title_keywords", [])
        if any(keyword in title for title in top_titles)
    ]
    return {
        "name": name,
        "function_ok": function_ok,
        "required_ok": required_ok,
        "forbidden_hit": forbidden_hit,
        "passed": function_ok and required_ok and not forbidden_hit,
        "old_function": old_result.get("function", ""),
        "new_function": new_result.get("function", ""),
        "old_jobs": [job.get("title", "") for job in old_result.get("jobs", [])[:3]],
        "new_jobs": [job.get("title", "") for job in new_result.get("jobs", [])[:3]],
        "lane_used": new_result.get("lane_used", ""),
        "subdomain": new_result.get("subdomain", ""),
        "new_gaps": new_result.get("gaps", [])[:5],
        "bridge_gaps": new_result.get("bridge_gaps", [])[:5],
        "stretch_gaps": new_result.get("stretch_gaps", [])[:5],
        "implicit_skills": [item.get("skill", "") for item in new_result.get("implicit_skills", [])],
        "verify_gaps": new_result.get("verify_gaps", [])[:5],
    }


def main() -> None:
    NEW_REPORTS.mkdir(parents=True, exist_ok=True)
    benchmarks = _load_benchmarks()
    benchmark_results = []

    for path in sorted(OLD_REPORTS.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            old_result = json.load(handle)
        name = path.stem
        resume_text = old_result.get("resume", "")
        student_dir = _find_student_dir(name)
        linkedin_text = ""
        if student_dir:
            linkedin_files = list(student_dir.glob("*LinkedIn.md"))
            if linkedin_files:
                linkedin_text = linkedin_files[0].read_text(encoding="utf-8")
        headline_text = _extract_headline(linkedin_text)
        study_text = _extract_study_text(linkedin_text, resume_text)
        new_result = analyze(
            resume_text=resume_text,
            linkedin_text=linkedin_text,
            headline_text=headline_text,
            study_text=study_text,
            top_k=5,
            lane_mode="hybrid",
        )
        out_path = NEW_REPORTS / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(new_result, handle, indent=2)

        benchmark = _evaluate_benchmark(name, old_result, new_result, benchmarks)
        if benchmark:
            benchmark_results.append(benchmark)

    lines = [
        "# Tingli Reworked Recommender Summary",
        "",
        f"Generated {len(list(NEW_REPORTS.glob('*.json')))} reworked reports.",
        "",
        "## Benchmarks",
        "",
    ]
    for result in benchmark_results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.extend(
            [
                f"### {result['name']} — {status}",
                f"- Old function: `{result['old_function']}`",
                f"- New function: `{result['new_function']}`",
                f"- Lane/subdomain: `{result['lane_used']}` / `{result['subdomain']}`",
                f"- Old top jobs: {', '.join(result['old_jobs']) or 'None'}",
                f"- New top jobs: {', '.join(result['new_jobs']) or 'None'}",
                f"- New gaps: {', '.join(result['new_gaps']) or 'None'}",
                f"- Bridge gaps: {', '.join(result['bridge_gaps']) or 'None'}",
                f"- Stretch gaps: {', '.join(result['stretch_gaps']) or 'None'}",
                f"- Implicit skills: {', '.join(result['implicit_skills']) or 'None'}",
                f"- Verify gaps: {', '.join(result['verify_gaps']) or 'None'}",
                f"- Forbidden hits: {', '.join(result['forbidden_hit']) or 'None'}",
                "",
            ]
        )

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
