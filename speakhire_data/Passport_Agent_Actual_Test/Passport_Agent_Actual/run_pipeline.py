"""
run_pipeline.py — Run all 5 agents for every student discovered in agent1/outputs/.

Usage:
  python run_pipeline.py                        # run all students
  python run_pipeline.py --exclude ousmane_diallo
  python run_pipeline.py --only abigail_rodriguez bianka_pena

Student names are read from existing agent1/outputs/*_raw_data.json files.
No student names are hardcoded in this script.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "Full_speakhire_data.zip")

A1_OUTPUTS = os.path.join(HERE, "agent1", "outputs")
A2_OUTPUTS = os.path.join(HERE, "agent2", "outputs")
A3_OUTPUTS = os.path.join(HERE, "agent3", "outputs")
A4_OUTPUTS = os.path.join(HERE, "agent4", "outputs")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _discover_students() -> list[tuple[str, str]]:
    """Return list of (student_name, slug) from existing agent1 output files."""
    students = []
    if not os.path.isdir(A1_OUTPUTS):
        print(f"[Pipeline] ERROR: agent1/outputs/ not found at {A1_OUTPUTS}")
        return students
    for fname in sorted(os.listdir(A1_OUTPUTS)):
        if not fname.endswith("_raw_data.json"):
            continue
        fpath = os.path.join(A1_OUTPUTS, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("student_name", "")
            if name:
                students.append((name, _slugify(name)))
        except Exception as e:
            print(f"[Pipeline] Warning: could not read {fname}: {e}")
    return students


def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess command. Returns True on success."""
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=HERE,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[Pipeline] FAILED — {label} (exit {result.returncode})")
        return False
    return True


def run_student(name: str, slug: str) -> bool:
    print(f"\n{'='*60}")
    print(f"[Pipeline] Processing: {name}")
    print(f"{'='*60}")

    a1_out = os.path.join(A1_OUTPUTS, f"{slug}_raw_data.json")
    a2_out = os.path.join(A2_OUTPUTS, f"{slug}_survey_scores.json")
    a3_out = os.path.join(A3_OUTPUTS, f"{slug}_parsed_docs.json")
    a4_out = os.path.join(A4_OUTPUTS, f"{slug}_enriched_scores.json")

    steps = [
        (["agent1/agent1_data_loader.py", "--student", name, "--zip", ZIP_PATH],
         "Agent1 data loader"),
        (["agent2/agent2_survey_scorer.py", "--input", a1_out],
         "Agent2 survey scorer"),
        (["agent3/agent3_doc_parser.py", "--input", a1_out],
         "Agent3 doc parser"),
        (["agent4/agent4_enrichment.py", "--survey", a2_out, "--docs", a3_out],
         "Agent4 enrichment"),
        (["agent5/agent5_renderer.py", "--input", a4_out],
         "Agent5 renderer"),
    ]

    for cmd, label in steps:
        print(f"\n[Pipeline] Running {label}...")
        if not _run(cmd, f"{name} / {label}"):
            return False

    print(f"\n[Pipeline] DONE — {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full 5-agent pipeline for all students")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Slugs to skip (e.g. ousmane_diallo)")
    parser.add_argument("--only", nargs="*", default=[],
                        help="If set, only run these slugs")
    args = parser.parse_args()

    students = _discover_students()
    if not students:
        print("[Pipeline] No students found in agent1/outputs/. Run Agent 1 first.")
        sys.exit(1)

    exclude_set = set(args.exclude)
    only_set    = set(args.only)

    selected = [
        (name, slug) for name, slug in students
        if slug not in exclude_set
        and (not only_set or slug in only_set)
    ]

    print(f"[Pipeline] Students to process: {len(selected)} / {len(students)} total")
    if exclude_set:
        print(f"[Pipeline] Excluded: {', '.join(exclude_set)}")

    passed, failed = [], []
    for name, slug in selected:
        ok = run_student(name, slug)
        (passed if ok else failed).append(name)

    print(f"\n{'='*60}")
    print(f"[Pipeline] SUMMARY: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"[Pipeline] Failed students: {', '.join(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
