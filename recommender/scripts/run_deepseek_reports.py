"""Regenerate all 19 Tingli reports using DeepSeek v4 flash as the rescue LLM."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recommender.analyze import analyze
from recommender.llm import LLMConfig

OLD_REPORTS = ROOT / "reports" / "tingli_reports"
NEW_REPORTS = ROOT / "reports" / "tingli_deepseek"
STUDENT_DATA = ROOT / "Passport_Agent_Actual_Test" / "Passport_Agent_Actual" / "student_data"
BENCHMARKS_PATH = ROOT / "recommender" / "tests" / "fixtures" / "benchmark_expectations.json"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # resolves to deepseek-v4-flash


class DeepSeekProvider:
    def __init__(self, api_key: str = "", model: str = DEEPSEEK_MODEL):
        self.api_key = api_key or DEEPSEEK_KEY
        self.model = model

    def complete_json(self, stage: str, system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DeepSeek API key not configured")

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model or self.model,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }

        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    DEEPSEEK_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                if attempt == 2:
                    print(f"  [DeepSeek] Failed after 3 attempts: {e}")
                    raise
                time.sleep(2 ** attempt)
        return {}


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


def _derive_ideal_careers(old_result: dict, headline: str) -> list[str]:
    """Derive ideal careers from old function + LinkedIn headline."""
    careers = []
    if headline:
        match = re.search(r'seeking\s+(.+?)\s+(?:Internship|career|job|role|position)', headline, re.I)
        if match:
            careers.append(match.group(1).strip())
        match = re.search(r'(\w+)\s+Student', headline, re.I)
        if match:
            careers.append(match.group(1).strip())
    func_map = {
        "education": ["Teacher", "Teaching Assistant", "Youth Counselor"],
        "healthcare": ["Nurse", "Medical Assistant", "Healthcare Professional"],
        "technology": ["Software Engineer", "IT Support Specialist", "Developer"],
        "finance": ["Financial Analyst", "Accountant", "Finance Professional"],
        "sales": ["Sales Representative", "Account Manager"],
        "design": ["Graphic Designer", "Creative Professional"],
        "arts-media": ["Content Creator", "Photographer", "Media Professional"],
        "social-service": ["Social Worker", "Community Organizer", "Youth Advocate"],
        "protective-service": ["Police Officer", "Security Professional"],
        "legal": ["Paralegal", "Legal Assistant"],
        "ops": ["Operations Coordinator", "Project Manager"],
        "support": ["Customer Service Representative", "Support Specialist"],
        "food-service": ["Chef", "Restaurant Manager"],
        "skilled-trade": ["Electrician", "Technician"],
    }
    old_func = (old_result.get("function") or "").lower()
    for c in func_map.get(old_func, []):
        if c not in careers:
            careers.append(c)
    return careers[:3]


def _load_benchmarks() -> dict:
    if BENCHMARKS_PATH.exists():
        with open(BENCHMARKS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _evaluate_benchmark(name: str, result: dict, benchmarks: dict) -> dict | None:
    if name not in benchmarks:
        return None
    expected = benchmarks[name]
    top_titles = [job.get("title", "").lower() for job in result.get("jobs", [])[:3]]
    function_ok = result.get("function") in expected.get("expected_functions", [])
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
        "function": result.get("function", ""),
        "subdomain": result.get("subdomain", ""),
        "lane_used": result.get("lane_used", ""),
        "top_jobs": [job.get("title", "") for job in result.get("jobs", [])[:3]],
        "core_gaps": result.get("core_gaps", [])[:5],
        "needs_review": result.get("needs_review", False),
        "review_reasons": result.get("review_reasons", []),
    }


def main():
    if not DEEPSEEK_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    NEW_REPORTS.mkdir(parents=True, exist_ok=True)
    benchmarks = _load_benchmarks()

    provider = DeepSeekProvider()
    llm_config = LLMConfig(
        mode="force",  # force LLM rescue for all students
        provider_name="deepseek",
        route_model=DEEPSEEK_MODEL,
        evidence_model=DEEPSEEK_MODEL,
        jobs_model=DEEPSEEK_MODEL,
        max_calls=9,
        debug=False,
        cache_dir=str(ROOT / "recommender" / ".cache" / "llm_deepseek"),
    )

    results = []
    benchmark_results = []
    total_start = time.time()

    for path in sorted(OLD_REPORTS.glob("*.json")):
        name = path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        with open(path, encoding="utf-8") as f:
            old_result = json.load(f)

        resume_text = old_result.get("resume", "")
        student_dir = _find_student_dir(name)

        linkedin_text = ""
        if student_dir:
            linkedin_files = list(student_dir.glob("*LinkedIn.md"))
            if linkedin_files:
                linkedin_text = linkedin_files[0].read_text(encoding="utf-8")

        headline_text = _extract_headline(linkedin_text)
        study_text = _extract_study_text(linkedin_text, resume_text)
        ideal_careers = _derive_ideal_careers(old_result, headline_text)

        t0 = time.time()
        try:
            new_result = analyze(
                resume_text=resume_text,
                linkedin_text=linkedin_text,
                headline_text=headline_text,
                study_text=study_text,
                ideal_careers=ideal_careers,
                top_k=5,
                lane_mode="hybrid",
                llm_config=llm_config,
                llm_provider=provider,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            new_result = {"error": str(e), "function": "?", "lane_used": "error"}

        elapsed = time.time() - t0
        print(f"  Function: {new_result.get('function','?')}")
        print(f"  Subdomain: {new_result.get('subdomain','?')}")
        print(f"  Lane: {new_result.get('lane_used','?')}")
        print(f"  Confidence: {new_result.get('confidence','?')}%")
        print(f"  needs_review: {new_result.get('needs_review','?')}")
        print(f"  LLM stages: {[s['stage'] for s in new_result.get('llm_trace',[])]}")
        print(f"  Time: {elapsed:.1f}s")
        top_jobs = new_result.get("jobs", [])[:3]
        for i, j in enumerate(top_jobs, 1):
            print(f"  Job {i}: [{j.get('fit',0)}%] {j.get('title','')[:70]}")

        out_path = NEW_REPORTS / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_result, f, indent=2, ensure_ascii=False)

        results.append({
            "name": name,
            "function": new_result.get("function", "?"),
            "subdomain": new_result.get("subdomain", "?"),
            "lane": new_result.get("lane_used", "?"),
            "confidence": new_result.get("confidence", 0),
            "needs_review": new_result.get("needs_review", True),
            "elapsed": elapsed,
            "skills": len(new_result.get("skills", [])),
            "rejected": len(new_result.get("rejected_skills", [])),
        })

        bm = _evaluate_benchmark(name, new_result, benchmarks)
        if bm:
            benchmark_results.append(bm)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY — {len(results)} reports in {total_elapsed/60:.1f}m")
    print(f"{'='*60}")
    print(f"{'Student':<28} {'Func':<16} {'Subdomain':<22} {'Lane':<8} {'Conf':>4} {'Rv':>3} {'Skills':>5} {'Rej':>4} {'Time':>5}")
    print("-" * 120)
    for r in results:
        rv = "YES" if r["needs_review"] else "OK"
        print(f"{r['name']:<28} {r['function']:<16} {r['subdomain']:<22} {r['lane']:<8} {r['confidence']:>3}% {rv:>3} {r['skills']:>4}s {r['rejected']:>3}r {r['elapsed']:>4.0f}s")

    needs_count = sum(1 for r in results if r["needs_review"])
    rescue_count = sum(1 for r in results if r["lane"] == "rescue")
    print(f"\nneeds_review: {needs_count}/{len(results)}")
    print(f"rescue lane: {rescue_count}/{len(results)}")
    print(f"avg confidence: {sum(r['confidence'] for r in results)/len(results):.0f}%")

    if benchmark_results:
        print(f"\nBenchmarks:")
        for bm in benchmark_results:
            status = "PASS" if bm["passed"] else "FAIL"
            print(f"  {bm['name']}: {status} | {bm['function']} | {bm['subdomain']} | {bm['lane_used']}")
            print(f"    Jobs: {bm['top_jobs']}")
            if bm["forbidden_hit"]:
                print(f"    Forbidden: {bm['forbidden_hit']}")

    # Write summary
    summary_path = NEW_REPORTS / "summary.md"
    lines = [
        "# DeepSeek v4 Flash — Recommender Reports",
        "",
        f"Generated {len(results)} reports in {total_elapsed/60:.1f}m using DeepSeek v4 flash.",
        f"LLM config: mode=force, model={DEEPSEEK_MODEL}, max_calls=9",
        "",
        f"## Stats",
        f"- needs_review: {needs_count}/{len(results)}",
        f"- rescue lane: {rescue_count}/{len(results)}",
        f"- avg confidence: {sum(r['confidence'] for r in results)/len(results):.0f}%",
        "",
        "## Per-Student Results",
        "",
        "| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |",
        "|---------|----------|-----------|------|------|--------|--------|----------|------|",
    ]
    for r in results:
        rv = "YES" if r["needs_review"] else "OK"
        lines.append(f"| {r['name']} | {r['function']} | {r['subdomain']} | {r['lane']} | {r['confidence']}% | {rv} | {r['skills']} | {r['rejected']} | {r['elapsed']:.0f}s |")

    if benchmark_results:
        lines.extend(["", "## Benchmarks", ""])
        for bm in benchmark_results:
            status = "PASS" if bm["passed"] else "FAIL"
            lines.extend([
                f"### {bm['name']} — {status}",
                f"- Function: `{bm['function']}` (subdomain: `{bm['subdomain']}`)",
                f"- Lane: `{bm['lane_used']}`",
                f"- Top jobs: {bm['top_jobs']}",
                f"- Core gaps: {bm['core_gaps']}",
                f"- Forbidden hits: {bm['forbidden_hit'] or 'None'}",
                "",
            ])

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
