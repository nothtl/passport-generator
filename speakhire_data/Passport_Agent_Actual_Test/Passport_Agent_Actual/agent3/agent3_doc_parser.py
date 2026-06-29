"""
Agent 3 — Document Parser
Extracts structured data from resume PDF, LinkedIn markdown, and GitHub.
No scoring, no LLM — pure extraction.

Usage:
  python agent3_doc_parser.py --input agent1/outputs/ousmane_diallo_raw_data.json
"""

import argparse
import json
import os
import re


def _load_dotenv():
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(here, '.env'),
        os.path.join(here, '..', '.env'),
        os.path.join(here, '..', '..', '.env'),
    ]:
        if os.path.isfile(candidate):
            with open(candidate, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, _, v = line.partition('=')
                    k = k.strip()
                    if k:
                        os.environ[k] = v.strip()
            return

_load_dotenv()  # must run before tools imports so GEMINI_API_KEY is set at import time

from tools.student_folder_finder import find_student_folder, find_all_resume_pdfs, find_linkedin_md
from tools.resume_parser   import parse_resume
from tools.linkedin_parser import parse_linkedin
from tools.github_scraper  import scrape_github
from tools.github_finder   import find_github_username

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def run(input_path: str) -> dict:
    print(f"[Agent3] Document Parser")
    print(f"[Agent3] Input: {input_path}")

    with open(input_path, encoding='utf-8') as f:
        agent1_data = json.load(f)

    student_name = agent1_data["student_name"]
    slug = _slugify(student_name)

    # ── Find student folder and files ────────────────────────────────────────
    student_dir    = find_student_folder(student_name)
    resume_paths   = find_all_resume_pdfs(student_dir) if student_dir else []
    linkedin_path  = find_linkedin_md(student_dir) if student_dir else None

    # ── Parse available sources ───────────────────────────────────────────────
    resume_data   = None
    linkedin_data = None
    github_data   = None
    sources_found = []

    if resume_paths:
        all_texts: list[str] = []
        merged_sections: dict = {}
        for path in resume_paths:
            print(f"[Agent3] Parsing resume: {os.path.basename(path)}")
            r = parse_resume(path)
            if r.get("raw_text"):
                all_texts.append(f"=== {os.path.basename(path)} ===\n{r['raw_text']}")
                for k, v in (r.get("sections") or {}).items():
                    if k != "contact" and v and not merged_sections.get(k):
                        merged_sections[k] = v

        if all_texts:
            resume_data = {
                "raw_text": "\n\n".join(all_texts),
                "sections": merged_sections,
            }
            first_parsed = parse_resume(resume_paths[0])
            resume_data["sections"]["contact"] = (
                first_parsed.get("sections", {}).get("contact") or {}
            )
            sources_found.append("resume")
            if len(resume_paths) > 1:
                print(f"[Agent3] Merged {len(resume_paths)} resumes: "
                      f"{[os.path.basename(p) for p in resume_paths]}")
        else:
            print("[Agent3] Resume(s) produced no text — excluded from sources_found")
    else:
        print("[Agent3] No resume PDF found")

    if linkedin_path:
        print(f"[Agent3] Parsing LinkedIn: {os.path.basename(linkedin_path)}")
        linkedin_data = parse_linkedin(linkedin_path)
        sources_found.append("linkedin")
    else:
        print(f"[Agent3] No LinkedIn markdown found")

    # ── Find GitHub username ──────────────────────────────────────────────────
    resume_text = (resume_data or {}).get("raw_text", "")
    linkedin_text = ""
    if linkedin_path and os.path.exists(linkedin_path):
        with open(linkedin_path, encoding='utf-8') as f:
            linkedin_text = f.read()

    github_username = find_github_username(resume_text, linkedin_text)

    if github_username:
        print(f"[Agent3] Scraping GitHub: @{github_username}")
        github_data = scrape_github(github_username)
        sources_found.append("github")
    else:
        print("[Agent3] No GitHub username found in resume or LinkedIn")

    # ── Assemble and write output ─────────────────────────────────────────────
    output = {
        "student_name":  student_name,
        "sources_found": sources_found,
        "resume":        resume_data,
        "linkedin":      linkedin_data,
        "github":        github_data,
    }

    out_path = os.path.join(OUTPUTS_DIR, f"{slug}_parsed_docs.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print(f"[Agent3] Sources found: {sources_found}")
    print(f"[Agent3] Output: {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Agent 3: Document Parser")
    parser.add_argument("--input", required=True, help="Path to Agent 1 raw_data JSON")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
