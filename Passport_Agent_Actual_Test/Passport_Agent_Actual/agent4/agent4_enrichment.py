"""
Agent 4 — Enrichment Agent
Enriches Agent 2 survey scores with document-derived field values,
re-runs EC/GC/RFF/CR scorers, and adds new CT + CI pillars from documents.

Usage:
  python agent4_enrichment.py \
    --survey agent2/outputs/ousmane_diallo_survey_scores.json \
    --docs   agent3/outputs/ousmane_diallo_parsed_docs.json
"""

import argparse
import json
import os
import re
import sys

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _load_dotenv():
    """Load environment variables from .env — tries the project root and parent directories."""
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


def _slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _derive_raw_path(survey_path: str, slug: str) -> str | None:
    """
    Try to locate agent1's raw_data JSON from the survey path.
    Walks up from agent2/outputs/ → agent2/ → parent/ → agent1/outputs/.
    """
    survey_abs = os.path.abspath(survey_path)
    survey_dir = os.path.dirname(survey_abs)       # .../agent2/outputs/
    agent2_dir = os.path.dirname(survey_dir)        # .../agent2/
    base_dir   = os.path.dirname(agent2_dir)        # .../Passport_Agent_Actual/

    candidate = os.path.join(base_dir, 'agent1', 'outputs', f'{slug}_raw_data.json')
    if os.path.exists(candidate):
        return candidate

    # Fallback: look relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../agent4/
    parent_dir = os.path.dirname(script_dir)
    candidate2 = os.path.join(parent_dir, 'agent1', 'outputs', f'{slug}_raw_data.json')
    if os.path.exists(candidate2):
        return candidate2

    return None


def _extract_overlap(ct_arc: str, ci_arc: str) -> str:
    """Return capitalized project/name tokens shared between CT and CI arcs."""
    stop = {"The", "This", "His", "Her", "Their", "AI", "An", "In", "For", "By",
            "And", "Of", "To", "A", "On", "Is", "With", "From", "That", "It"}
    ct_tokens = set(re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", ct_arc))
    ci_tokens = set(re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", ci_arc))
    shared = (ct_tokens & ci_tokens) - stop
    # Also check lowercase project names like 'ml-from-scratch-od'
    ct_words = set(ct_arc.lower().split())
    ci_words = set(ci_arc.lower().split())
    shared_lower = {w.strip(".,;:\"'") for w in (ct_words & ci_words) if len(w) > 4}
    return ", ".join(shared | shared_lower)



def run(survey_path: str, docs_path: str) -> dict:
    print(f"[Agent4] Enrichment Agent")
    print(f"[Agent4] Survey  : {survey_path}")
    print(f"[Agent4] Docs    : {docs_path}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    with open(survey_path, encoding='utf-8') as f:
        survey_data = json.load(f)
    with open(docs_path, encoding='utf-8') as f:
        docs_data = json.load(f)

    student_name = survey_data.get("student_name", "")
    slug = _slugify(student_name)

    # ── Load Agent 1 raw fields ───────────────────────────────────────────────
    raw_path = _derive_raw_path(survey_path, slug)
    if not raw_path:
        print(f"[Agent4] ERROR: Could not locate agent1 raw_data JSON for slug '{slug}'")
        sys.exit(1)
    print(f"[Agent4] Raw data: {raw_path}")
    with open(raw_path, encoding='utf-8') as f:
        raw_data = json.load(f)

    missing = set(survey_data.get("missing_fields", []))
    print(f"[Agent4] Missing fields in survey: {len(missing)}")

    # ── Extract resume sections via LLM — handles any header format dynamically ──
    # Gemini identifies whatever section headers exist in this specific resume.
    # Only fills sections that keyword matching in Agent 3 left empty or thin.
    from tools.resume_section_extractor import extract_sections as _llm_extract_sections
    _resume = docs_data.get("resume") or {}
    _sections = _resume.get("sections") or {}
    if _resume.get("raw_text"):
        _llm_secs = _llm_extract_sections(_resume["raw_text"])
        if _llm_secs:
            for _k, _v in _llm_secs.items():
                if _v and str(_v).strip() and len(str(_sections.get(_k) or "").strip()) < 30:
                    _sections[_k] = _v
            print(f"[Agent4] Resume sections enriched via LLM: {sorted(_llm_secs)}")

    # ── Run enrichers ─────────────────────────────────────────────────────────
    from tools.ec_enricher  import enrich_ec
    from tools.gc_enricher  import enrich_gc
    from tools.rff_enricher import enrich_rff
    from tools.cr_enricher  import enrich_cr

    # Extract CPC session text for EC and RFF enrichers.
    # Champions log skill observations and career discussions that are directly
    # relevant to interpersonal communication (EC) and goal-setting (RFF).
    _cpc_fields = ["CPC All Session Text", "CPC What skill did you cover",
                   "CPC Discussion topics", "CPC What component skill did you cover"]
    cpc_parts = []
    for _f in _cpc_fields:
        _entry = raw_data["fields"].get(_f, {})
        if _entry.get("status") == "found" and _entry.get("value"):
            cpc_parts.append(f"{_f}: {_entry['value']}")
    cpc_session_text = "\n".join(cpc_parts)

    enriched: dict = {}
    enriched.update(enrich_ec(docs_data,  missing, cpc_session_text=cpc_session_text))
    enriched.update(enrich_gc(docs_data,  missing))
    enriched.update(enrich_rff(docs_data, missing, cpc_session_text=cpc_session_text))
    enriched.update(enrich_cr(docs_data,  missing))

    print(f"[Agent4] Fields enriched from documents: {len(enriched)}")

    # ── Merge + re-score EC/GC/RFF/CR ────────────────────────────────────────
    from tools.score_merger import merge_and_rescore, get_a2_call_count
    merged = merge_and_rescore(
        raw_data["fields"],
        enriched,
        missing,
        survey_data.get("scores", {}),
    )

    # ── Override C4 from actual resume if CPC logged no resume work ──────────
    # C4 is binary (0 or 100). If it's 0 but the student has a substantial resume,
    # the passport should reflect that — use the resume content as the evidence.
    _cr = merged["cr"]
    if _cr["sub_scores"].get("C4", 0) == 0:
        _resume_raw = (_resume.get("raw_text") or "").strip()
        if len(_resume_raw) > 300:
            from tools.gemini_client import call_gemini as _call_gemini, parse_json_resp as _parse_json
            _c4_check_prompt = (
                "Does the following text represent a real, complete student resume "
                "with substantive content (at least one job, internship, education record, "
                "or meaningful skills section)?\n\n"
                "RESUME TEXT:\n{text}\n\n"
                "Answer TRUE if this is a real resume with actual content. "
                "Answer FALSE only if the text is blank or has no real content.\n\n"
                'Respond ONLY in this JSON: {{"resume_built": true}} or {{"resume_built": false}}'
            ).replace("{text}", _resume_raw[:1200])
            _c4_resp = _call_gemini(_c4_check_prompt)
            _c4_parsed = _parse_json(_c4_resp)
            if _c4_parsed and str(_c4_parsed.get("resume_built", "")).lower() in ("true", "1", "yes"):
                _cr["sub_scores"]["C4"] = 100
                _cr["score"] = round(sum(_cr["sub_scores"].values()) / len(_cr["sub_scores"]), 1)
                print("[Agent4] CR C4 updated from resume evidence: 0 -> 100")

    # ── Score CT and CI ───────────────────────────────────────────────────────
    from tools.ct_scorer import score_ct
    from tools.ci_scorer import score_ci
    from tools.pillar_reasoner import generate_reasoning
    from tools.gemini_client import get_call_count as a4_call_count

    ct_result = score_ct(docs_data, student_name)
    ct_arc = ct_result.get("thinking_arc", "")

    ci_result = score_ci(docs_data, student_name, ct_arc=ct_arc)

    # Generic overlap check — re-run CI once with forbidden terms if same work cited
    overlap = _extract_overlap(ct_arc, ci_result.get("innovation_arc", ""))
    if overlap:
        ci_result = score_ci(docs_data, student_name, ct_arc=ct_arc, forbidden=overlap)

    # ── Generate reasoning for EC, GC, RFF, CR ────────────────────────────────
    # Notes are generated sequentially so each pillar gets the prior notes as
    # forbidden context — prevents all notes citing the same dominant role.
    # CT arc is included first since it's already written before this loop.
    print(f"[Agent4] Generating pillar reasoning...")
    prior_notes = [ct_arc] if ct_arc else []
    for key, result in [("EC", merged["ec"]), ("GC", merged["gc"]),
                        ("RFF", merged["rff"]), ("CR", merged["cr"])]:
        result["reasoning"] = generate_reasoning(
            key, result, docs_data, student_name, raw_data["fields"],
            prior_notes=list(prior_notes)
        )
        if result["reasoning"]:
            prior_notes.append(result["reasoning"])

    # CT and CI carry their own narrative fields; unify under "reasoning" too
    ct_result["reasoning"] = ct_result.get("thinking_arc", "")
    ci_result["reasoning"] = ci_result.get("innovation_arc", "")

    # ── Print score summary ───────────────────────────────────────────────────
    print(f"[Agent4] EC  : {merged['ec']['score']} (survey: {survey_data['scores']['EC']['score']})")
    print(f"[Agent4] GC  : {merged['gc']['score']} (survey: {survey_data['scores']['GC']['score']})")
    print(f"[Agent4] RFF : {merged['rff']['score']} (survey: {survey_data['scores']['RFF']['score']})")
    print(f"[Agent4] CR  : {merged['cr']['score']} (survey: {survey_data['scores']['CR']['score']})")
    print(f"[Agent4] CT  : {ct_result['score']} (source: {ct_result.get('source')})")
    print(f"[Agent4] CI  : {ci_result['score']} (source: {ci_result.get('source')})")

    # ── Assemble output ───────────────────────────────────────────────────────
    enriched_log = merged["enriched_log"]
    fields_still_missing = [f for f in missing if f not in enriched_log]
    total_llm_calls = a4_call_count() + get_a2_call_count()

    output = {
        "student_name": student_name,
        "email":        survey_data.get("email"),
        "scores": {
            "EC":  merged["ec"],
            "GC":  merged["gc"],
            "RFF": merged["rff"],
            "CR":  merged["cr"],
            "CT":  ct_result,
            "CI":  ci_result,
        },
        "enriched_fields":      enriched_log,
        "fields_enriched_count": len(enriched_log),
        "fields_still_missing": fields_still_missing,
        "llm_calls_made":       total_llm_calls,
    }

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUTS_DIR, f"{slug}_enriched_scores.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"[Agent4] Fields still missing (incl. Likert): {len(fields_still_missing)}")
    print(f"[Agent4] LLM calls made: {total_llm_calls}")
    print(f"[Agent4] Output: {out_path}")
    return output


def main():
    _load_dotenv()  # load .env before any tool imports read GEMINI_API_KEY
    parser = argparse.ArgumentParser(description="Agent 4: Enrichment Agent")
    parser.add_argument("--survey", required=True, help="Path to Agent 2 survey_scores JSON")
    parser.add_argument("--docs",   required=True, help="Path to Agent 3 parsed_docs JSON")
    args = parser.parse_args()
    run(args.survey, args.docs)


if __name__ == "__main__":
    main()
