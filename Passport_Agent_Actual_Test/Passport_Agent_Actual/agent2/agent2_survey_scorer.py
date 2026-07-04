"""
Agent 2 — Survey Scorer
Reads Agent 1 raw_data JSON, computes EC/GC/RFF/CR scores, writes survey_scores JSON.

Usage:
  python agent2_survey_scorer.py --input agent1/outputs/ousmane_diallo_raw_data.json
"""

import argparse
import json
import os
import re
import sys


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

from tools.gemini_client import get_call_count, reset_call_count, GEMINI_API_KEY
from tools.gc_scorer  import score_gc
from tools.rff_scorer import score_rff
from tools.cr_scorer  import score_cr
from tools.ec_scorer  import score_ec

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _data_coverage(fields_raw: dict, pillar: str) -> float:
    pillar_fields = {k: v for k, v in fields_raw.items() if v.get("pillar") == pillar}
    if not pillar_fields:
        return 0.0
    found = sum(1 for v in pillar_fields.values() if v.get("status") == "found")
    return round(found / len(pillar_fields), 2)


def run(input_path: str) -> dict:
    print(f"Agent 2: Survey Scorer")
    print(f"Input:   {input_path}")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    student_name = data["student_name"]
    fields_raw   = data["fields"]

    # Build flat {canonical: value_or_None} — None when status is missing
    fields = {
        k: (v["value"] if v.get("status") == "found" else None)
        for k, v in fields_raw.items()
    }

    reset_call_count()

    print("  Scoring GC ...")
    gc_result  = score_gc(fields)

    print("  Scoring RFF ...")
    rff_result = score_rff(fields)

    print("  Scoring CR ...")
    cr_result  = score_cr(fields)

    print("  Scoring EC ...")
    ec_result  = score_ec(fields)

    # Attach data_coverage and strip internal status keys
    for pillar, result in [("EC", ec_result), ("GC", gc_result),
                           ("RFF", rff_result), ("CR", cr_result)]:
        result["data_coverage"] = _data_coverage(fields_raw, pillar)

    # Remove internal _c3_status/_c4_status before output
    cr_result.pop("_c3_status", None)
    cr_result.pop("_c4_status", None)

    missing_fields = [k for k, v in fields_raw.items() if v.get("status") == "missing"]

    print(f"  EC={ec_result['score']}  GC={gc_result['score']}  "
          f"RFF={rff_result['score']}  CR={cr_result['score']}")
    print(f"  Missing fields: {len(missing_fields)}")
    print(f"  LLM calls made: {get_call_count()}")

    output = {
        "student_name":   student_name,
        "email":          data.get("email"),
        "scores": {
            "EC":  ec_result,
            "GC":  gc_result,
            "RFF": rff_result,
            "CR":  cr_result,
        },
        "missing_fields":  missing_fields,
        "llm_calls_made":  get_call_count(),
    }

    slug = re.sub(r"[^a-z0-9]+", "_", student_name.lower()).strip("_")
    out_path = os.path.join(OUTPUTS_DIR, f"{slug}_survey_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print(f"Output: {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Agent 2: Survey Scorer")
    parser.add_argument("--input", required=True, help="Path to Agent 1 raw_data JSON")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
