"""
Score merger.
Re-runs the exact EC/GC/RFF/CR scorers from agent2/tools/ on the merged field set.
Agent2 tools are loaded under the 'a2t' package alias to avoid colliding with
agent4's own 'tools' package that is already registered in sys.modules.
"""
import os, sys, types, importlib, importlib.util

_A2_TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'agent2', 'tools')
)


def _load_a2_module(pkg: str, name: str) -> types.ModuleType:
    full = f'{pkg}.{name}'
    if full in sys.modules:
        return sys.modules[full]
    path = os.path.join(_A2_TOOLS_DIR, f'{name}.py')
    spec = importlib.util.spec_from_file_location(full, path)
    mod  = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_a2(pkg: str = 'a2t') -> None:
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__    = [_A2_TOOLS_DIR]
        p.__package__ = pkg
        sys.modules[pkg] = p
    # Load in dependency order: gemini_client must precede scorers that use it
    for name in ('gemini_client', 'gc_scorer', 'rff_scorer', 'cr_scorer', 'ec_scorer'):
        _load_a2_module(pkg, name)


_bootstrap_a2()

_score_ec  = sys.modules['a2t.ec_scorer'].score_ec
_score_gc  = sys.modules['a2t.gc_scorer'].score_gc
_score_rff = sys.modules['a2t.rff_scorer'].score_rff
_score_cr  = sys.modules['a2t.cr_scorer'].score_cr


def get_a2_call_count() -> int:
    mod = sys.modules.get('a2t.gemini_client')
    return mod.get_call_count() if mod else 0


def _is_falsy(val) -> bool:
    if val is None:
        return True
    sv = str(val).strip().upper()
    return sv in ("FALSE", "NO", "0", "0.0", "", "NONE", "NAN", "N/A")


def _numeric_val(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def merge_and_rescore(
    raw_fields:      dict,
    enriched_fields: dict,
    missing_fields:  set,
    survey_scores:   dict,
) -> dict:
    """
    raw_fields:      agent1 fields dict {canonical: {"value": ..., "status": ...}}
    enriched_fields: {canonical: enriched_value} from all enrichers
    missing_fields:  set of field names that were missing in the survey
    survey_scores:   agent2 scores dict {"EC": {...}, "GC": {...}, ...}
    Returns merged dict with re-scored results and enriched_log.
    """
    # Start with Agent 1 raw field values
    merged = {k: v.get("value") for k, v in raw_fields.items()}

    # Pass 1: fill missing fields from docs
    enriched_log = {}
    for field_name, value in enriched_fields.items():
        if field_name in missing_fields and merged.get(field_name) is None and value is not None:
            merged[field_name] = value
            enriched_log[field_name] = {"value": value, "source": "docs", "was_missing": True}

    # Pass 2: doc-override — when SPEAKHIRE has a lower/false value but docs show better evidence.
    # Resume and LinkedIn are more current than enrollment self-reports; take the stronger value.
    for field_name, value in enriched_fields.items():
        if field_name in enriched_log or value is None:
            continue
        current = merged.get(field_name)
        should_override = False
        if str(value).strip().upper() in ("TRUE", "YES") and _is_falsy(current):
            should_override = True
        if not should_override:
            e_num = _numeric_val(value)
            c_num = _numeric_val(current) if not _is_falsy(current) else 0.0
            if e_num is not None and c_num is not None and e_num > c_num:
                should_override = True
        if should_override:
            merged[field_name] = value
            enriched_log[field_name] = {
                "value": value, "source": "docs",
                "was_missing": False, "doc_override": True,
            }

    # Re-run all 4 scorers with merged fields
    ec_result  = _score_ec(merged)
    gc_result  = _score_gc(merged)
    rff_result = _score_rff(merged)
    cr_result  = _score_cr(merged)

    # Attach source tag and carry forward data_coverage from Agent 2
    for result, pillar_key in (
        (ec_result,  "EC"),
        (gc_result,  "GC"),
        (rff_result, "RFF"),
        (cr_result,  "CR"),
    ):
        survey_pillar = survey_scores.get(pillar_key) or {}
        orig_score    = survey_pillar.get("score", 0)
        result["source"]        = "survey+docs" if result["score"] != orig_score else "survey"
        result["data_coverage"] = survey_pillar.get("data_coverage", 0.0)
        result.pop("_c3_status", None)
        result.pop("_c4_status", None)

    return {
        "ec":          ec_result,
        "gc":          gc_result,
        "rff":         rff_result,
        "cr":          cr_result,
        "enriched_log": enriched_log,
    }
