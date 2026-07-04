import json
import pandas as pd


def load_registry(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def extract_fields(student_rows: dict,
                   registry: dict,
                   aggregated_session: dict = None) -> dict:
    """
    For each canonical field in the registry, search all available
    student rows (keyed by source filename) for a real value.

    student_rows:       {filename: pd.Series of the student's row}
    registry:           loaded field_registry.json
    aggregated_session: pre-built dict of session-aggregated values
                        (populated by aggregate_session_feedback in agent1_data_loader.py)

    Returns:
    {
      "canonical field name": {
        "value":  <value or None>,
        "source": <filename or None>,
        "status": "found" | "missing",
        "pillar": "EC" | "GC" | "RFF" | "CR"
      }
    }
    """
    result = {}

    for pillar, fields in registry.items():
        for field_key, field_def in fields.items():
            canonical = field_def['canonical']
            variants  = field_def.get('column_variants', [canonical])
            search_in = field_def.get('search_in', [])
            multi_row = field_def.get('multi_row', False)

            # Multi-row fields are pre-populated from session feedback aggregation
            if multi_row:
                agg = aggregated_session or {}
                val = agg.get(canonical)
                result[canonical] = {
                    'value':  val if val not in (None, '') else None,
                    'source': 'Internship Session Feedback (Responses).xlsx' if val else None,
                    'status': 'found' if val not in (None, '') else 'missing',
                    'pillar': pillar,
                }
                continue

            found_val    = None
            found_source = None

            for preferred_file in search_in:
                matching_key = next(
                    (k for k in student_rows if preferred_file in k), None
                )
                if matching_key is None:
                    continue

                row = student_rows[matching_key]
                for variant in variants:
                    if variant in row.index:
                        val = row[variant]
                        if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                            found_val    = val
                            found_source = matching_key
                            break
                if found_val is not None:
                    break

            result[canonical] = {
                'value':  found_val if found_val is not None else None,
                'source': found_source,
                'status': 'found' if found_val is not None else 'missing',
                'pillar': pillar,
            }

    return result
