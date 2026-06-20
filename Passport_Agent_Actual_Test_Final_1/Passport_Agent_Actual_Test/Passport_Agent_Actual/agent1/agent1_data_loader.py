"""
Agent 1 -- Data Loader
Finds any student in Full_speakhire_data.zip and extracts all raw fields.

Usage:
  python agent1_data_loader.py --student "Student Name" --zip path/to/data.zip
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

from tools.zip_tools import list_files, load_file
from tools.student_finder import find_in_dataframe, get_email
from tools.field_mapper import load_registry, extract_fields

OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "config", "field_registry.json")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Search order: richest / most complete files first
FILE_PRIORITY = [
    "Interns_20260512T143927-0400.csv",
    "Mentee Data Evaluation .xlsx",
    "FULL SpeakHire Database 2026.xlsx",
    "SPEAKHIRE FY Internship Round Exit Survey.csv",
    "SPEAKHIRE Foundational Year Exit Survey.csv",
    "SPEAKHIRE Leadership Course Survey.csv",
    "SPEAKHIRE Seminars Survey.csv",
    "Career Pathways Champions_20260512T144018-0400.csv",
    "Hub Interns (2).csv",
]

# Session feedback is handled separately (multi-row aggregation)
SESSION_FEEDBACK_FILE = "Internship Session Feedback (Responses).xlsx"


def _find_session_col(df: pd.DataFrame, keywords: list) -> str:
    """Find a column by checking if its lowercase name contains all keywords."""
    for c in df.columns:
        cl = str(c).lower()
        if all(kw in cl for kw in keywords):
            return c
    return None


def _get_cpc_emails(zip_path: str, all_files: list,
                    connected_champions_str: str) -> set:
    """
    Given a comma-separated string of champion names, look up their emails
    from the Champions sheet in FULL SpeakHire Database 2026.xlsx.
    """
    if not connected_champions_str:
        return set()

    champ_names = [n.strip() for n in connected_champions_str.split(',') if n.strip()]
    db_path = next(
        (f for f in all_files if 'FULL SpeakHire Database 2026' in f), None
    )
    if not db_path:
        return set()

    try:
        data = load_file(zip_path, db_path)
        if not isinstance(data, dict) or 'Champions' not in data:
            return set()
        champ_df = data['Champions']
    except Exception as e:
        print(f"  Warning: could not load Champions sheet: {e}")
        return set()

    email_col = next(
        (c for c in champ_df.columns if str(c).lower().strip() == 'email address'), None
    )
    name_col = next(
        (c for c in champ_df.columns if str(c).lower().strip() == 'full name'), None
    )
    if not email_col or not name_col:
        return set()

    emails = set()
    for name in champ_names:
        mask = (
            champ_df[name_col].fillna('').astype(str)
            .str.lower().str.strip() == name.lower()
        )
        if mask.any():
            e = champ_df[mask].iloc[0][email_col]
            if pd.notna(e) and '@' in str(e):
                emails.add(str(e).strip().lower())

    return emails


def aggregate_session_feedback(zip_path: str,
                                all_files: list,
                                student_email: str,
                                student_rows: dict) -> dict:
    """
    Load Internship Session Feedback and aggregate CPC-submitted rows for
    this student.

    Session feedback is submitted by CPCs (Mentors), not interns directly.
    Strategy:
      1. Read 'Connected Champions' from the student's Interns-sheet row
      2. Look up those champions' emails from the Champions sheet
      3. Filter session feedback to rows where Email Address matches a CPC email
         AND the submitter role is Mentor/Champion
      4. Aggregate text columns

    Returns dict keyed by canonical field names.
    """
    # Step 1: get Connected Champions string from already-loaded student rows
    connected_champions_str = None
    for key, row in student_rows.items():
        for col in ['Connected Champions', 'Connected_Champions', 'Connected Champion']:
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                connected_champions_str = str(row[col]).strip()
                break
        if connected_champions_str:
            break

    if connected_champions_str:
        print(f"  Connected Champions: {connected_champions_str}")
    else:
        print("  Connected Champions: not found")

    # Step 2: resolve CPC emails
    cpc_emails = _get_cpc_emails(zip_path, all_files, connected_champions_str)
    print(f"  CPC emails resolved: {cpc_emails if cpc_emails else 'none'}")

    # Step 3: load session feedback
    feedback_path = next(
        (f for f in all_files if SESSION_FEEDBACK_FILE in f), None
    )
    if feedback_path is None:
        print("  Session feedback file not found in zip.")
        return {}

    try:
        data = load_file(zip_path, feedback_path)
        if isinstance(data, dict):
            df = next(iter(data.values()))
        else:
            df = data
    except Exception as e:
        print(f"  Warning: could not load session feedback: {e}")
        return {}

    email_col = next(
        (c for c in df.columns if str(c).lower().strip() == 'email address'), None
    )
    if email_col is None:
        print("  Warning: no email column found in session feedback.")
        return {}

    # Filter to Mentor/CPC rows only
    role_col = next((c for c in df.columns if str(c).strip() == 'I am a '), None)
    if role_col:
        mentor_mask = ~df[role_col].fillna('').astype(str).str.lower().str.contains('mentee|intern')
        df = df[mentor_mask].copy()

    # Match by CPC emails (primary) or intern email fallback
    if cpc_emails:
        student_mask = (
            df[email_col].fillna('').astype(str)
            .str.lower().str.strip().isin(cpc_emails)
        )
    elif student_email:
        # Fallback: try intern's own email (covers cases where intern submitted)
        student_mask = (
            df[email_col].fillna('').astype(str)
            .str.lower().str.strip() == student_email.lower().strip()
        )
    else:
        return {}

    student_sessions = df[student_mask].copy()
    session_count = len(student_sessions)
    print(f"  Session feedback rows found: {session_count}")

    if session_count == 0:
        return {
            'CPC Session Count': 0,
            'CPC All Session Text': None,
            'CPC Resume Text': None,
            'CPC What skill did you cover': None,
            'CPC What component skill did you cover': None,
            'CPC Discussion topics': None,
            'CPC Why': None,
            'CPC Resume additions': None,
            'CPC Anything else': None,
        }

    def _agg_col(col_name):
        if col_name and col_name in student_sessions.columns:
            parts = [
                str(v).strip()
                for v in student_sessions[col_name]
                if pd.notna(v) and str(v).strip() not in ('', 'nan')
            ]
            return ' | '.join(parts) if parts else None
        return None

    def _find_col_prefix(prefix: str):
        """Find column whose name starts with prefix (case-insensitive)."""
        for c in student_sessions.columns:
            if str(c).lower().startswith(prefix.lower()):
                return c
        return None

    skill_col     = next((c for c in df.columns if str(c).strip() == 'What skill did you cover?'), None)
    comp_col      = next((c for c in df.columns if str(c).strip() == 'What component skill did you cover?'), None)
    disc_col      = _find_col_prefix('What were some things you discussed')
    why_col       = next((c for c in df.columns if str(c).strip() == 'Why?'), None)
    resume_col    = next((c for c in df.columns if 'We added the following to the Intern resume' in str(c)), None)
    anything_col  = _find_col_prefix('Anything else you')

    skill_text    = _agg_col(skill_col)
    comp_text     = _agg_col(comp_col)
    disc_text     = _agg_col(disc_col)
    why_text      = _agg_col(why_col)
    resume_text   = _agg_col(resume_col)
    anything_text = _agg_col(anything_col)

    # Build combined session text (C3) and resume text (C4)
    all_parts = [t for t in [skill_text, comp_text, disc_text, why_text] if t]
    all_session_text = ' | '.join(all_parts) if all_parts else None

    return {
        'CPC Session Count':                    session_count,
        'CPC All Session Text':                 all_session_text,
        'CPC Resume Text':                      resume_text,
        'CPC What skill did you cover':         skill_text,
        'CPC What component skill did you cover': comp_text,
        'CPC Discussion topics':                disc_text,
        'CPC Why':                              why_text,
        'CPC Resume additions':                 resume_text,
        'CPC Anything else':                    anything_text,
    }


def run(student_name: str, zip_path: str) -> dict:
    print("Agent 1: Data Loader")
    print(f"Student: {student_name}")

    all_files = list_files(zip_path)
    print(f"Zip contains {len(all_files)} files.")

    # --- Step 1: Find student rows across all priority files ---
    student_rows: dict = {}
    student_email: str = None

    for priority_file in FILE_PRIORITY:
        matched_path = next(
            (f for f in all_files if priority_file in f), None
        )
        if not matched_path:
            continue

        try:
            data = load_file(zip_path, matched_path)
            sheets = data if isinstance(data, dict) else {'_default': data}

            for sheet_name, df in sheets.items():
                row = find_in_dataframe(df, student_name, student_email)
                if row is not None:
                    key = (
                        priority_file
                        if sheet_name == '_default'
                        else f"{priority_file}::{sheet_name}"
                    )
                    student_rows[key] = row
                    print(f"  Found in: {key}")

                    if student_email is None:
                        student_email = get_email(row)

        except Exception as e:
            print(f"  Warning: could not load {priority_file}: {e}")

    if not student_rows:
        print(f"ERROR: '{student_name}' not found in any file.")
        sys.exit(1)

    print(f"Found in {len(student_rows)} source(s). Email: {student_email}")

    # --- Step 2: Aggregate session feedback (multi-row) ---
    aggregated_session = aggregate_session_feedback(
        zip_path, all_files, student_email, student_rows
    )

    # --- Step 3: Extract all registered fields ---
    registry = load_registry(REGISTRY_PATH)
    fields   = extract_fields(student_rows, registry, aggregated_session)

    found   = sum(1 for f in fields.values() if f['status'] == 'found')
    missing = sum(1 for f in fields.values() if f['status'] == 'missing')
    print(f"Fields found: {found} | Missing: {missing}")

    # --- Step 4: Assemble and write output ---
    output = {
        'student_name':  student_name,
        'email':         student_email,
        'found_in':      list(student_rows.keys()),
        'fields':        fields,
        'field_summary': {
            'total':   len(fields),
            'found':   found,
            'missing': missing,
        },
    }

    slug = re.sub(r'[^a-z0-9]+', '_', student_name.lower()).strip('_')
    out_path = os.path.join(OUTPUTS_DIR, f"{slug}_raw_data.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print(f"Output: {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Agent 1: Data Loader')
    parser.add_argument('--student', required=True, help='Student full name')
    parser.add_argument('--zip',     required=True, help='Path to zip file')
    args = parser.parse_args()
    run(args.student, args.zip)


if __name__ == '__main__':
    main()
