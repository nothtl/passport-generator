import os
import re

STUDENT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'student_data')


def _slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _name_similarity(a: str, b: str) -> float:
    ta = set(_slugify(a).split('_'))
    tb = set(_slugify(b).split('_'))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def find_student_folder(student_name: str) -> str | None:
    """Return path to best-matching student folder, or None if not found."""
    if not os.path.isdir(STUDENT_DATA_DIR):
        print(f"[Agent3] student_data directory not found: {STUDENT_DATA_DIR}")
        return None
    best_path, best_score = None, 0.0
    for entry in os.listdir(STUDENT_DATA_DIR):
        full = os.path.join(STUDENT_DATA_DIR, entry)
        if not os.path.isdir(full):
            continue
        score = _name_similarity(student_name, entry)
        if score > best_score:
            best_score, best_path = score, full
    if best_score > 0.4:
        print(f"[Agent3] Matched folder '{os.path.basename(best_path)}' (score={best_score:.2f}) for '{student_name}'")
        return best_path
    print(f"[Agent3] No folder match found for '{student_name}' (best score={best_score:.2f})")
    return None


def find_all_resume_pdfs(folder: str) -> list[str]:
    """Return all PDF files in folder, sorted alphabetically."""
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith('.pdf')
    ]


def find_resume_pdf(folder: str) -> str | None:
    pdfs = find_all_resume_pdfs(folder)
    return pdfs[0] if pdfs else None


def find_linkedin_md(folder: str) -> str | None:
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith('.md'):
            return os.path.join(folder, f)
    return None
