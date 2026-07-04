import pandas as pd
import re


def norm_name(s: str) -> str:
    """
    Normalise any student name string for comparison.
    Handles: \\xa0 non-breaking spaces, double spaces,
    'AÂ' encoding artifacts, leading/trailing whitespace.
    """
    s = str(s).strip()
    s = s.replace('\xa0', ' ').replace('\xc2', '').replace('Â', '')
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def find_in_dataframe(df: pd.DataFrame,
                      student_name: str,
                      email: str = None) -> pd.Series:
    """
    Search any DataFrame for a student.
    Returns matching row as Series or None.

    Strategy (in order):
    1. Find name column (full name / intern name / student name / contact name)
    2. Exact normalised match
    3. First-name + last-name prefix match (handles suffix variations)
    4. Email match if email provided
    """
    target = norm_name(student_name)
    name_col = _find_name_col(df)

    if name_col:
        normed = df[name_col].fillna('').astype(str).apply(norm_name)

        mask = normed == target
        if mask.any():
            return df[mask].iloc[0]

        parts = target.split()
        if len(parts) >= 2:
            prefix = parts[0] + ' ' + parts[1]
            mask2 = normed.str.startswith(prefix)
            if mask2.sum() == 1:
                return df[mask2].iloc[0]

    if email:
        email_col = _find_email_col(df)
        if email_col:
            mask = (df[email_col].fillna('')
                                 .astype(str)
                                 .str.lower()
                                 .str.strip() == email.lower().strip())
            if mask.any():
                return df[mask].iloc[0]

    return None


def get_email(row: pd.Series) -> str:
    """Extract primary email from a student row."""
    for col in row.index:
        cl = str(col).lower()
        if 'email' in cl and 'guardian' not in cl and 'parent' not in cl:
            val = row[col]
            if pd.notna(val) and '@' in str(val):
                return str(val).strip().lower()
    return None


def _find_name_col(df: pd.DataFrame) -> str:
    priority = [
        'full name', 'intern name (autofills)', 'student name',
        'contact name (autofills)', 'name'
    ]
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    for p in priority:
        if p in cols_lower:
            return cols_lower[p]
    for c in df.columns:
        cl = str(c).lower()
        if 'full name' in cl or ('intern name' in cl and 'auto' in cl):
            return c
    return None


def _find_email_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ('email', 'email address'):
            return c
    for c in df.columns:
        cl = str(c).lower()
        if 'email' in cl and 'guardian' not in cl and 'parent' not in cl:
            return c
    return None
