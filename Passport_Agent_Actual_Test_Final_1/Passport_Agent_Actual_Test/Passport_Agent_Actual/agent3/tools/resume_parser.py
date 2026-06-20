"""
Resume PDF parser using pdfminer.six.
Extracts raw text, then uses Gemini to identify sections dynamically.
"""

import json
import re

try:
    from pdfminer.high_level import extract_text as _pdf_extract_text
    _PDFMINER_OK = True
except ImportError:
    _PDFMINER_OK = False

from .gemini_client import call_gemini, GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Contact-info regex
# ---------------------------------------------------------------------------

_EMAIL_RE    = re.compile(r'[\w.+\-]+@[\w.\-]+\.\w{2,}')
_PHONE_RE    = re.compile(r'(\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}')
_GITHUB_RE   = re.compile(r'github\.com/([a-zA-Z0-9_\-]+)', re.IGNORECASE)
_LINKEDIN_RE = re.compile(r'linkedin\.com/in/([a-zA-Z0-9_\-]+)', re.IGNORECASE)


def _extract_contact(text: str) -> dict:
    top = '\n'.join(text.splitlines()[:30])
    email = phone = github_url = linkedin_url = None

    m = _EMAIL_RE.search(top)
    if m:
        email = m.group(0)

    m = _PHONE_RE.search(top)
    if m:
        phone = m.group(0).strip()

    m = _GITHUB_RE.search(top)
    if m:
        github_url = f"github.com/{m.group(1)}"

    m = _LINKEDIN_RE.search(top)
    if m:
        linkedin_url = f"linkedin.com/in/{m.group(1)}"

    return {"email": email, "phone": phone,
            "github_url": github_url, "linkedin_url": linkedin_url}


# ---------------------------------------------------------------------------
# Gemini-based section extractor
# ---------------------------------------------------------------------------

_SECTION_PROMPT = """You are extracting structured sections from a student resume.

RESUME TEXT:
{raw_text}

Return ONLY this exact JSON. Use "" for any section not present in the resume.
{{
  "summary": "<all text from Summary / Objective / Profile / About section>",
  "experience": "<all text from Experience / Work / Clinical / Research / Internship / Employment sections>",
  "education": "<all text from Education section>",
  "skills": "<all text from Skills / Competencies / Qualifications section>",
  "projects": "<all text from Projects section>",
  "certifications": "<all text from Certifications / Licenses section>",
  "leadership": "<all text from Leadership / Volunteer / Community / Extracurricular / Activities / Campus Engagement sections>"
}}

Rules:
- Copy the original text verbatim — do not summarize or paraphrase
- experience includes: jobs, internships, research roles, clinical roles, work-study, part-time work
- leadership includes: volunteer work, community service, clubs, campus activities, civic engagement
- If a section header does not match standard names, map it to the closest canonical key above
- If the same content could belong to multiple keys, put it under the most specific one
"""


def _gemini_extract_sections(raw_text: str) -> dict:
    """Use Gemini to extract sections from resume text. Returns {} on failure."""
    if not GEMINI_API_KEY or not raw_text or len(raw_text.strip()) < 100:
        return {}
    prompt = _SECTION_PROMPT.replace("{raw_text}", raw_text[:4000])
    resp = call_gemini(prompt)
    if not resp:
        print("[Agent3] Warning: Gemini section extraction failed — Agent 4 will retry")
        return {}
    resp = re.sub(r'```(?:json)?\s*', '', resp).replace('```', '').strip()
    try:
        data = json.loads(resp)
        return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}
    except Exception:
        print("[Agent3] Warning: Could not parse Gemini section response — Agent 4 will retry")
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_resume(pdf_path: str) -> dict:
    if not _PDFMINER_OK:
        print("[Agent3] pdfminer.six not installed — cannot parse PDF")
        return {"raw_text": "", "sections": {}}

    try:
        raw_text = _pdf_extract_text(pdf_path)
    except Exception as e:
        print(f"[Agent3] Warning: could not parse PDF '{pdf_path}': {e}")
        return {"raw_text": "", "sections": {}}

    if not raw_text or not raw_text.strip():
        print(f"[Agent3] Warning: PDF produced no text (may be image-only): '{pdf_path}'")
        return {"raw_text": "", "sections": {}}

    sections = _gemini_extract_sections(raw_text)
    sections["contact"] = _extract_contact(raw_text)

    return {"raw_text": raw_text, "sections": sections}
