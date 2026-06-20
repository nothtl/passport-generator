"""
Gemini-based resume section extractor.
Called from agent4 when keyword-based parsing produced sparse results (< 2 rich sections).
Handles any resume format — non-standard headers, clinical, research, arts, etc.
"""
import json
import re

from .gemini_client import call_gemini, GEMINI_API_KEY

_PROMPT = """You are extracting structured sections from a student resume.

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
  "leadership": "<all text from Leadership / Volunteer / Community / Extracurricular / Activities sections>"
}}

Rules:
- Copy the original text verbatim — do not summarize or paraphrase
- experience includes: jobs, internships, research roles, clinical roles, work-study, part-time work
- leadership includes: volunteer work, community service, clubs, campus activities, civic engagement
- If a section header doesn't match standard names exactly, include it under the closest key above
- If the same content could belong to multiple keys, put it under the most specific one
"""


def extract_sections(raw_text: str) -> dict:
    """
    Use Gemini to parse resume raw_text into structured sections.
    Returns a dict of section_name -> text. Empty sections are excluded.
    Returns {} if Gemini is unavailable or text is too short.
    """
    if not GEMINI_API_KEY or not raw_text or len(raw_text.strip()) < 100:
        return {}
    prompt = _PROMPT.replace("{raw_text}", raw_text[:4000])
    resp = call_gemini(prompt)
    if not resp:
        return {}
    resp = re.sub(r'```(?:json)?\s*', '', resp).replace('```', '').strip()
    try:
        data = json.loads(resp)
        return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}
    except Exception:
        return {}
