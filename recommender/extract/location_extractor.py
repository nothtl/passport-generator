"""Location extraction from resume text — pulls city/state for location-aware recs."""

from __future__ import annotations

import re

_NYC_BOROUGHS = [
    "new york", "brooklyn", "bronx", "queens", "manhattan", "staten island",
]

_STATE_MAP: dict[str, str] = {
    "ny": "NY", "new york": "NY", "nj": "NJ", "ct": "CT", "pa": "PA",
    "ca": "CA", "tx": "TX", "fl": "FL", "il": "IL", "ma": "MA",
    "ga": "GA", "nc": "NC", "oh": "OH", "mi": "MI", "va": "VA",
    "wa": "WA", "az": "AZ", "co": "CO", "md": "MD", "dc": "DC",
    "mn": "MN", "wi": "WI", "or": "OR", "mo": "MO", "tn": "TN",
    "in": "IN", "al": "AL", "sc": "SC", "ky": "KY", "la": "LA",
}


def extract_location(resume_text: str) -> dict[str, str]:
    """Extract city and state from resume text.

    Returns {'city': ..., 'state': ...}. Both empty on failure.
    """
    if not resume_text:
        return {"city": "", "state": ""}

    text = resume_text[:1000].lower()
    city = ""
    state = ""

    # NYC boroughs first
    for borough in _NYC_BOROUGHS:
        if borough in text:
            city = borough.title()
            state = "NY"
            break

    # Explicit state patterns
    if not state:
        for pattern, abbr in sorted(_STATE_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(rf"\b{re.escape(pattern)}\b", text):
                state = abbr
                break

    # Try to find city near state mention
    if state and not city:
        m = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+)*),\s*" + state, text.upper(), re.IGNORECASE)
        if m:
            city = m.group(1)

    return {"city": city, "state": state}
