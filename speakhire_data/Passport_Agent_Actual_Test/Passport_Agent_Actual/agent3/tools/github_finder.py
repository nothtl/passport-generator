import re

_EXCLUDED = {"login", "signup", "about", "features", "pricing", "orgs", "marketplace",
             "explore", "topics", "trending", "collections", "events", "sponsors"}


def find_github_username(resume_text: str = "", linkedin_text: str = "") -> str | None:
    """Search resume then linkedin text for github.com/username. Return username or None."""
    combined = (resume_text or "") + "\n" + (linkedin_text or "")
    matches = re.findall(r'github\.com/([a-zA-Z0-9_-]+)', combined)
    for m in matches:
        if m.lower() not in _EXCLUDED:
            return m
    return None
