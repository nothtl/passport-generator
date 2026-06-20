"""
GitHub REST API scraper.
Token from os.environ.get("GITHUB_TOKEN") — unauthenticated fallback (60 req/hr) if unset.
"""

import base64
import os

import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
BASE = "https://api.github.com"
_TIMEOUT = 10


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _get(url: str, params: dict = None):
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[Agent3] GitHub request failed: {url} — {e}")
    return None


def _fetch_readme(username: str, repo_name: str) -> str | None:
    data = _get(f"{BASE}/repos/{username}/{repo_name}/readme")
    if not data:
        return None
    try:
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return content[:1500]
    except Exception:
        return None


def _fetch_languages(username: str, repo_name: str) -> list[str]:
    data = _get(f"{BASE}/repos/{username}/{repo_name}/languages")
    if not data:
        return []
    return list(data.keys())


def scrape_github(username: str) -> dict:
    # ── Profile ──────────────────────────────────────────────────────────────
    profile = _get(f"{BASE}/users/{username}")
    if not profile:
        print(f"[Agent3] GitHub user '{username}' not found or API error")
        return {"username": username, "bio": None, "public_repos": 0,
                "followers": 0, "repos": []}

    # ── Repos ────────────────────────────────────────────────────────────────
    repos_raw = _get(f"{BASE}/users/{username}/repos",
                     params={"sort": "updated", "per_page": 10}) or []

    repos = []
    for repo in repos_raw:
        if repo.get("fork"):
            continue
        name = repo["name"]
        languages = _fetch_languages(username, name)
        readme    = _fetch_readme(username, name)
        repos.append({
            "name":        name,
            "description": repo.get("description"),
            "languages":   languages,
            "stars":       repo.get("stargazers_count", 0),
            "forks":       repo.get("forks_count", 0),
            "readme":      readme,
        })

    print(f"[Agent3] GitHub: @{username} — {len(repos)} non-fork repos scraped")
    return {
        "username":     username,
        "bio":          profile.get("bio"),
        "public_repos": profile.get("public_repos", 0),
        "followers":    profile.get("followers", 0),
        "repos":        repos,
    }
