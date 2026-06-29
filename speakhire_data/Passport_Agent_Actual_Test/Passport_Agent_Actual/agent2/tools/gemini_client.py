import os
import re
import json
import time
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash-lite"

_call_count = 0


def get_call_count():
    return _call_count


def reset_call_count():
    global _call_count
    _call_count = 0


def call_gemini(prompt, retries=3):
    global _call_count
    if not GEMINI_API_KEY:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    delays = [2, 4, 8]
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                _call_count += 1
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            elif r.status_code == 429:
                time.sleep(delays[min(attempt, len(delays) - 1)])
        except Exception:
            time.sleep(delays[min(attempt, len(delays) - 1)])
    return None


def parse_json_resp(text):
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None
