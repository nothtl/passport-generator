import sys, os, requests
sys.path.insert(0, 'agent4')

from tools.gemini_client import GEMINI_API_KEY, GEMINI_MODEL
print("Key:", GEMINI_API_KEY[:15] if GEMINI_API_KEY else "NONE")
print("Model:", GEMINI_MODEL)

url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
       f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
payload = {"contents": [{"parts": [{"text": "Say: HELLO"}]}]}
r = requests.post(url, json=payload, timeout=30)
print("Status:", r.status_code)
print("Response:", r.text[:300])
