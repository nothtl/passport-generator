"""One-time script: generate 16 synthetic entry-level resumes for HYRE-lite.

HYRE (Hypothetical Resume) bridges the vocabulary gap between how students
write resumes and how employers write job descriptions. For each function,
we generate a realistic entry-level resume that mentions the skills, tools,
and experiences typical for that career path.

Cost: 16 LLM calls × ~$0.001 = ~$0.016 total (one-time).
"""
import json, os, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "hyre_resumes.json")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

FUNCTIONS = [
    "technology", "healthcare", "education", "finance", "sales",
    "arts-media", "design", "social-service", "ops", "support",
    "administrative", "legal", "engineering", "food-service",
    "hospitality", "protective-service",
]

PROMPT = """Write a realistic entry-level resume for a {function} candidate with 0-2 years of experience.
Include: 2-3 specific technical or domain skills with tool names, 1-2 relevant projects or internships,
and 1 education entry. Use natural resume language (bullet points, action verbs).
Keep it under 200 words. Write ONLY the resume text, no explanations."""


def generate_one(function: str) -> str:
    payload = {
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 400,
        "messages": [
            {"role": "user", "content": PROMPT.format(function=function)},
        ],
    }
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def main():
    if os.path.exists(OUT_PATH):
        print(f"Already exists: {OUT_PATH}")
        with open(OUT_PATH) as f:
            existing = json.load(f)
        print(f"  {len(existing)} resumes cached")
        return

    if not DEEPSEEK_KEY:
        print("No DEEPSEEK_API_KEY — writing empty file")
        with open(OUT_PATH, "w") as f:
            json.dump({}, f)
        return

    resumes = {}
    for func in FUNCTIONS:
        print(f"  {func}...", end=" ", flush=True)
        try:
            text = generate_one(func)
            resumes[func] = text
            print(f"OK ({len(text)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")
            resumes[func] = f"Entry-level {func} professional with relevant skills and training."
        time.sleep(0.5)  # rate limit

    with open(OUT_PATH, "w") as f:
        json.dump(resumes, f, indent=2)
    print(f"Saved {len(resumes)} resumes to {OUT_PATH}")


if __name__ == "__main__":
    main()
