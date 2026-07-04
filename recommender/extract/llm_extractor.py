"""DeepSeek-powered skill extraction — maps informal student language to O*NET controlled vocabulary.

Ported from the zip pipeline's Gemini prompt. Uses DeepSeek v4 flash instead of Gemini
for 250x cost savings ($0.0005 vs $0.0125 per call).
"""
from __future__ import annotations

import json, os, urllib.request

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(HERE, "..", "data", "onet_importance.json")

_vocab_cache = None


def _load_vocab() -> list[str]:
    global _vocab_cache
    if _vocab_cache is None:
        with open(_VOCAB_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _vocab_cache = data.get("skill_names", [])
    return _vocab_cache


PROMPT = """You read a young person's resume and LinkedIn text and identify the workplace SKILLS
their experience demonstrates. These are NYC high-schoolers and recent grads who describe
themselves informally ("ran our church instagram", "helped at the food pantry", "cashier at a
milk tea store"). Infer the IMPLICIT skill behind each activity -- running an Instagram shows
communication and (some) marketing; cashiering shows customer service, math, and reliability.

You MUST map every skill to one of the allowed O*NET skill names below. Do NOT invent skill names
and do NOT output a name that is not in this list verbatim:

{vocab}

For each skill the evidence clearly supports, output:
- onet_skill: the exact name from the allowed list
- evidence: the short phrase from their text that demonstrates it
- proficiency: one of emerging / developing / proficient, judged from how central and sustained
  the experience is (a one-off mention = emerging; a year-long job using it = proficient)

Only include skills with real evidence. Better to omit a skill than to guess.
Return EXACTLY this JSON:
{{"skills": [{{"onet_skill": "...", "evidence": "...", "proficiency": "proficient|developing|emerging"}}]}}"""


def extract_skills_deepseek(text: str, max_chars: int = 12000) -> dict:
    """Extract O*NET skills using DeepSeek with the zip's proven prompt.

    Returns {"skills": [names], "detected": [{onet_skill, evidence, proficiency}]}.
    """
    if not DEEPSEEK_KEY or not text.strip():
        return {"skills": [], "detected": []}

    vocab = _load_vocab()
    if not vocab:
        return {"skills": [], "detected": []}

    prompt = PROMPT.format(vocab="\n".join(f"- {v}" for v in sorted(vocab)))
    user = f"{prompt}\n\n=== STUDENT TEXT ===\n{text.strip()[:max_chars]}"

    payload = {
        "model": MODEL, "temperature": 0.2, "max_tokens": 1200,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": user}],
    }

    try:
        req = urllib.request.Request(
            DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        data = json.loads(content)

        # Intersect against vocabulary to prevent hallucinations
        vocab_lower = {v.lower(): v for v in vocab}
        detected = []
        for item in data.get("skills", []):
            name = vocab_lower.get(str(item.get("onet_skill", "")).strip().lower())
            if not name or name in {d["onet_skill"] for d in detected}:
                continue
            prof = str(item.get("proficiency", "")).strip().lower()
            detected.append({
                "onet_skill": name,
                "evidence": str(item.get("evidence", "")).strip(),
                "proficiency": prof if prof in ("emerging", "developing", "proficient") else "developing",
            })

        return {
            "skills": [d["onet_skill"] for d in detected],
            "detected": detected,
        }
    except Exception:
        return {"skills": [], "detected": []}
