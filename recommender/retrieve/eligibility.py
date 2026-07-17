"""Youth eligibility screening — LLM judges each job for age-appropriateness (16-20).

Cached per job title. Swaps OpenAI for DeepSeek.
"""
from __future__ import annotations

import json, logging, os, time, urllib.error, urllib.request
from recommender.utils import retry

from recommender.config import get_llm, get_eligibility
_LLM = get_llm()
_ELIG = get_eligibility()
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "data", "eligibility_cache")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = _LLM.deepseek_url
MODEL = _LLM.deepseek_model
BATCH_SIZE = _ELIG.batch_size

PROMPT = """You screen roles for a youth career-readiness program. Candidates are about 16-20 years old
(high-schoolers and recent grads). For each numbered item decide if it is ELIGIBLE for them.

Mark INELIGIBLE only when the role itself legally requires age 21+ or is clearly unsuitable for minors /
young adults:
- serving or making alcoholic drinks (bartender, barback, sommelier)
- gambling / casino gaming-floor roles (dealer, cage cashier, slot/sportsbook/pit work)
- cannabis / dispensary, tobacco / vape sales
- firearms or armed security
- adult / 18+ / nightlife-host content

A role is ELIGIBLE if a 16-20 year old could lawfully hold it, even at a venue that also serves alcohol
or is inside a casino, as long as THIS role does not require being 21+ (e.g. busser, host, barista,
food runner, dishwasher, retail cashier, line cook, server where permitted). When genuinely unsure,
mark it ELIGIBLE.

Return JSON: {"results": [{"index": <int>, "eligible": <bool>, "reason": "<short>"}]} for every item."""


def _load_cache() -> dict[str, dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {}
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(CACHE_DIR, fname), encoding="utf-8") as f:
                cache.update(json.load(f))
    return cache


def _save_batch(batch: dict[str, dict]) -> None:
    import hashlib
    key = hashlib.md5(str(sorted(batch.keys())).encode()).hexdigest()[:12]
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2)


@retry(max_attempts=3, delay=1.0, backoff=2.0)
def _call_llm(items: list[tuple[int, str, str]]) -> dict[int, dict]:
    """Call DeepSeek to judge eligibility for a batch of job titles."""
    numbered = "\n".join(f"{i}. {title} @ {company}" for i, title, company in items)
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": numbered},
        ],
    }
    try:
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        data = json.loads(content)
        results = {}
        for r in data.get("results", []):
            results[r["index"]] = {"eligible": r["eligible"], "reason": r.get("reason", "")}
        return results
    except Exception as e:
        logger.warning(f"Eligibility LLM failed: {e}")
        return {i: {"eligible": True, "reason": "could not screen"} for i, _, _ in items}


def screen_jobs(jobs: list[dict]) -> list[dict]:
    """Add eligible/eligibility_reason to each job. Cached per title."""
    if not DEEPSEEK_KEY:
        for j in jobs:
            j["eligible"] = True
            j["eligibility_reason"] = ""
        return jobs

    cache = _load_cache()
    uncached = []
    for i, job in enumerate(jobs):
        title = str(job.get("title", "")).strip()
        company = str(job.get("company", "")).strip()
        cache_key = f"{title}|||{company}" if company else title
        if cache_key in cache:
            job["eligible"] = cache[cache_key]["eligible"]
            job["eligibility_reason"] = cache[cache_key]["reason"]
        else:
            uncached.append((i, title, company))

    if uncached:
        new_batch = {}
        for batch_start in range(0, len(uncached), BATCH_SIZE):
            batch = uncached[batch_start:batch_start + BATCH_SIZE]
            results = _call_llm(batch)
            for i, title, company in batch:
                cache_key = f"{title}|||{company}" if company else title
                r = results.get(i, {"eligible": True, "reason": "default"})
                new_batch[cache_key] = {"eligible": r["eligible"], "reason": r.get("reason", r.get("eligibility_reason", ""))}
            time.sleep(0.3)
        _save_batch(new_batch)
        cache.update(new_batch)
        # Apply to jobs
        for i, job in enumerate(jobs):
            title = str(job.get("title", "")).strip()
            company = str(job.get("company", "")).strip()
            cache_key = f"{title}|||{company}" if company else title
            if cache_key in cache:
                job["eligible"] = cache[cache_key]["eligible"]
                job["eligibility_reason"] = cache[cache_key]["reason"]

    return jobs


def filter_ineligible(jobs: list[dict]) -> list[dict]:
    """Remove ineligible jobs from the list."""
    screened = screen_jobs(jobs)
    return [j for j in screened if j.get("eligible", True)]
