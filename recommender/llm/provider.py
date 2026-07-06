from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Protocol


class LLMProvider(Protocol):
    def complete_json(self, stage: str, system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
        ...


class StaticLLMProvider:
    def __init__(self, responses: dict[str, dict[str, Any]]):
        self._responses = responses
        self._calls: Counter[str] = Counter()

    def complete_json(self, stage: str, system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
        self._calls[stage] += 1
        return dict(self._responses.get(stage, {}))

    def call_count(self, stage: str) -> int:
        return self._calls.get(stage, 0)


def load_openrouter_api_key(search_roots: list[str | Path] | None = None) -> str:
    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    roots = [Path(root) for root in (search_roots or [])]
    if not roots:
        here = Path(__file__).resolve()
        roots = [here.parents[2], Path.cwd()]

    checked = set()
    for root in roots:
        if root in checked:
            continue
        checked.add(root)
        dotenv_path = root / ".env"
        if dotenv_path.exists():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        api_path = root / "api"
        if api_path.exists():
            candidate = api_path.read_text(encoding="utf-8").strip()
            if candidate.startswith("sk-or-"):
                return candidate
    return ""


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        referer: str | None = None,
        title: str = "speakhire-recommender",
    ):
        self.api_key = (api_key or load_openrouter_api_key()).strip()
        self.base_url = base_url
        self.referer = referer or os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        self.title = title

    def complete_json(self, stage: str, system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OpenRouter API key is not configured")

        payload = {
            "model": model,
            "temperature": 0,
            "top_p": 1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.title,
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)


class DeepSeekProvider:
    """DeepSeek API provider with retry + backoff. OpenAI-compatible endpoint."""

    def __init__(self, api_key: str = "", model: str = "deepseek-chat"):
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def complete_json(self, stage: str, system_prompt: str, user_prompt: str, model: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DeepSeek API key not configured")

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model or self.model,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }

        last_error = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.base_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return json.loads(body["choices"][0]["message"]["content"])
            except Exception as e:
                last_error = e
                if attempt < 2:
                    import time
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek failed after 3 attempts: {last_error}")
