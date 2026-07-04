from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


class LLMCache:
    def __init__(self, cache_dir: str | None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_key(self, stage: str, payload: dict[str, Any], model: str, version: str = "v1") -> str:
        encoded = json.dumps(
            {
                "stage": stage,
                "model": model,
                "version": version,
                "payload": _jsonable(payload),
            },
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
