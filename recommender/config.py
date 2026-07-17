"""Central configuration loader — single source of truth for all recommender settings.

Usage:
    from recommender.config import get_config
    cfg = get_config()
    junk = cfg.resume.junk_skills
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _HERE / "config.yaml"
_config: Config | None = None
_lock = threading.Lock()


class _DotDict(dict):
    """Dict with attribute-style access, recursively."""
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, dict) and not isinstance(val, _DotDict):
            val = _DotDict(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class Config:
    """Typed configuration wrapper loaded from config.yaml."""

    def __init__(self, raw: dict) -> None:
        self._raw = _DotDict(raw)
        self._ensure_paths()

    def _ensure_paths(self) -> None:
        """Resolve relative paths in config."""
        # Ensure cache/llm dirs are absolute
        project_dir = _HERE.parent  # "passport generator/"
        self.project_dir = str(project_dir)

    # ── Read-only convenience properties ──────────────────────────

    @property
    def pipeline(self) -> dict:
        return self._raw.pipeline

    @property
    def functions(self) -> dict:
        return self._raw.functions

    @property
    def resume(self) -> dict:
        return self._raw.resume

    @property
    def skill_extraction(self) -> dict:
        return self._raw.skill_extraction

    @property
    def student_intent(self) -> dict:
        return self._raw.student_intent

    @property
    def cross_function(self) -> dict:
        return self._raw.cross_function

    @property
    def function_keywords(self) -> dict:
        return self._raw.function_keywords

    @property
    def subdomains(self) -> dict:
        return self._raw.subdomains

    @property
    def ensemble(self) -> dict:
        return self._raw.ensemble

    @property
    def ranking(self) -> dict:
        return self._raw.ranking

    @property
    def llm(self) -> dict:
        return self._raw.llm

    @property
    def eligibility(self) -> dict:
        return self._raw.eligibility

    @property
    def student_lifecycle(self) -> dict:
        return self._raw.student_lifecycle

    @property
    def college(self) -> dict:
        return self._raw.college

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)


def _replace_env_vars(value: Any) -> Any:
    """Replace ${VAR} patterns in config values with environment variables."""
    if isinstance(value, str):
        def _replacer(m: re.Match) -> str:
            return os.getenv(m.group(1), "")
        return re.sub(r'\$\{(\w+)\}', _replacer, value)
    if isinstance(value, dict):
        return {k: _replace_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_env_vars(v) for v in value]
    return value


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and cache configuration from YAML file.

    Thread-safe — only loads once per process.
    """
    global _config
    if _config is not None:
        return _config

    with _lock:
        if _config is not None:
            return _config

        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Resolve environment variables
        raw = _replace_env_vars(raw)

        _config = Config(raw or {})
        return _config


def get_config() -> Config:
    """Get the cached config, loading defaults if not yet loaded."""
    if _config is not None:
        return _config
    return load_config()


def reload_config(config_path: str | Path | None = None) -> Config:
    """Force reload configuration (e.g., after config file changes)."""
    global _config
    with _lock:
        _config = None
    return load_config(config_path)


# ── Convenience accessors for common lookups ──────────────────────

def get_pipeline() -> _DotDict:
    return _DotDict(get_config().pipeline)


def get_functions() -> _DotDict:
    return _DotDict(get_config().functions)


def get_resume() -> _DotDict:
    return _DotDict(get_config().resume)


def get_skill_extraction() -> _DotDict:
    return _DotDict(get_config().skill_extraction)


def get_student_intent() -> _DotDict:
    return _DotDict(get_config().student_intent)


def get_cross_function() -> _DotDict:
    return _DotDict(get_config().cross_function)


def get_function_keywords() -> _DotDict:
    return _DotDict(get_config().function_keywords)


def get_subdomains() -> _DotDict:
    return _DotDict(get_config().subdomains)


def get_ensemble() -> _DotDict:
    return _DotDict(get_config().ensemble)


def get_ranking() -> _DotDict:
    return _DotDict(get_config().ranking)


def get_llm() -> _DotDict:
    return _DotDict(get_config().llm)


def get_eligibility() -> _DotDict:
    return _DotDict(get_config().eligibility)


def get_student_lifecycle() -> _DotDict:
    return _DotDict(get_config().student_lifecycle)


def get_college() -> _DotDict:
    return _DotDict(get_config().college)