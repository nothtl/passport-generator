from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recommender.llm.cache import LLMCache
from recommender.llm.evidence_judge import EvidenceNormalizerJudge
from recommender.llm.job_judge import JobFitGapJudge
from recommender.llm.provider import OpenRouterProvider
from recommender.llm.route_judge import RouteIntentJudge


@dataclass
class LLMConfig:
    mode: str = "hybrid"
    provider_name: str = "openrouter"
    route_model: str = "openrouter/free"
    evidence_model: str = "openrouter/free"
    jobs_model: str = "openrouter/free"
    max_calls: int = 3
    debug: bool = False
    cache_dir: str | None = None


class LLMOrchestrator:
    def __init__(self, config: LLMConfig | None = None, provider=None):
        self.config = config or LLMConfig()
        self.provider = provider or OpenRouterProvider()
        self.cache = LLMCache(self.config.cache_dir)
        self._call_count = 0
        self._judges = {
            "route": RouteIntentJudge(self.provider, self.config.route_model),
            "evidence": EvidenceNormalizerJudge(self.provider, self.config.evidence_model),
            "jobs": JobFitGapJudge(self.provider, self.config.jobs_model),
        }
        self._models = {
            "route": self.config.route_model,
            "evidence": self.config.evidence_model,
            "jobs": self.config.jobs_model,
        }

    def is_enabled(self) -> bool:
        return self.config.mode != "off"

    def run_stage(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_enabled():
            return {"accepted": False, "review_reasons": ["llm_disabled"]}
        if stage not in self._judges:
            raise KeyError(f"Unknown LLM stage: {stage}")
        if self._call_count >= self.config.max_calls:
            return {"accepted": False, "review_reasons": ["llm_call_budget_exceeded"]}

        cache_key = self.cache.build_key(stage, payload, self._models[stage])
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self._call_count += 1
        try:
            result = self._judges[stage].judge(payload)
        except Exception as exc:
            result = {
                "accepted": False,
                "review_reasons": ["llm_provider_error"],
                "error": str(exc),
            }
        self.cache.set(cache_key, result)
        return result

    def apply_job_decisions(self, jobs: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
        return self._judges["jobs"].apply(jobs, decision)
