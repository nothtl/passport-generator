"""College Scorecard API client.

Fetches college data from the U.S. Department of Education College Scorecard
API using only the standard-library urllib.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recommender.config import get_college

_HERE = Path(__file__).resolve().parent
_CACHE_DIR = _HERE.parent / "data" / "college_cache"


@dataclass
class ScorecardQuery:
    state: str = ""
    cip_code: str = ""
    admission_rate_min: float = 0.0
    admission_rate_max: float = 1.0
    cost_min: int = 0
    cost_max: int = 100_000
    per_page: int = 100
    page: int = 0

    def cache_key(self) -> str:
        raw = (
            f"{self.state}|{self.cip_code}|{self.admission_rate_min}"
            f"|{self.admission_rate_max}|{self.cost_min}|{self.cost_max}"
            f"|{self.per_page}|{self.page}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ScorecardClient:
    api_key: str | None = None
    base_url: str = "https://api.data.gov/ed/collegescorecard/v1/schools"
    cache_dir: Path = field(default_factory=lambda: _CACHE_DIR)
    cache_ttl: int = 86_400
    timeout: int = 30

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("COLLEGE_SCORECARD_API_KEY", "")
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_schools(
        self, state="", cip_code="", admission_rate_min=0.0, admission_rate_max=1.0,
        cost_min=0, cost_max=100_000, per_page=100, page=0, max_pages=1,
    ) -> tuple[list[dict[str, Any]], str]:
        if not self.api_key:
            return [], "COLLEGE_SCORECARD_API_KEY not set"

        all_schools: list[dict[str, Any]] = []
        for pg in range(max_pages):
            query = ScorecardQuery(
                state=state, cip_code=cip_code,
                admission_rate_min=admission_rate_min, admission_rate_max=admission_rate_max,
                cost_min=cost_min, cost_max=cost_max,
                per_page=per_page, page=page + pg,
            )
            schools, err = self._fetch_page(query)
            if err:
                return [], err
            all_schools.extend(schools)
            if len(schools) < per_page:
                break
        return all_schools, ""

    def load_cached(self, query: ScorecardQuery) -> list[dict[str, Any]] | None:
        if self.cache_ttl <= 0:
            return None
        cache_file = self.cache_dir / f"{query.cache_key()}.json"
        if not cache_file.exists():
            return None
        age = time.time() - cache_file.stat().st_mtime
        if age > self.cache_ttl:
            return None
        try:
            with cache_file.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload.get("schools", [])
        except Exception:
            return None

    def save_cache(self, query: ScorecardQuery, schools: list[dict[str, Any]]) -> None:
        cache_file = self.cache_dir / f"{query.cache_key()}.json"
        try:
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump({
                    "query": {
                        "state": query.state, "cip_code": query.cip_code,
                        "admission_rate_min": query.admission_rate_min,
                        "admission_rate_max": query.admission_rate_max,
                        "cost_min": query.cost_min, "cost_max": query.cost_max,
                        "per_page": query.per_page, "page": query.page,
                    },
                    "schools": schools,
                    "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, fh, indent=2)
        except Exception:
            pass

    def _fetch_page(self, query: ScorecardQuery) -> tuple[list[dict[str, Any]], str]:
        cached = self.load_cached(query)
        if cached is not None:
            return cached, ""
        params = self._build_params(query)
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return [], f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            return [], str(e)
        if "results" not in data:
            return [], "API returned no results key"
        schools = self._extract_schools(data)
        self.save_cache(query, schools)
        return schools, ""

    def _build_params(self, query: ScorecardQuery) -> dict[str, str]:
        cfg = get_college()
        api_cfg = cfg.get("scorecard_api", {})
        fields = api_cfg.get("fields", [])
        per_page = query.per_page or api_cfg.get("per_page", 100)
        params: dict[str, str] = {
            "api_key": self.api_key or "", "per_page": str(per_page), "page": str(query.page),
        }
        if fields:
            params["fields"] = ",".join(fields)
        if query.state:
            params["school.state"] = query.state.upper()
        if query.cip_code:
            params["latest.programs.cip_4_digit.code"] = query.cip_code.zfill(4)
        if query.admission_rate_min > 0:
            params["latest.admissions.admission_rate.overall__range"] = f"{query.admission_rate_min}.."
        if query.admission_rate_max < 1.0:
            existing = params.get("latest.admissions.admission_rate.overall__range", "")
            if existing:
                params["latest.admissions.admission_rate.overall__range"] = f"{query.admission_rate_min}..{query.admission_rate_max}"
            else:
                params["latest.admissions.admission_rate.overall__range"] = f"..{query.admission_rate_max}"
        if query.cost_min > 0:
            params["latest.cost.attendance.academic_year__range"] = f"{query.cost_min}.."
        if query.cost_max < 100_000:
            existing = params.get("latest.cost.attendance.academic_year__range", "")
            if existing:
                params["latest.cost.attendance.academic_year__range"] = f"{query.cost_min}..{query.cost_max}"
            else:
                params["latest.cost.attendance.academic_year__range"] = f"..{query.cost_max}"
        return params

    @staticmethod
    def _extract_schools(data: dict[str, Any]) -> list[dict[str, Any]]:
        results = data.get("results", [])
        schools: list[dict[str, Any]] = []
        for row in results:
            cip_raw = row.get("latest.programs.cip_4_digit", [])
            cip_codes: list[str] = []
            if isinstance(cip_raw, list):
                for item in cip_raw:
                    if isinstance(item, dict):
                        code = str(item.get("code", ""))
                        if code:
                            cip_codes.append(code)
                    elif isinstance(item, str):
                        cip_codes.append(item)
            schools.append({
                "id": row.get("id", ""),
                "name": row.get("school.name", "") or row.get("name", ""),
                "city": row.get("school.city", ""),
                "state": row.get("school.state", ""),
                "url": row.get("school.school_url", ""),
                "admission_rate": row.get("latest.admissions.admission_rate.overall"),
                "cost_academic_year": row.get("latest.cost.attendance.academic_year"),
                "tuition_in_state": row.get("latest.cost.tuition.in_state"),
                "tuition_out_of_state": row.get("latest.cost.tuition.out_of_state"),
                "earnings_10yr": row.get("latest.earnings.10_yrs_after_entry.median"),
                "completion_rate": row.get("latest.completion.completion_rate_4yr_150nt"),
                "student_size": row.get("latest.student.size"),
                "degrees_predominant": row.get("school.degrees_awarded.predominant"),
                "carnegie_size": row.get("school.carnegie_size_setting"),
                "cip_codes": cip_codes,
            })
        return schools