"""College ranker — scores and ranks colleges for a student profile.

Each college receives four sub-scores in ``[0, 1]``:

* **program_fit** — how well the college's CIP programs match the student's
  career function (derived from ``config.college.cip_to_function``).
* **cost_fit** — how affordable the college is relative to a max budget.
* **admission_fit** — how closely the admission rate matches the student's
  academic competitiveness (GPA / test-optional heuristic).
* **outcome_fit** — earnings 10 years after entry (higher is better).

The final score is a weighted sum; weights are configurable via
``config.college.ranker.weights`` and default to:

    program_fit: 0.35
    cost_fit:    0.20
    admission_fit: 0.20
    outcome_fit:  0.25
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Default weights (overridable via config) ────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "program_fit": 0.35,
    "cost_fit": 0.20,
    "admission_fit": 0.20,
    "outcome_fit": 0.25,
}


@dataclass
class CollegeResult:
    """A single ranked college recommendation."""

    name: str
    city: str
    state: str
    admission_rate: float | None
    cost: float | None
    earnings: float | None
    tier: str  # "reach" | "match" | "safety"
    score: float  # overall [0, 1]
    why: str  # human-readable explanation
    url: str = ""
    score_breakdown: dict[str, float] = field(default_factory=dict)
    cip_codes: list[str] = field(default_factory=list)
    student_size: int | None = None


@dataclass
class CollegeRanker:
    """Ranks colleges by configurable weighted criteria.

    Parameters
    ----------
    weights:
        Override the default scoring weights.
    max_cost:
        Maximum annual cost considered "affordable" (default $60 000).
        Colleges at or below this receive ``cost_fit = 1.0``; those at
        ``2 * max_cost`` or above receive ``cost_fit = 0.0``.
    target_admission_rate:
        The admission rate that represents the student's best fit.
        Colleges with this rate get ``admission_fit = 1.0``; the score
        falls off symmetrically.
    min_earnings / max_earnings:
        Range for normalising the ``outcome_fit`` score.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_cost: float = 60_000.0
    target_admission_rate: float = 0.45
    min_earnings: float = 25_000
    max_earnings: float = 90_000
    preferred_state: str = ""

    def rank(
        self,
        schools: list[dict[str, Any]],
        student_functions: list[str],
        cip_to_function: dict[str, str] | None = None,
    ) -> list[CollegeResult]:
        """Score and sort *schools*, returning a list of :class:`CollegeResult`.

        ``student_functions`` are the career function labels (e.g.
        ``["technology", "engineering"]``) derived from the student's
        intent profile.
        ``cip_to_function`` maps 2-digit CIP code prefixes to function
        labels (from ``config.college.cip_to_function``).
        """
        cip_map = cip_to_function or {}
        results: list[CollegeResult] = []
        for school in schools:
            sub_scores, breakdown_reasons = self._score_school(
                school, student_functions, cip_map
            )
            total = sum(
                self.weights.get(k, 0.0) * v for k, v in sub_scores.items()
            )
            # Location bonus for in-state schools
            if self.preferred_state and school.get("state", "").upper() == self.preferred_state.upper():
                total += 0.12
            total = max(0.0, min(1.0, total))

            admission_rate = school.get("admission_rate")
            tier = _classify_tier(admission_rate)

            why = self._build_explanation(
                school, sub_scores, breakdown_reasons, student_functions, cip_map
            )

            results.append(
                CollegeResult(
                    name=school.get("name", "Unknown"),
                    city=school.get("city", ""),
                    state=school.get("state", ""),
                    admission_rate=admission_rate,
                    cost=school.get("cost_academic_year"),
                    earnings=school.get("earnings_10yr"),
                    tier=tier,
                    score=round(total, 4),
                    why=why,
                    url=school.get("url", ""),
                    score_breakdown={
                        k: round(v, 4) for k, v in sub_scores.items()
                    },
                    cip_codes=school.get("cip_codes", []),
                    student_size=school.get("student_size"),
                )
            )

        results.sort(key=lambda r: (-r.score, r.name))
        return results

    # ── Scoring helpers ──────────────────────────────────────────────

    def _score_school(
        self,
        school: dict[str, Any],
        student_functions: list[str],
        cip_map: dict[str, str],
    ) -> tuple[dict[str, float], list[str]]:
        program_fit, prog_reasons = self._program_fit(
            school, student_functions, cip_map
        )
        cost_fit, cost_reasons = self._cost_fit(school)
        admission_fit, adm_reasons = self._admission_fit(school)
        outcome_fit, outcome_reasons = self._outcome_fit(school)

        scores = {
            "program_fit": program_fit,
            "cost_fit": cost_fit,
            "admission_fit": admission_fit,
            "outcome_fit": outcome_fit,
        }
        reasons = prog_reasons + cost_reasons + adm_reasons + outcome_reasons
        return scores, reasons

    @staticmethod
    def _program_fit(
        school: dict[str, Any],
        student_functions: list[str],
        cip_map: dict[str, str],
    ) -> tuple[float, list[str]]:
        """Score how well the college's programs match the student's functions."""
        if not student_functions:
            return 0.3, ["no student functions to match"]
        cip_codes = school.get("cip_codes", [])
        if not cip_codes:
            return 0.2, ["college has no program data"]

        student_set = {f.lower() for f in student_functions}
        matches = 0
        matched_functions: list[str] = []
        for code in cip_codes:
            prefix = str(code)[:2]
            func = cip_map.get(prefix)
            if func and func.lower() in student_set:
                matches += 1
                if func not in matched_functions:
                    matched_functions.append(func)

        if matches == 0:
            return 0.0, ["no matching programs"]

        # Normalise: 1 match = 0.6, 2 = 0.8, 3+ = 1.0
        score = min(1.0, 0.4 + 0.2 * matches)
        return score, [f"matched programs in {', '.join(matched_functions)}"]

    def _cost_fit(self, school: dict[str, Any]) -> tuple[float, list[str]]:
        cost = school.get("cost_academic_year")
        if cost is None or cost <= 0:
            return 0.5, ["cost data unavailable"]
        if cost <= self.max_cost:
            return 1.0, [f"affordable (${cost:,.0f}/yr)"]
        if cost >= self.max_cost * 2:
            return 0.0, [f"expensive (${cost:,.0f}/yr)"]
        # Linear falloff
        score = 1.0 - (cost - self.max_cost) / self.max_cost
        return max(0.0, score), [f"moderate cost (${cost:,.0f}/yr)"]

    def _admission_fit(self, school: dict[str, Any]) -> tuple[float, list[str]]:
        rate = school.get("admission_rate")
        if rate is None or rate <= 0:
            return 0.5, ["admission data unavailable"]
        # Gaussian-style falloff around target
        diff = abs(rate - self.target_admission_rate)
        score = max(0.0, 1.0 - (diff / 0.5) ** 2)
        pct = round(rate * 100)
        return score, [f"admission rate {pct}%"]

    def _outcome_fit(self, school: dict[str, Any]) -> tuple[float, list[str]]:
        earnings = school.get("earnings_10yr")
        if earnings is None or earnings <= 0:
            return 0.3, ["earnings data unavailable"]
        if earnings >= self.max_earnings:
            return 1.0, [f"strong earnings (${earnings:,.0f})"]
        if earnings <= self.min_earnings:
            return 0.0, [f"low earnings (${earnings:,.0f})"]
        score = (earnings - self.min_earnings) / (
            self.max_earnings - self.min_earnings
        )
        return max(0.0, min(1.0, score)), [
            f"earnings ${earnings:,.0f} 10 yrs post-entry"
        ]

    # ── Explanation builder ─────────────────────────────────────────

    @staticmethod
    def _build_explanation(
        school: dict[str, Any],
        scores: dict[str, float],
        reasons: list[str],
        student_functions: list[str],
        cip_map: dict[str, str],
    ) -> str:
        name = school.get("name", "this school")
        parts: list[str] = []

        # Program match explanation
        cip_codes = school.get("cip_codes", [])
        matched: list[str] = []
        student_set = {f.lower() for f in student_functions}
        for code in cip_codes:
            prefix = str(code)[:2]
            func = cip_map.get(prefix)
            if func and func.lower() in student_set and func not in matched:
                matched.append(func)
        if matched:
            parts.append(
                f"Offers programs aligned with your interests in {', '.join(matched)}"
            )
        else:
            parts.append("Limited program overlap with your stated interests")

        # Cost explanation
        cost = school.get("cost_academic_year")
        if cost and cost > 0:
            parts.append(f"Annual cost ~${cost:,.0f}")

        # Admission explanation
        rate = school.get("admission_rate")
        if rate and rate > 0:
            parts.append(f"Admission rate {round(rate * 100)}%")

        # Earnings explanation
        earnings = school.get("earnings_10yr")
        if earnings and earnings > 0:
            parts.append(f"Median earnings ${earnings:,.0f} 10 yrs after entry")

        # Top reason from sub-scores
        best_dim = max(scores, key=scores.get) if scores else ""
        if best_dim == "program_fit" and matched:
            parts.append("Strong program match")
        elif best_dim == "cost_fit" and scores.get("cost_fit", 0) > 0.7:
            parts.append("Good value")
        elif best_dim == "outcome_fit" and scores.get("outcome_fit", 0) > 0.7:
            parts.append("Strong career outcomes")

        return "; ".join(parts)


# ── Tier classification ─────────────────────────────────────────────


def _classify_tier(
    admission_rate: float | None,
    reach_max: float = 0.25,
    match_min: float = 0.25,
    match_max: float = 0.60,
    safety_min: float = 0.60,
) -> str:
    """Classify a college into reach / match / safety based on admission rate.

    * **Reach** — admission rate ≤ 25 % (very selective)
    * **Match** — admission rate 25 %–60 %
    * **Safety** — admission rate ≥ 60 %

    Thresholds are overridable from ``config.college.tiers``.
    """
    if admission_rate is None or admission_rate <= 0:
        return "match"  # unknown — default to match
    if admission_rate <= reach_max:
        return "reach"
    if admission_rate >= safety_min:
        return "safety"
    return "match"
