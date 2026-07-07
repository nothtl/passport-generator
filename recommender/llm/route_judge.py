from __future__ import annotations

import json
from typing import Any

from recommender.llm.cache import _jsonable
from recommender.llm._validation import _as_list, _as_text_list, _clamp_confidence


_ALLOWED_DOMAINS = {
    "technology",
    "healthcare",
    "education",
    "finance",
    "sales",
    "food-service",
    "skilled-trade",
    "design",
    "marketing",
    "ops",
    "legal",
    "arts-media",
    "administrative",
    "logistics",
    "hospitality",
    "manufacturing",
    "agriculture",
    "science",
    "social-service",
    "personal-care",
    "protective-service",
    "building-grounds",
    "support",
}


class RouteIntentJudge:
    def __init__(self, provider, model: str):
        self.provider = provider
        self.model = model

    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate_funcs = _as_text_list(payload.get("candidate_functions"))
        candidate_subs = _as_text_list(payload.get("candidate_subdomains"))
        system_prompt = (
            "You classify ambiguous student career direction for high school / early college students. "
            "Choose ONLY from the provided candidate functions. "
            "Return EXACTLY this JSON structure (no extra fields, no markdown):\n"
            '{\n'
            '  "chosen_function": "<one of the candidate functions>",\n'
            '  "chosen_subdomain": "<one of the candidate subdomains, or empty string>",\n'
            '  "secondary_function": "<second choice or empty string>",\n'
            '  "confidence": <0.0 to 1.0, be honest — 0.55+ for clear cases, lower if unsure>,\n'
            '  "evidence_spans": [\n'
            '    {"section": "<section name>", "text": "<exact sentence from the resume that supports this choice>"},\n'
            '    {"section": "<section name>", "text": "<another exact sentence>"}\n'
            '  ],\n'
            '  "goal_domains": ["<domain1>", "<domain2>"],\n'
            '  "study_domains": ["<domain1>"],\n'
            '  "goal_subdomains": ["<subdomain1>"],\n'
            '  "study_subdomains": ["<subdomain1>"],\n'
            '  "goal_roles": ["<role1>"],\n'
            '  "reason_codes": ["<reason>"]\n'
            '}\n'
            "Provide at least 2 evidence_spans with exact sentences from the input. "
            "CLASSIFICATION RULES (in priority order):\n"
            "1. If the student has explicit career goals, prioritize those over past experience.\n"
            "2. If NO explicit goals: use demonstrated professional experience as the primary signal.\n"
            "3. Education/major is secondary — steer but don't override strong professional credentials.\n"
            "4. Part-time student jobs (waiter, retail) are supporting evidence, not primary direction.\n"
            "5. EVIDENCE QUALITY — distinguish between WEAK and STRONG evidence:\n"
            "   - WEAK: a skill merely listed in a skills section (e.g., 'Skills: Python, Java, Project Management'). "
            "This shows awareness, not competence. Do NOT use listed skills as primary evidence for a function.\n"
            "   - STRONG: a skill demonstrated through WORK EXPERIENCE (e.g., 'Developed Python scripts to automate reports'). "
            "This is real evidence. Use work experience and job titles as your primary signal.\n"
            "   - RULE: If the ONLY evidence for a function is listed skills without any work experience backing, "
            "treat that function as WEAKLY SUPPORTED and prefer functions with actual work experience evidence.\n"
            "6. ELIMINATION RULE — check each candidate function: does the student have STRONG evidence "
            "(work experience, not just listed skills)? If a function has zero work-experience evidence, "
            "ELIMINATE or demote it even if the classifier ranked it highly.\n"
            "7. When EXPERIENCE is thin (only part-time/student jobs) and EDUCATION is clear (specific major): "
            "the major becomes the PRIMARY signal. A History major with no arts/media work experience should "
            "be education or social-service, NOT arts-media. A Psychology major with no tech work experience "
            "should be healthcare, not technology.\n"
            "8. DEGREE MAPPING: Psychology/Biology → healthcare; History/American Studies/Political Science → "
            "education or social-service; Criminal Justice → social-service or legal; English/Communications → "
            "arts-media or education.\n"
            "9. For VERY THIN resumes (<5 clear skills), prefer functions with broad entry-level job pools "
            "(education, healthcare, support, sales, ops, food-service) over narrow ones "
            "(legal, engineering, arts-media, science)."
        )
        user_prompt = (
            f"CANDIDATE FUNCTIONS: {json.dumps(candidate_funcs)}\n"
            f"CANDIDATE SUBDOMAINS: {json.dumps(candidate_subs)}\n\n"
            f"STUDENT DATA:\n{json.dumps(_jsonable(payload), sort_keys=True)}"
        )
        response = self.provider.complete_json("route", system_prompt, user_prompt, self.model)
        candidate_functions = set(_as_text_list(payload.get("candidate_functions")))
        candidate_subdomains = set(_as_text_list(payload.get("candidate_subdomains")))
        chosen = str(response.get("chosen_function", "")).strip()
        chosen_subdomain = str(response.get("chosen_subdomain", "")).strip()
        review_reasons: list[str] = []

        if chosen not in candidate_functions:
            review_reasons.append("invalid_chosen_function")
        if candidate_subdomains and chosen_subdomain and chosen_subdomain not in candidate_subdomains:
            review_reasons.append("invalid_chosen_subdomain")
        confidence = _clamp_confidence(response.get("confidence"))
        if confidence < 0.30:
            review_reasons.append("low_llm_confidence")
        evidence_spans = [
            {"section": str(item.get("section", "")).strip(), "text": str(item.get("text", "")).strip()}
            for item in _as_list(response.get("evidence_spans"))
            if str(item.get("section", "")).strip() and str(item.get("text", "")).strip()
        ]
        if len(evidence_spans) < 1:
            review_reasons.append("insufficient_evidence_spans")

        goal_domains = [domain for domain in _as_text_list(response.get("goal_domains")) if domain in _ALLOWED_DOMAINS]
        study_domains = [domain for domain in _as_text_list(response.get("study_domains")) if domain in _ALLOWED_DOMAINS]

        return {
            "accepted": not review_reasons,
            "chosen_function": chosen,
            "chosen_subdomain": chosen_subdomain,
            "secondary_function": str(response.get("secondary_function", "")).strip(),
            "goal_domains": goal_domains,
            "study_domains": study_domains,
            "goal_subdomains": _as_text_list(response.get("goal_subdomains")),
            "study_subdomains": _as_text_list(response.get("study_subdomains")),
            "goal_roles": _as_text_list(response.get("goal_roles")),
            "confidence": confidence,
            "evidence_spans": evidence_spans,
            "reason_codes": _as_text_list(response.get("reason_codes")),
            "review_reasons": review_reasons,
        }
