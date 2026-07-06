"""
JD retriever — parquet-backed with IDF-weighted skill scoring.

Each extracted skill is weighted by its IDF (inverse document frequency)
computed from the function's JD corpus. Skills that appear in many JDs
get low weight; rare, distinctive skills get high weight. This auto-filters
generic terms like "building", "career", "claims" without any hand-coded rules.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")

# Functions without dedicated parquets fall back to the closest related one
_FALLBACK_MAP = {
    "arts-media": "design",
    "agriculture": "other",
    "building-grounds": "other",
    "personal-care": "other",
    "protective-service": "security",
    "science": "technology",
    "social-service": "education",
    "administrative": "ops",
    "food-service": "other",
    "hospitality": "other",
    "logistics": "ops",
    "manufacturing": "ops",
}

_cached_df: dict[str, Any] = {}
_cached_idf: dict[str, dict[str, float]] = {}
_cached_pmi: dict[str, dict[tuple[str, str], float]] = {}

def _norm_skill(s: str) -> str:
    """Normalize a skill name: lowercase, strip hyphens/spaces/punctuation."""
    return re.sub(r"[- ,/]", "", str(s).lower())

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_BLOCKED_TITLE_PATTERNS = (
    "physician",
    "doctor",
    "surgeon",
    "dvm",
    "vmd",
    "licensed practical nurse",
    "registered nurse",
    "nurse practitioner",
    "director",
    "principal",
    "manager",
)
_SENIORITY_PATTERNS = ("senior",)

def _resolve_func(func_lower: str) -> str:
    """Resolve function to parquet file, using fallback if missing or too small."""
    path = os.path.join(_CORPUS_DIR, f"{func_lower}.parquet")
    if os.path.exists(path) and os.path.getsize(path) > 100_000:  # >100KB = real data
        return func_lower
    # Try subset function map first (e.g., technology → engineering)
    mapped = _SUBSET_FUNCTION_MAP.get(func_lower)
    if mapped:
        mapped_path = os.path.join(_CORPUS_DIR, f"{mapped}.parquet")
        if os.path.exists(mapped_path) and os.path.getsize(mapped_path) > 100_000:
            return mapped
    # Try fallback map
    fallback = _FALLBACK_MAP.get(func_lower)
    if fallback:
        fallback_path = os.path.join(_CORPUS_DIR, f"{fallback}.parquet")
        if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 100_000:
            return fallback
    return func_lower

def _load_df(func_lower: str) -> Any:
    cache_key = func_lower  # always cache under original name
    if cache_key in _cached_df:
        return _cached_df[cache_key]

    resolved = _resolve_func(func_lower)
    path = os.path.join(_CORPUS_DIR, f"{resolved}.parquet")
    if not os.path.exists(path):
        _cached_df[cache_key] = None
        return None
    import pyarrow.parquet as pq
    table = pq.read_table(
        path,
        filters=[("level", "in", ["intern", "entry", "junior", "Intern", "Entry", "Junior"])],
    )
    df = table.to_pandas()
    if df.empty:
        _cached_df[cache_key] = None
        return None
    # Filter out O*NET placeholder rows (e.g. "technology (Entry)")
    df = df[~df["title"].str.lower().str.contains(r"^[a-z]+ \(entry\)$", regex=True, na=False)]
    if df.empty:
        _cached_df[cache_key] = None
        return None
    _cached_df[cache_key] = df
    return df

def _compute_idf(func_lower: str) -> dict[str, float]:
    """Compute IDF for each unique skill in this function's JD corpus.

    IDF(skill) = log(N / df(skill))
    N = total JDs, df(skill) = number of JDs containing this skill.

    Computed once, cached forever.
    """
    if func_lower in _cached_idf:
        return _cached_idf[func_lower]

    df = _load_df(func_lower)
    if df is None or "skills" not in df.columns:
        _cached_idf[func_lower] = {}
        return {}

    N = len(df)

    def _norm_skill(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    # Count document frequency per skill
    skill_df: dict[str, int] = {}
    for skills in df["skills"]:
        if skills is None:
            continue
        seen_in_jd = set()
        for s in (list(skills) if hasattr(skills, "__iter__") else []):
            if isinstance(s, str):
                normed = _norm_skill(s)
                if normed and normed not in seen_in_jd:
                    skill_df[normed] = skill_df.get(normed, 0) + 1
                    seen_in_jd.add(normed)

    idf = {
        skill: math.log(N / max(1, df_count))
        for skill, df_count in skill_df.items()
    }
    _cached_idf[func_lower] = idf
    return idf

def _compute_pmi(func_lower: str) -> dict[tuple[str, str], float]:
    """Compute PMI for skill pairs in this function's JD corpus."""
    df = _load_df(func_lower)
    if df is None or "skills" not in df.columns:
        return {}

    N = len(df)

    def _norm_skill(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    single_freq: dict[str, int] = {}
    pair_freq: dict[tuple[str, str], int] = {}

    for skills in df["skills"]:
        if skills is None:
            continue
        normed = []
        seen = set()
        for s in (list(skills) if hasattr(skills, "__iter__") else []):
            if isinstance(s, str):
                n = _norm_skill(s)
                if n and n not in seen:
                    normed.append(n)
                    seen.add(n)
        for n in normed:
            single_freq[n] = single_freq.get(n, 0) + 1
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                a, b = normed[i], normed[j]
                if a > b:
                    a, b = b, a
                pair_freq[(a, b)] = pair_freq.get((a, b), 0) + 1

    pmi = {}
    for (a, b), pair_count in pair_freq.items():
        if pair_count < 3:
            continue
        p_a = single_freq.get(a, 0) / N
        p_b = single_freq.get(b, 0) / N
        if p_a <= 0 or p_b <= 0:
            continue
        p_ab = pair_count / N
        score = math.log(p_ab / (p_a * p_b))
        if score > 0.15:
            pmi[(a, b)] = round(score, 3)

    return pmi

def get_related_skills(function: str, student_skills: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    """Find skills the student lacks that co-occur with ones they have."""
    def _norm_skill(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    pmi = _compute_pmi(function.lower())
    student_normed = {_norm_skill(s) for s in student_skills}

    related = {}
    for (a, b), score in pmi.items():
        has_a = a in student_normed
        has_b = b in student_normed
        if has_a and not has_b:
            related[b] = max(related.get(b, 0), score)
        elif has_b and not has_a:
            related[a] = max(related.get(a, 0), score)

    ranked = sorted(related.items(), key=lambda x: -x[1])
    return ranked[:top_k]

def get_jd_skill_vocabulary(function: str) -> set[str]:
    """Return the set of all unique skill names in this function's JDs.
    Used for filtering student skills to only market-relevant ones."""
    func_lower = function.lower()
    df = _load_df(func_lower)
    if df is None or 'skills' not in df.columns:
        return set()
    
    import re
    def _norm_skill(s):
        return re.sub(r'[- ,/]', '', str(s).lower())
    
    vocab = set()
    for skills in df['skills']:
        if skills is None: continue
        for s in (list(skills) if hasattr(skills, '__iter__') else []):
            if isinstance(s, str) and len(s) > 2:
                vocab.add(_norm_skill(s))
    return vocab

def filter_job_records(
    jobs: list[dict[str, Any]],
    candidate_function: str = "",
    goal_domains: list[str] | None = None,
    study_domains: list[str] | None = None,
    goal_subdomains: list[str] | None = None,
    study_subdomains: list[str] | None = None,
    study_level: str = "",
) -> list[dict[str, Any]]:
    goal_domains = goal_domains or []
    study_domains = study_domains or []
    goal_subdomains = goal_subdomains or []
    study_subdomains = study_subdomains or []
    filtered: list[dict[str, Any]] = []

    for raw in jobs:
        job = dict(raw)
        title = str(job.get("title", "")).strip()
        company = str(job.get("company", "")).strip()
        if _UUID_RE.match(company):
            company = ""
        if not title or len(re.sub(r"[^a-zA-Z]", "", title)) < 4:
            continue
        if _UUID_RE.match(title):
            continue
        if not company and len(title) < 5:
            continue
        job["company"] = company
        job["_quality_penalty"] = 0.0
        lowered = f"{title} {job.get('jd_markdown', '')}".lower()
        subdomain_hints = " ".join(goal_subdomains + study_subdomains).lower()
        has_student_credentials = any(
            token in lowered or token in subdomain_hints
            for token in ["medical school", "premed", "pre-med", "doctor", "physician shadowing"]
        ) and study_level in {"master", "doctoral"}
        if any(token in lowered for token in _BLOCKED_TITLE_PATTERNS) and not has_student_credentials:
            continue
        if any(token in lowered for token in _SENIORITY_PATTERNS):
            if study_level not in {"master", "doctoral"}:
                continue
        if candidate_function in {"education", "social-service"} and any(
            token in lowered for token in ["pizza", "kitchen", "wine steward", "bartender"]
        ):
            if not set(goal_domains + study_domains) & {"food-service", "hospitality"}:
                job["_quality_penalty"] = 0.35
        if candidate_function in {"education", "social-service"} and any(
            token in lowered for token in ["graphic designer", "design intern", "creative", "marketing"]
        ):
            if not set(goal_domains + study_domains) & {"design", "arts-media", "marketing"}:
                job["_quality_penalty"] = max(job["_quality_penalty"], 0.45)
        if candidate_function == "healthcare" and any(
            token in lowered for token in ["delivery", "veterinary", "animal hospital", "vet tech"]
        ):
            if not set(goal_subdomains + study_subdomains) & {"research", "public-health"}:
                job["_quality_penalty"] = max(job["_quality_penalty"], 0.4)
        if candidate_function == "technology" and any(
            token in lowered for token in ["cashier", "retail associate", "food service"]
        ):
            job["_quality_penalty"] = max(job["_quality_penalty"], 0.35)
        filtered.append(job)

    return filtered

_SUBSET_PATH = os.path.join(_CORPUS_DIR, "open_jobs_subset.parquet")
_subset_cache = None

def _load_subset():
    global _subset_cache
    if _subset_cache is None and os.path.exists(_SUBSET_PATH):
        import pyarrow.parquet as pq
        _subset_cache = pq.read_table(_SUBSET_PATH).to_pandas()
    return _subset_cache

def _stream_full_parquet(function_labels: set[str], keyword_filters: list[str], top_k: int) -> list[dict]:
    """Stream through the full 21GB parquet by row groups, filtering on the fly."""
    full_path = os.path.join(_CORPUS_DIR, "_open_jobs_full.parquet")
    if not os.path.exists(full_path):
        return []

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(full_path)
    matched = []
    target_levels = {"intern", "entry", "junior", "", "unknown"}
    _senior = ("manager", "director", "principal", "senior", "lead ", "chief ",
               "attorney", "counsel", "general application", "account executive", "vp ")

    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=["id", "ats", "company", "title", "url",
                                                "jd_markdown", "level", "function", "skills",
                                                "country_code"])
        df = table.to_pandas()
        # Filter by function labels
        if function_labels:
            mask = df["function"].str.lower().isin(function_labels)
            df = df[mask]
        # Filter by level
        if "level" in df.columns:
            mask = df["level"].fillna("").str.lower().isin(target_levels)
            df = df[mask]
        # Filter out senior titles
        if len(df) > 0:
            mask = ~df["title"].fillna("").str.lower().apply(
                lambda t: any(p in str(t).lower() for p in _senior))
            df = df[mask]
        # Keyword filter for specific functions
        if keyword_filters and len(df) > 0:
            kmask = df["title"].fillna("").str.lower().apply(
                lambda t: any(kw in str(t).lower() for kw in keyword_filters))
            df = df[kmask]
        matched.append(df)
        if sum(len(m) for m in matched) >= top_k * 3:
            break

    if matched:
        import pandas as pd
        result = pd.concat(matched, ignore_index=True)
        return result.head(top_k).to_dict("records")
    return []

# Map our pipeline functions to subset parquet labels (different vocabularies)
_ONET_ONLY_FUNCTIONS = {"protective-service"}  # Subset has no physical security jobs — use O*NET fallback exclusively

_SUBSET_FUNCTION_MAP = {
    # Subset uses different labels — map our vocabulary to theirs
    "technology": "engineering",
    "social-service": "education",
    "arts-media": "design",
    "agriculture": "other",
    "building-grounds": "other",
    "personal-care": "other",
    "science": "research",
    "food-service": "other",
    "administrative": "ops",
    "logistics": "ops",
    "manufacturing": "ops",
    "hospitality": "other",
    "protective-service": "security",
    "security": "security",
}

# Keywords that redirect generic-function jobs to our specific functions
_FUNCTION_KEYWORDS = {
    "protective-service": [
        "security guard", "security officer", "loss prevention",
        "patrol", "surveillance", "asset protection", "doorperson",
        "front desk monitor", "store detective",
    ],
    "it-support": [
        "help desk", "technical support", "it support", "desktop support",
        "network support", "systems administrator", "it technician",
        "tech support", "support engineer", "support technician",
    ],
    "social-service": [
        "youth", "community", "outreach", "social work", "case management",
        "peer support", "nonprofit", "non-profit", "advocacy", "counseling",
        "americorps", "volunteer coordinator",
    ],
    "technology": [
        "software", "developer", "engineer", "data", "systems", "network",
        "cloud", "devops", "full stack", "frontend", "backend", "python",
        "java", "react", "aws", "machine learning", "ai", "embedded",
    ],
    "arts-media": [
        "graphic design", "photographer", "videographer", "video editor",
        "content creator", "social media", "illustrator", "photo",
        "creative", "visual", "designer",
    ],
}

def retrieve_from_subset(
    function: str = "",
    student_skills: list[str] | None = None,
    top_k: int = 10,
    function_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve jobs from the clean open-jobs-subset parquet instead of per-function files.

    Uses keyword-based function routing to bridge the gap between our function vocabulary
    and the subset's labels (e.g., 'protective-service' doesn't exist in the subset,
    so we pull from 'security' and keyword-filter for guard/patrol roles).
    """
    df = _load_subset()
    if df is None or df.empty:
        return retrieve_jds(function, "Entry", student_skills, top_k=top_k)

    # O*NET-only functions: subset has no good data for these
    if function in _ONET_ONLY_FUNCTIONS:
        return _get_onet_fallback(function, top_k)

    # Build function labels: map our functions to subset labels
    all_labels = function_labels or ([function] if function else [])
    subset_labels = set()
    for label in all_labels:
        mapped = _SUBSET_FUNCTION_MAP.get(label, label)
        subset_labels.add(mapped.lower())

    # For thin functions with keyword filters, stream the full 21GB parquet.
    # Only for truly thin functions where the subset has no data.
    keyword_filters = _FUNCTION_KEYWORDS.get(function, [])
    full_path = os.path.join(_CORPUS_DIR, "_open_jobs_full.parquet")
    STREAM_FUNCTIONS = {"protective-service", "social-service", "arts-media", "legal"}
    if function in STREAM_FUNCTIONS and keyword_filters and os.path.exists(full_path):
        results = _stream_full_parquet(subset_labels, keyword_filters, top_k)
        if len(results) >= top_k:
            return results
        # Fall through to subset + O*NET fallback

    # Filter by subset labels
    if subset_labels:
        mask = df["function"].str.lower().isin(subset_labels)
        df = df[mask]

    if len(df) == 0:
        return []

    # Filter to entry-level (including Unknown — the zip keeps these)
    target_levels = {"intern", "entry", "junior", "", "unknown"}
    if "level" in df.columns:
        df = df[df["level"].fillna("").str.lower().isin({l.lower() for l in target_levels})]

    # Drop clearly non-entry titles from the Unknown pool
    _senior_patterns = (
        "manager", "director", "principal", "senior", "lead ", "chief ",
        "attorney", "counsel", "reporter", "editor", "general application",
        "account executive", "vp ", "vice president",
    )
    # Block non-matching job types from tech/engineering pools
    _tech_blocklist = (
        "delivery driver", "warehouse", "truck", "box truck", "non-cdl",
        "cdl", "forklift", "amazon delivery", "package handler",
    )
    if len(df) > 0:
        mask = ~df["title"].fillna("").str.lower().apply(
            lambda t: any(p in str(t).lower() for p in _senior_patterns)
        )
        df = df[mask]
        # Apply tech blocklist only for engineering/data/product functions
        if any(f in subset_labels for f in ["engineering", "data", "product", "technology"]):
            mask2 = ~df["title"].fillna("").str.lower().apply(
                lambda t: any(p in str(t).lower() for p in _tech_blocklist)
            )
            df = df[mask2]

    if len(df) == 0:
        return []

    # Keyword-based filtering for functions that don't have direct subset labels
    # EXCLUSIVE: only keep jobs that explicitly match the keywords, then fall back to O*NET if none found
    if keyword_filters:
        # For keyword-filtered functions, ALWAYS apply the filter and use O*NET fallback if empty
        keyword_mask = df["title"].str.lower().apply(
            lambda t: any(kw in str(t).lower() for kw in keyword_filters)
        )
        if "jd_markdown" in df.columns:
            jd_mask = df["jd_markdown"].fillna("").str.lower().apply(
                lambda j: any(kw in str(j).lower() for kw in keyword_filters)
            )
            df = df[keyword_mask | jd_mask]
        else:
            df = df[keyword_mask]

    # Skill-based scoring
    if student_skills and "skills" in df.columns and len(df) > 0:
        import re as _re2
        _norm2 = lambda s: _re2.sub(r"[- ,/]", "", str(s).lower())
        student_set = {_norm_skill(s) for s in student_skills}

        def _score(row_skills):
            if row_skills is None:
                return 0
            total = 0.0
            try:
                skills_list = list(row_skills) if hasattr(row_skills, "__iter__") else []
            except Exception:
                return 0
            for s in skills_list:
                if isinstance(s, str) and _norm_skill(s) in student_set:
                    total += 1.0
            return total

        scores = df["skills"].apply(_score)
        df = df.iloc[(-scores).argsort()]

    results = df.head(top_k).to_dict("records") if len(df) > 0 else []

    # Fill gaps with O*NET synthetic fallback jobs for functions the subset doesn't cover.
    # Also fill if keyword-filtered results are too few (subset lacks these job types).
    if len(results) < top_k or (keyword_filters and len(results) < 3):
        needed = top_k - len(results)
        fallbacks = _get_onet_fallback(function, needed)
        results.extend(fallbacks)

    return results

# Synthetic O*NET fallback jobs for functions with thin/no subset coverage
_ONET_FALLBACKS: dict[str, list[dict]] = {
    "protective-service": [
        {
            "id": "onet_protective_01", "ats": "onet", "company": "",
            "title": "Security Officer (Entry Level)",
            "url": "", "level": "Entry", "function": "protective-service",
            "jd_markdown": "Monitor premises to prevent theft and ensure safety. Conduct bag checks and access control. Write incident reports and communicate with law enforcement when needed.",
            "skills": ["security", "access control", "surveillance", "incident reporting", "customer service", "communication", "attention to detail", "conflict resolution"],
        },
        {
            "id": "onet_protective_02", "ats": "onet", "company": "",
            "title": "Loss Prevention Associate",
            "url": "", "level": "Entry", "function": "protective-service",
            "jd_markdown": "Prevent inventory shrinkage through monitoring and investigation. Maintain store security procedures and conduct employee bag checks. Document incidents and support store management.",
            "skills": ["loss prevention", "monitoring", "report writing", "customer service", "attention to detail", "communication", "teamwork"],
        },
        {
            "id": "onet_protective_03", "ats": "onet", "company": "",
            "title": "Transportation Security Officer",
            "url": "", "level": "Entry", "function": "protective-service",
            "jd_markdown": "Screen passengers and baggage at checkpoints. Operate X-ray and metal detection equipment. Enforce security regulations and respond to security incidents.",
            "skills": ["security screening", "attention to detail", "communication", "customer service", "equipment operation", "regulatory compliance"],
        },
        {
            "id": "onet_protective_04", "ats": "onet", "company": "",
            "title": "Safety Monitor / Hall Monitor",
            "url": "", "level": "Entry", "function": "protective-service",
            "jd_markdown": "Monitor hallways and common areas to ensure safety and order. Check visitor credentials and report incidents. Support emergency evacuation procedures.",
            "skills": ["monitoring", "observation", "communication", "safety procedures", "incident reporting", "customer service"],
        },
        {
            "id": "onet_protective_05", "ats": "onet", "company": "",
            "title": "Event Security Staff",
            "url": "", "level": "Entry", "function": "protective-service",
            "jd_markdown": "Provide security at events and venues. Check tickets and credentials at entry points. Monitor crowds and respond to safety concerns. Coordinate with event staff.",
            "skills": ["security", "crowd management", "communication", "customer service", "incident response", "teamwork"],
        },
    ],
    "social-service": [
        {
            "id": "onet_social_svc_04", "ats": "onet", "company": "",
            "title": "Peer Support Specialist",
            "url": "", "level": "Entry", "function": "social-service",
            "jd_markdown": "Provide peer mentoring and support to youth in community programs. Share lived experience to build trust. Connect peers to resources and advocate for their needs.",
            "skills": ["peer mentoring", "communication", "advocacy", "community resources", "active listening", "empathy"],
        },
        {
            "id": "onet_social_svc_05", "ats": "onet", "company": "",
            "title": "Volunteer and Events Assistant",
            "url": "", "level": "Entry", "function": "social-service",
            "jd_markdown": "Support volunteer recruitment and coordination for community events. Help with event setup, attendee registration, and post-event follow-up. Assist with social media outreach.",
            "skills": ["volunteer coordination", "event support", "communication", "social media", "organization", "customer service"],
        },
    ],
    "social-service": [
        {
            "id": "onet_social_01", "ats": "onet", "company": "",
            "title": "Community Outreach Coordinator",
            "url": "", "level": "Entry", "function": "social-service",
            "jd_markdown": "Coordinate community programs and outreach events. Build relationships with local organizations and stakeholders. Recruit and manage volunteers for community initiatives.",
            "skills": ["community outreach", "volunteer coordination", "event planning", "communication", "public speaking", "program management", "bilingual"],
        },
        {
            "id": "onet_social_02", "ats": "onet", "company": "",
            "title": "Youth Program Assistant",
            "url": "", "level": "Entry", "function": "social-service",
            "jd_markdown": "Support after-school and youth development programs. Mentor young people and assist with educational activities. Help with program planning, attendance tracking, and parent communication.",
            "skills": ["youth development", "mentoring", "program coordination", "communication", "childcare", "activity planning", "teamwork"],
        },
        {
            "id": "onet_social_03", "ats": "onet", "company": "",
            "title": "Case Management Aide",
            "url": "", "level": "Entry", "function": "social-service",
            "jd_markdown": "Assist case managers with client intake, documentation, and follow-up. Help clients access community resources and benefits. Maintain case files and schedule appointments.",
            "skills": ["case management", "documentation", "customer service", "data entry", "communication", "organization", "empathy"],
        },
    ],
    "arts-media": [
        {
            "id": "onet_arts_01", "ats": "onet", "company": "",
            "title": "Content Creator / Social Media Assistant",
            "url": "", "level": "Entry", "function": "arts-media",
            "jd_markdown": "Create engaging social media content including photos, videos, and written posts. Manage content calendar and track engagement metrics. Support brand storytelling across platforms.",
            "skills": ["content creation", "social media management", "photography", "videography", "writing", "graphic design", "communication"],
        },
        {
            "id": "onet_arts_02", "ats": "onet", "company": "",
            "title": "Photography / Video Production Assistant",
            "url": "", "level": "Entry", "function": "arts-media",
            "jd_markdown": "Assist with photo and video shoots including equipment setup, lighting, and post-production. Organize digital assets and support the creative team with editing tasks.",
            "skills": ["photography", "videography", "video editing", "lighting", "organization", "creativity", "teamwork"],
        },
    ],
    "legal": [
        {
            "id": "onet_legal_01", "ats": "onet", "company": "",
            "title": "Legal Assistant / Paralegal Aide",
            "url": "", "level": "Entry", "function": "legal",
            "jd_markdown": "Support attorneys with document preparation, filing, and client communication. Organize case files, schedule appointments, and conduct basic legal research under supervision.",
            "skills": ["documentation", "organization", "communication", "research", "data entry", "attention to detail", "writing", "bilingual"],
        },
        {
            "id": "onet_legal_02", "ats": "onet", "company": "",
            "title": "Legal Intake Specialist",
            "url": "", "level": "Entry", "function": "legal",
            "jd_markdown": "Conduct initial client interviews and intake assessments. Gather documentation and determine eligibility for legal services. Maintain client confidentiality and accurate records.",
            "skills": ["client intake", "documentation", "communication", "data entry", "bilingual", "organization", "empathy"],
        },
    ],
    "security": [
        {
            "id": "onet_sec_01", "ats": "onet", "company": "",
            "title": "Security Guard (Unarmed)",
            "url": "", "level": "Entry", "function": "security",
            "jd_markdown": "Patrol assigned areas to ensure safety and security. Monitor access points and verify credentials. Respond to incidents and write detailed reports.",
            "skills": ["security", "patrol", "access control", "observation", "incident reporting", "communication", "customer service"],
        },
    ],
}

def _get_onet_fallback(function: str, count: int) -> list[dict]:
    """Return synthetic O*NET fallback jobs for functions with thin subset coverage."""
    fallbacks = _ONET_FALLBACKS.get(function, [])
    if not fallbacks:
        return []
    return fallbacks[:count]

def retrieve_jds(
    function: str,
    level: str = "Entry",
    student_skills: list[str] | None = None,
    top_k: int = 10,
    broad_sample: int = 0,
) -> list[dict[str, Any]]:
    func_lower = function.lower()
    df = _load_df(func_lower)
    if df is None:
        return []

    import pandas as pd

    if not student_skills or "skills" not in df.columns:
        result = df.head(top_k)
        if broad_sample > 0 and len(df) > top_k:
            extra = df.iloc[top_k:].sample(
                n=min(broad_sample, len(df) - top_k), random_state=42
            )
            result = pd.concat([result, extra])
        return result.to_dict("records")

    def _norm_skill(s):
        return re.sub(r"[- ,/]", "", str(s).lower())

    idf = _compute_idf(func_lower)
    student_set = set(_norm_skill(s) for s in student_skills)

    # Build TF weights from the skill list: if a skill like "security" appears
    # multiple times in the extracted skills (from different n-grams), weight it higher.
    # Single-occurrence skills get TF=1.0 baseline.
    from collections import Counter as _Counter
    _tf_counts = _Counter(_norm_skill(s) for s in student_skills)
    student_tf = {n: 1 + math.log(c) for n, c in _tf_counts.items()}

    # Expand via ESCO synonyms (data-driven, 85K alt-labels)
    try:
        _here = os.path.dirname(__file__)
        _syn_path = os.path.join(_here, "..", "data", "esco_synonyms.json")
        with open(_syn_path) as _f:
            _esco = json.load(_f)
        for canonical, aliases in _esco.items():
            if _norm_skill(canonical) in student_set:
                continue
            if any(_norm_skill(a) in student_set for a in aliases):
                student_set.add(_norm_skill(canonical))
                student_tf[_norm_skill(canonical)] = 1.5  # inferred synonyms get slight boost
    except Exception:
        pass

    # TF-IDF weighted scoring: TF(skill) × IDF(corpus)
    def _score_jd(row_skills):
        if row_skills is None:
            return 0
        total = 0.0
        for s in (list(row_skills) if hasattr(row_skills, "__iter__") else []):
            if isinstance(s, str):
                normed = _norm_skill(s)
                if normed in student_set:
                    tf = student_tf.get(normed, 1.0)
                    total += tf * idf.get(normed, 1.0)
        return total

    scores = df["skills"].apply(_score_jd)
    ranked = df.iloc[(-scores).argsort()]

    top = ranked.head(top_k)
    if broad_sample > 0 and len(ranked) > top_k:
        tail = ranked.iloc[top_k:]
        extra = tail.sample(n=min(broad_sample, len(tail)), random_state=42)
        result = pd.concat([top, extra])
        return filter_job_records(result.to_dict("records"), candidate_function=function)
    return filter_job_records(top.to_dict("records"), candidate_function=function)
