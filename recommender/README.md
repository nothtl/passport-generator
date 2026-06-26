# Recommender

Takes a student resume → produces role profile with skills (has ✓ / missing ✗), ideal passport, and real job openings.

## Pipeline

```
resume / linkedin
      │
      ▼
┌─ extract/ ─────────────────────┐
│ skill_extractor.py              │
│ keyword match → skills[]        │
└──────────────┬─────────────────┘
               │
               ▼
┌─ match/ ───────────────────────┐
│ lookup_table.json               │
│ role_matcher.py                 │
│ → best role + match %           │
└──────────────┬─────────────────┘
               │
               ▼
┌─ retrieve/ ────────────────────┐
│ retriever.py                   │
│ corpus/our_jobs.parquet         │
│ → real JDs for this role       │
└──────────────┬─────────────────┘
               │
               ▼
┌─ profile/ ─────────────────────┐
│ aggregator.py + gap_analyzer.py │
│ → RoleProfile (skills, gaps,   │
│    passport, openings)          │
└────────────────────────────────┘
```

## Modules

| Module | What it does | Dependencies |
|--------|-------------|-------------|
| `extract/` | Extract named skills from resume text (keyword + regex) | None |
| `match/` | Compare student skills against role lookup table, return best fit | `lookup_table.json` |
| `retrieve/` | Hull filter + cosine rank against local JD corpus | `corpus/our_jobs.parquet` (optional), `passport_agent_v2` embeddings |
| `profile/` | Aggregate JDs into skill frequency + passport scores, mark ✓/✗ gaps | `passport_agent_v2.pillars` |

## Getting started

```python
from recommender.extract.skill_extractor import extract_skills_from_text
from recommender.match.role_matcher import match_role
from recommender.retrieve.retriever import retrieve_jds
from recommender.profile.aggregator import aggregate_skills, aggregate_passport
from recommender.profile.gap_analyzer import analyze_gaps

# Stage 1 — extract skills
skills = extract_skills_from_text(resume_text)

# Stage 2 — find best-fit role
best_role = match_role(skills)

# Stage 3 — retrieve real JDs
jds = retrieve_jds(best_role["function"], best_role["level"], skills)

# Stage 4 — build role profile
all_skills = aggregate_skills(jds, best_role["matched_skills"], best_role["missing_skills"])
passport = aggregate_passport(jds)
result = analyze_gaps(
    role_title=best_role["function"],
    function=best_role["function"],
    level=best_role["level"],
    match_pct=best_role["match_pct"],
    matched_skills=best_role["matched_skills"],
    missing_skills=best_role["missing_skills"],
    all_skills=all_skills,
    ideal_passport=passport,
)
```

## Running tests

```
python -m recommender.tests.test_pipeline
```

## Corpus

Download and build the local JD corpus:

```
python -m recommender.corpus.download
```

This merges O\*NET synthetic JDs with filtered open-jobs into `corpus/our_jobs.parquet`.
Without a corpus, the retriever returns empty but the rest of the pipeline still works.
