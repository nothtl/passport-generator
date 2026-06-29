# LLM Integration Plan

## Where It Adds Value

| Stage | Current (no LLM) | With LLM | Impact |
|---|---|---|---|
| **1. Classification** | Ensemble voting, 89% accuracy | LLM re-classifies when confidence < 30% | Fixes paralegal→finance, nurse→legal |
| **2. Skill evidence** | Just lists skill names | Shows WHERE each skill was found + proficiency | Old pipeline's best feature |
| **3. Gap explanation** | Lists missing skills | "Python appears in 847 of 1,915 tech JDs. With your FastAPI and Azure skills, adding Python would open up backend roles." | Actionable guidance |
| **4. Ready vs Target** | One job list | LLM splits jobs into "ready now" vs "growth path" | Student sees both immediate + aspirational |

## Architecture

```
resume text
    │
    ├──► Current pipeline (always runs, always fast)
    │       function @ 89%, skills, gaps, jobs
    │
    ├──► LLM enhancer (runs only if API key present)
    │       │
    │       ├──► Low confidence fix (< 30%): LLM re-classifies
    │       ├──► Skill evidence: LLM extracts spans from resume
    │       ├──► Gap explanation: LLM writes human-readable guidance
    │       └──► Ready/Target split: LLM labels each job
    │
    └──► Merged output: classifier + LLM enrichments
```

## API Key

File: `.env` in project root
```
OPENROUTER_API_KEY=sk-or-v1-...
```

Loaded via `python-dotenv` or os.getenv. Not committed to git.

## Implementation

One new file: `recommender/llm/__init__.py`

```python
def enhance(analysis_result: dict, resume_text: str) -> dict:
    """Add LLM enrichments to pipeline result. No-op if no API key."""
    if not API_KEY:
        return analysis_result
    
    # 1. Fix classification if low confidence
    if analysis_result['match_pct'] < 30:
        better = llm_classify(resume_text)
        analysis_result['function'] = better['function']
        analysis_result['llm_reclassified'] = True
    
    # 2. Add skill evidence
    analysis_result['skill_evidence'] = llm_extract_evidence(
        resume_text, analysis_result['skills_extracted']
    )
    
    # 3. Add gap explanation
    analysis_result['gap_explanations'] = llm_explain_gaps(
        analysis_result['missing_skills'], analysis_result['skills_extracted']
    )
    
    # 4. Split jobs into ready/target
    analysis_result['ready_jobs'], analysis_result['target_jobs'] = llm_split_jobs(
        analysis_result['openings'], analysis_result['skills_extracted']
    )
    
    return analysis_result
```

## Cost

OpenRouter free tier: GPT-OSS-20B (free).  
1 resume = 1-3 API calls = ~2,000 tokens total = ~$0.001 or free.

## Fallback

If LLM fails or no API key: pipeline works exactly as today. LLM is pure enrichment — never required.
