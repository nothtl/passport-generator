# SpeakHire Recommender

Resume analysis pipeline for [SpeakHire](https://speakhire.org) — extract skills, match to careers, identify market gaps, and recommend real job openings.

```bash
python recommender/analyze.py "paste resume text here"
```

---

## What It Does

| Stage | Input | Output | Example |
|---|---|---|---|
| 1. Classify | Resume text | Best-fit career + confidence | `design @ 27%` |
| 2. Extract | Resume text | Skills found in real JDs | `graphic design, google maps, content` |
| 3. Gaps | Skills + function | What the market demands that you lack | `photoshop, adobe suite, typography` |
| 4. Recommend | Skills + function | Real job openings | `Junior Designer @ Abercrombie` |

**Accuracy**: 89% function top-1, 100% top-3 (500 real resumes).  
**Speed**: ~50ms per resume.  
**Data**: O\*NET (US govt) + open-jobs (68K live postings) + ESCO (EU skills taxonomy).  

Zero hardcoded rules. No LLM. All local.

---

## Architecture

```
Resume text
    │
    ├──► Signal 1: ML Classifier (SVC, 2,484 resumes)
    ├──► Signal 2: O*NET Task Overlap (18,796 govt job descriptions)
    ├──► Signal 3: Sentence Embeddings (semantic similarity)
    │         │
    │         ▼  Expert voting (2/3 wins)
    │     Function: "design"
    │
    ├──► N-gram tokenizer → check against 67K JD vocabulary
    │     Skills: ["graphic design", "google maps", "content", ...]
    │
    ├──► Load design.parquet (1,044 real JDs) → compare
    │     Gaps: ["photoshop", "adobe suite", "typography", ...]
    │
    └──► IDF-weight skills → score each JD → return top 5
          Jobs: ["Junior Designer @ Abercrombie", ...]
```

Full walkthrough with example data: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Files

```
recommender/
  analyze.py                        ← python recommender/analyze.py "resume..."
  mcp_server.py                     ← MCP server (6 tools for chatbot)
  train_classifier.py               ← retrain ML model

  extract/skill_extractor.py        ← n-gram + JD vocabulary
  match/classifier_matcher.py       ← ML classifier
  match/ensemble_matcher.py         ← 3-signal voting
  retrieve/retriever.py             ← IDF-weighted JD matching + synonyms

  data/
    resume_classifier.pkl           ← trained model (5 MB)
    skill_vocabulary.json           ← 67K JD skill names
    esco_synonyms.json              ← 85K alternative labels
    classifier_skills.json          ← top features per function
```

12 source files. All data-driven.

---

## Run

```bash
# Paste a resume
python recommender/analyze.py "your resume text here"

# From a file
python recommender/analyze.py resume.txt

# Interactive
python recommender/analyze.py
```

## MCP Server

```bash
python -m recommender.mcp_server
```

Tools: `analyze_resume`, `extract_skills`, `search_jobs`, `rebuild_corpus`.

## Retrain

```bash
python recommender/train_classifier.py
```

## Rebuild Corpus

```bash
python -m recommender.corpus.download
```

Set `OPEN_JOBS_SKIP=1` for O\*NET-only (15 rows, instant).
