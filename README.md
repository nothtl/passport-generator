# SpeakHire Recommender

Resume analysis pipeline — extract skills, match to career paths, identify market gaps, and recommend real job openings. Zero hardcoded rules, zero LLM calls, entirely data-driven.

Built for [SpeakHire](https://speakhire.org).

---

## Quick Start

```bash
python recommender/analyze.py "paste resume text here"
python recommender/analyze.py resume.txt
python recommender/analyze.py                    # interactive (Ctrl+Z to finish)
```

**Output**: best-fit function with confidence, extracted skills, market skill gaps (from real JDs), alternatives to explore, and top 5 job openings with companies.

---

## Pipeline

```
Resume text
    │
    ▼
┌──────────────────────────────────────────┐
│  STAGE 1: FUNCTION CLASSIFICATION          │
│                                            │
│  3-signal expert voting:                   │
│    Signal 1: ML Classifier (SVC, 73.9%)    │
│    Signal 2: O*NET Task Overlap            │
│    Signal 3: Sentence Embeddings           │
│                                            │
│  → technology @ 46%                        │
│  → alternatives: arts-media, agriculture   │
│                                            │
│  Accuracy: 89% top-1, 100% top-3           │
│  (500 real resumes)                        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  STAGE 2: SKILL EXTRACTION                │
│                                            │
│  N-gram tokenizer (1-3 words)             │
│  → normalize (strip hyphens/spaces)       │
│  → check against 67K JD skill vocabulary  │
│  → return matching skills                 │
│                                            │
│  ~2ms. Vocabulary from real job postings. │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  STAGE 3: MARKET GAP ANALYSIS             │
│                                            │
│  Filter skills to JD vocabulary           │
│  Find missing: JD skills student lacks     │
│  Sort by document frequency (most common)  │
│                                            │
│  Has:    python, aws, docker, react        │
│  Missing: sql, typescript, git, kubernetes │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  STAGE 4: JOB RECOMMENDATIONS             │
│                                            │
│  IDF-weight each skill                    │
│  Score JDs by weighted skill overlap      │
│  Expand via 85K ESCO synonyms             │
│                                            │
│  → ML Computer Vision Intern @ Syntiant   │
│  → AI Software Engineer Intern @ Intel    │
│  → Robotics Co-op @ AeroVect             │
└──────────────────────────────────────────┘
```

---

## Architecture

```
recommender/
  analyze.py                        ← python recommender/analyze.py "resume..."
  mcp_server.py                     ← MCP server (6 tools for chatbot)
  train_classifier.py               ← retrain ML model on new data

  extract/
    skill_extractor.py              ← n-gram + JD vocabulary matching

  match/
    classifier_matcher.py           ← ML classifier (SVC-10K, 2,484 resumes)
    ensemble_matcher.py             ← 3-signal expert voting
    onet_matcher.py                 ← O*NET keyword fallback

  retrieve/
    retriever.py                    ← IDF-weighted JD matching + ESCO synonyms

  corpus/
    download.py                     ← build corpus from O*NET + open-jobs

  data/
    resume_classifier.pkl           ← trained model (5 MB)
    skill_vocabulary.json           ← 67K JD skill names
    classifier_skills.json          ← top features per function
    esco_synonyms.json              ← 85K alt-labels
    onet_occ_index.json             ← O*NET occupation index
    resume_classifier_classes.json  ← function labels
```

**12 source files. No hardcoded patterns. No LLM. All local.**

---

## MCP Server

The pipeline is also exposed as an MCP server for chatbot integration:

```bash
python -m recommender.mcp_server
```

Tools: `extract_skills`, `analyze_resume`, `search_jobs`, `rebuild_corpus`

Configured via `.mcp.json` in project root.

---

## Accuracy

Measured on 2,484 real resumes from HuggingFace (`opensporks/resumes`).

| Metric | Score |
|---|---|
| Function (top-1) | 89% |
| Function (top-3) | 100% |
| Job relevance | 87% |

---

## Training

The ML classifier is trained on 2,484 real resumes. To retrain:

```bash
python recommender/train_classifier.py
```

This generates a new `resume_classifier.pkl`. The training data is downloaded from HuggingFace automatically.

---

## Corpus

The job corpus is built from O\*NET (US Department of Labor) + open-jobs (68K+ live US job postings). To rebuild:

```bash
python -m recommender.corpus.download
```

Parquet files are stored in `recommender/corpus/` (gitignored). Set `OPEN_JOBS_SKIP=1` for O\*NET-only (15 rows, instant).

---

## Data Sources

| Data | Source | Size |
|---|---|---|
| ML training | HuggingFace `opensporks/resumes` | 2,484 resumes |
| JD corpus | O\*NET + open-jobs | 19 parquet files |
| Skill vocabulary | JD corpus extraction | 67K unique skills |
| Synonyms | ESCO (EU taxonomy) | 85K alt-labels |
| Occupation tasks | O\*NET Task Statements | 18,796 statements |
| Sentence embeddings | `all-MiniLM-L6-v2` | 80 MB, cached locally |
