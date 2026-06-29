# SpeakHire Recommender

Resume analysis pipeline — extract skills, match to careers, identify market gaps, and recommend real jobs. Production-ready, 92% accuracy, zero hardcoded rules.

```bash
python recommender/analyze.py "paste resume text here"
```

---

## What It Does

| Stage | Input | Output | Example |
|---|---|---|---|
| Classify | Resume text | Best-fit career + confidence | `technology @ 33%` |
| Extract | Resume text | Skills from 67K JD vocabulary | `computer vision, azure, full stack` |
| Infer | Skills + gaps | LLM deduces implicit skills | `python, git, linux` |
| Gaps | Skills + function | Market demands you lack | `sql, typescript, documentation` |
| Coach | Skills + gaps | AI career advice | *"Focus on SQL and testing first..."* |
| Recommend | Skills + function | Real job openings with fit scores | `AI Engineer @ InnovationTeam (85%)` |

**Accuracy**: 92% function top-1 on 500 real resumes.  
**Speed**: ~50ms core + optional LLM enrichment (1-3s).  
**Data**: O\*NET (US govt) + open-jobs (170K postings) + ESCO (EU taxonomy).

---

## Architecture

```
Resume → 4 signals vote → function + confidence
       ↓
       N-gram tokenizer → check 67K JD vocabulary → skills
       ↓
       IDF-weighted filter → market gaps
       ↓
       IDF-scored JD matching → top 5 jobs with URLs
       ↓
       LLM enrichment (optional, DeepSeek V4)
         → inferred skills, coach notes, job fit scores
```

### 4-Signal Ensemble

| Signal | What | Accuracy |
|---|---|---|
| ML Classifier | TF-IDF + LinearSVC, 2,484 resumes | 74% |
| O\*NET Tasks | TF-IDF vs 18,796 govt job descriptions | 39% |
| Sentence Embeddings | all-MiniLM-L6-v2 semantic match | — |
| LLM (tiebreaker) | DeepSeek V4, called when confidence < 30% | — |

**Expert voting**: 2/4 wins. LLM fires only when needed.

Full walkthrough: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Output

```json
{
  "function": "technology",
  "confidence": 33,
  "skills": ["computer vision", "azure", "full stack", ...],
  "inferred": ["python", "git", "linux"],
  "gaps": ["sql", "typescript", "documentation"],
  "related": [{"skill": "flask", "pmi": 4.4}, ...],
  "coach_notes": "Focus on SQL and testing first...",
  "jobs": [{
    "title": "AI Engineer",
    "company": "InnovationTeam",
    "url": "https://...",
    "fit": 85,
    "label": "ready",
    "why": "Direct alignment with AI/ML skills",
    "gaps": "No explicit OpenCV listed"
  }]
}
```

---

## Files

```
recommender/
  analyze.py                        ← main entry point
  extract/skill_extractor.py        ← n-gram + 67K JD vocabulary
  match/ensemble_matcher.py         ← 4-signal voting
  match/classifier_matcher.py       ← ML classifier
  retrieve/retriever.py             ← IDF-weighted JD matching
  llm/__init__.py                   ← LLM enrichment (optional)
  train_classifier.py               ← retrain on new data

  data/
    resume_classifier.pkl           ← ML model (5 MB)
    skill_vocabulary.json           ← 67K JD skill names
    esco_synonyms.json              ← 85K alt-labels
```

## Run

```bash
python recommender/analyze.py "resume text here"
python recommender/analyze.py resume.txt
```

## LLM (Optional)

Place API key in `api` file at project root. Falls back to rule-based coach notes without it. Uses DeepSeek V4.
