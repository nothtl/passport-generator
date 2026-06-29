# Architecture

The pipeline: resume → career match → skills → gaps → coach notes → jobs. All data-driven, no hardcoded rules.

## Example: Hu Tingli (AI Engineer)

```
Resume: "Technical Skills: Agentic AI, Computer Vision, Full Stack,
         Cloud, Autonomous Systems. SpeakHire Tech Intern. 
         Azure Functions, Cosmos DB. Hoverr CTO. YOLO, ROS2..."

                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 1: 4-Signal Ensemble Classification                   │
│                                                            │
│ Signal 1 (ML):     technology 38%, arts-media 13%          │
│ Signal 2 (O*NET):  technology 25%, education 14%           │
│ Signal 3 (Embed):  technology 7%, science 7%               │
│ Signal 4 (LLM):    technology 85% ← fires only when needed  │
│                                                            │
│ Expert vote: 4/4 technology → confidence 33%               │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 2: Skill Extraction                                  │
│                                                            │
│ N-gram tokenizer → normalize → check 67K JD vocabulary     │
│                                                            │
│ "computer vision" → "computervision" → IN vocab ✓          │
│ "azure functions" → "azurefunctions" → IN vocab ✓          │
│ "agentic ai"      → "agenticai"      → IN vocab ✓          │
│                                                            │
│ Result: 43 skills extracted                                │
│ Filter: keep multi-word + single words above median IDF    │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 3: Market Gaps + Inference                           │
│                                                            │
│ Load technology.parquet → 1,915 tech JDs                   │
│ Top demanded: python, sql, javascript, documentation       │
│                                                            │
│ LLM Inference: "You use FastAPI + YOLO + ROS2 → Python,    │
│ Git, Linux are implicit. Remove from gaps."                │
│                                                            │
│ Remaining gaps: sql, typescript, documentation             │
│ Inferred: python, git, linux, debugging (capped at 30%)    │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ STAGE 4: Coach Notes + Job Matching                        │
│                                                            │
│ Coach (LLM): "Focus on SQL and testing. With your FastAPI  │
│ and Azure skills, SQL opens up backend engineering roles." │
│                                                            │
│ Jobs (IDF-weighted):                                       │
│   [85%] AI Developer @ atlassand                           │
│     Why: Direct match with AI/ML, agentic AI skills        │
│     Gaps: No explicit Python listed (inferred)             │
│   [80%] Backend Python @ tehora                            │
│     Why: Strong match with Python, Azure, cloud            │
│     Gaps: No API design experience listed                  │
└────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Resume text
    │
    ├── ensemble_matcher.py
    │   ├── Signal 1: resume_classifier.pkl (ML, 74%)
    │   ├── Signal 2: onet_occ_index.json (O*NET tasks)
    │   ├── Signal 3: all-MiniLM-L6-v2 (embeddings)
    │   └── Signal 4: DeepSeek V4 (LLM, when confidence < 30%)
    │   → function + confidence
    │
    ├── skill_extractor.py
    │   └── skill_vocabulary.json (67K JD terms)
    │   → extracted skills (filtered by IDF)
    │
    ├── analyze.py (gap + inference)
    │   ├── {function}.parquet → market gaps
    │   └── LLM inference → inferred skills
    │   → gaps + inferred + coach notes
    │
    └── retriever.py
        ├── {function}.parquet → IDF-weighted scoring
        └── esco_synonyms.json → 85K alt-labels
        → top 5 jobs with URLs + fit scores
```

## Key Properties

| Property | How |
|---|---|
| Zero hardcoded rules | All filters are data (vocabulary, IDF, classifier, PMI) |
| LLM optional | Pipeline works without API key (rule-based coach fallback) |
| Self-improving | Retrain classifier, rebuild corpus |
| US market | O\*NET + open-jobs (170K postings) |
| Fallback-safe | LLM failures → rule-based coach, API timeouts → retry |
| Company UUID filter | Auto-detects and removes UUID company names |

## Limitations

| Limitation | Mitigation |
|---|---|
| Cross-domain roles | LLM Signal 4 reclassifies low-confidence cases |
| Short resumes (<50 words) | LLM signal compensates |
| PDF extraction quality | Use clean resume text, not full PDF dump |
| Embedding flakiness | SentenceTransformer can fail; pipeline degrades gracefully |
| API timeouts | Retry with backoff, rule-based coach fallback |
