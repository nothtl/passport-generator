# Architecture

The pipeline takes a resume and returns: best-fit career, extracted skills, market gaps, and real job openings. All data-driven, no hardcoded rules, no LLM.

## Example: Abigail Rodriguez

Abigail is a graphic designer and content creator. Here's what happens step by step.

---

### Stage 1: Function Classification

*What career is this person in?*

Three independent signals vote. The winner becomes the function.

```
Resume: "Abigail Rodriguez. Graphic Design & Content Creator. I write and
         create content. I train students and mentor..."

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Signal 1: ML     │  │ Signal 2: O*NET  │  │ Signal 3:        │
│ Classifier       │  │ Task Overlap     │  │ Embeddings       │
│                  │  │                  │  │                  │
│ Trained on 2,484 │  │ 18,796 govt job  │  │ all-MiniLM-L6-v2 │
│ real resumes     │  │ descriptions     │  │ semantic match   │
│                  │  │                  │  │ vs 923 occupat.  │
│ design     27%   │  │ education   30%  │  │ design     22%   │
│ education  24%   │  │ design     25%  │  │ arts-media 18%   │
│ arts-media 21%   │  │ ops        18%  │  │ education  15%   │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         └──────────┬──────────┴──────────┬──────────┘
                    │   2 of 3 vote for   │
                    │   "design"          │
                    ▼
         Function: design (27% confidence)
         Alternatives: education (24%), arts-media (21%)
```

**Data source**: `resume_classifier.pkl` (SVC-10K, trained on HuggingFace `opensporks/resumes`, 73.9% individual accuracy).

**Accuracy**: 89% top-1, 100% top-3 on 500 real resumes.

---

### Stage 2: Skill Extraction

*What skills does this resume show?*

The resume is tokenized into 1-3 word chunks. Each chunk is checked against a vocabulary of 67,000 terms extracted from real job postings. If it appears in real JDs, it's a skill.

```
Resume tokens:
  unigrams: "graphic", "design", "content", "creator", "write", ...
  bigrams:  "graphic design", "content creator", "google maps", ...
  trigrams: "graphic design content", "content creator write", ...

Normalize (strip hyphens/spaces):
  "graphic design" → "graphicdesign"
  "Google Maps"    → "googlemaps"

Check against 67K JD vocabulary:
  "graphicdesign"  → IN vocabulary (from JD "graphic-design") ✓
  "contentcreator" → not in vocabulary ✗
  "googlemaps"     → IN vocabulary (from JD "google-maps") ✓
  "train"          → not in vocabulary ✗

Result: 9 market-relevant skills
  ["content", "design", "events", "graphic design", "google maps",
   "maps", "program", "store", "students"]
```

**Data source**: `skill_vocabulary.json` (67K unique skill names from all JD parquet files). Built once at corpus creation time, loaded at startup.

**Speed**: ~2ms per resume.

---

### Stage 3: Market Gap Analysis

*What skills do real jobs demand that this person doesn't show?*

The function's JD corpus is loaded. Every unique skill in those JDs is counted. Skills the student has are matched. Skills they don't have are ranked by how many JDs demand them.

```
Function: design → design.parquet (1,044 real design JDs)

Top skills from design jobs:
  customerservice       (412 JDs)  ← Abigail doesn't have
  photoshop             (389 JDs)  ← missing
  adobecreativesuite    (356 JDs)  ← missing
  typography            (201 JDs)  ← missing
  layout                (187 JDs)  ← missing
  communication         (165 JDs)  ← missing

Abigail HAS:
  design                (298 JDs)  ✓
  graphic design        (245 JDs)  ✓
  maps                  ( 45 JDs)  ✓

Result:
  has    = ["design", "graphic design", "content", "maps", "google maps"]
  gaps   = ["customerservice", "photoshop", "adobecreativesuite",
            "typography", "layout", "communication", ...]
```

These gaps are actionable: "Add Photoshop, Illustrator, and typography to your resume — they appear in 389, 245, and 201 of the 1,044 design jobs in our database."

**Data source**: Per-function parquet files in `recommender/corpus/`. Built from open-jobs (68K+ US entry-level jobs) + O\*NET synthetic data.

---

### Stage 4: Job Recommendations

*What jobs are available right now?*

Each student skill is weighted by IDF (Inverse Document Frequency). Rare skills like "google maps" get high weight. Common skills like "design" get moderate weight. Each JD is scored by summing the IDF of matched skills. The top 5 are returned.

```
Skills → IDF weights:
  "graphic design"  IDF 3.2 (rare — only in 87 JDs)
  "maps"            IDF 4.5 (very rare — only in 45 JDs)
  "design"          IDF 1.8 (moderate — in 298 JDs)
  "content"         IDF 1.1 (common — in most JDs)

JD #142: "Junior Designer @ Abercrombie & Fitch"
  skills: [graphic-design, photoshop, illustrator, layout]
  match:  "graphic-design" → IDF 3.2
  score:  3.2

JD #89: "Designer @ Studio Co"
  skills: [design, typography, adobe, branding]
  match:  "design" → IDF 1.8
  score:  1.8

Synonyms expand matching (85K ESCO alt-labels):
  "graphic-design" = "graphic design"
  "customer care"  = "customer service"

Result: Top 5 openings
  1. Instructional Designer @ contract company
  2. Planning Technician @ Lakewood
  3. Junior Designer @ Abercrombie & Fitch
  4. Designer @ Studio Co
  5. Level I Designer @ agency
```

**Data source**: Same parquet files as Stage 3. IDF computed per function, cached after first load. ESCO synonyms from `esco_synonyms.json` (85K alt-labels, EU government data).

**Speed**: ~10ms per query after IDF cache warm.

---

## Data Flow Summary

```
Resume text
    │
    ├──► Stage 1: ensemble_matcher.py
    │       resume_classifier.pkl (ML model)
    │       onet_occ_index.json (O*NET tasks)
    │       all-MiniLM-L6-v2 (embeddings, cached)
    │       → function + confidence + alternatives
    │
    ├──► Stage 2: skill_extractor.py
    │       skill_vocabulary.json (67K JD skill names)
    │       → extracted skills
    │
    ├──► Stage 3: analyze.py (gap computation)
    │       {function}.parquet (JD corpus)
    │       → has_skills + market_gaps
    │
    └──► Stage 4: retriever.py
            {function}.parquet (JD corpus)
            esco_synonyms.json (85K alt-labels)
            → top 5 job openings
```

## Key Properties

| Property | How |
|---|---|
| **Zero hardcoded rules** | All filters are data (vocabulary, IDF, classifier) |
| **No LLM** | ML model + TF-IDF + embeddings, all local |
| **Fast** | ~50ms without embeddings, ~150ms with |
| **Self-improving** | Retrain on new data (`train_classifier.py`), rebuild corpus (`download.py`) |
| **US market** | O\*NET (government) + open-jobs (68K live postings) |
| **Fallback-safe** | 12 functions without dedicated data fall back to related functions |

## Limitations

| Limitation | Impact |
|---|---|
| **Cross-domain roles** | "Marketing Analyst at a hospital" → might classify as healthcare, not marketing |
| **Short resumes** | <50 words have insufficient signal for the classifier |
| **Function bias** | Functions with more training data (finance: 351, sales: 236) perform better than small ones (agriculture: 63, support: 22) |
| **JD quality** | Some parquets have misclassified JDs (line cooks in skilled-trade, bartenders in design) |
