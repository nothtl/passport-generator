# SpeakHire Passport Agent

An AI pipeline that takes a student's resume, LinkedIn profile, and program survey responses, then outputs a calibrated 6-pillar competency score, a ranked list of matching job opportunities, and concrete next steps to close skill gaps.

Built for [SpeakHire](https://speakhire.org) — a career readiness program serving high school students through internship rounds, mentorship, and professional skill-building.

---

## What This Does

```
Student uploads resume + LinkedIn + surveys
                │
                ▼
    ┌───────────────────────┐
    │  Evidence Extraction   │
    │  rules → local → api   │
    │  3-tier semantic match │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  6-Pillar Scoring      │
    │  EC·GC·RFF·CR·CT·CI   │
    │  Deterministic, 0-100  │
    │  Zero LLM, reproducible │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Opportunity Matching  │
    │  Hull filter → rank    │
    │  → LLM explain top-5  │
    │  (open-jobs dataset,   │
    │   967K live roles)     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Student Output        │
    │  Scores + matches      │
    │  + gaps + next steps   │
    │  Self-contained HTML   │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Feedback Loop         │
    │  Outcomes → tune       │
    │  Clicks → re-rank      │
    │  Judgments → distill   │
    └───────────────────────┘
```

**Three outputs per student:**

| Output | What it is | Example |
|--------|-----------|---------|
| **PathCredits Score** | 0–100 composite across 6 competencies, with percentile rank and sub-score breakdown | `62.5/100 (65th percentile)` |
| **Top Opportunities** | Ranked list of real, currently-open jobs with per-role explanation | `Content Creator Intern @ Khan Academy — 82% match` |
| **Next Steps** | Concrete, actionable advice to close specific gaps, prioritized by expected score impact | `"Add your database project with specific SQL keywords (+5 CR expected)"` |

---

## The 6 Pillars

Each pillar is scored deterministically — no LLM is used in the scoring path. Scoring formulas are in [`passport_agent_v2/pillars/`](passport_agent_v2/pillars/).

| Pillar | What It Measures | Scoring Inputs |
|--------|-----------------|----------------|
| **EC** — Effective Communication | Verbal, written, interpersonal, and cross-cultural communication | English level, survey responses, evidence of teaching/mentoring/multilingual work |
| **GC** — Global Citizenship | Empathy, community connectedness, cultural values, volunteering | Pre/post empathy surveys, community feel, volunteer hours, cultural self-assessment |
| **RFF** — Ready for Future | Self-reflection, goal-setting, career clarity, college preparation | SMART goals, career aspiration text, speaker/topic inspiration measures |
| **CR** — Career Readiness | Internships, jobs, volunteering, session commitment, resume quality | Volunteer/internship/job history, session attendance ratio, resume length and keyword density |
| **CT** — Critical Thinking | Analytical depth, academic rigor, teaching translation, coordination | Evidence facts classified as analytical, academic, teaching, or coordination roles |
| **CI** — Creative Innovation | Origination, initiative, applied innovation, creative expression | Evidence facts classified as building, founding, creating, or expressing |

Each pillar outputs a `score` (0–100), `sub_scores` (dimensional breakdown), `source` (survey / docs / deterministic), a depth signal (`deep` / `developing` / `surface` / `pioneering` / `conventional`), and a human-readable `reasoning` string.

The **PathCredits Score** is the equal-weight mean of all six pillars. This can be tuned to a learned weighted mean once outcome data is collected (see [Feedback Loops](#feedback-loops--continuous-learning)).

---

## Architecture

### Stage-by-Stage Walkthrough

The pipeline has 8 stages. Stages 1–4 are built and running. Stages 5–8 are proposed for the opportunity matching integration.

---

#### STAGE 0 — Raw Inputs

Three pieces of information enter the system for each student:

| Input | Format | Example Content |
|-------|--------|----------------|
| Resume | PDF | Work history, skills, education — typically 150–1,200 words |
| LinkedIn Profile | Markdown | Structured roles with date ranges, summary, skills |
| Survey Responses | JSON (from Agent 1) | 42–80 fields from SpeakHire's database and exit surveys |

Not every student has all three. The system degrades gracefully — a student with only survey data still gets scored, just at lower confidence with `needs_review: true`.

---

#### STAGE 1 — Ingest

**Code:** [`tools/ingest.py`](passport_agent_v2/tools/ingest.py)

```
resume.pdf ──────► pdfminer → resume_text (string)
linkedin.md ─────► read file → linkedin_text (string)
raw_data.json ───► parse JSON → raw_fields (dict, 42–80 fields)
                              → missing_fields (list, field names)
```

- Fuzzy-matches student names to folder paths (handles typos, partial names, aliases like "Alex Aquino" → "Khristal Aquino Cruz")
- Extracts PDF text via `pdfminer.high_level.extract_text()`
- Returns a `StudentBundle` — all raw data in one object. Nothing is scored yet.

**What comes out:**
```
StudentBundle
├── student_name: "Abigail Rodriguez"
├── email: "abigailrodriguez@crotonaihs.org"
├── slug: "abigail_rodriguez"
├── raw_fields: { "English - Spoken": "Somewhat Comfortable", ...42 total }
├── missing_fields: ["Community Feel (Quant)", "Pre Empathy", ...38 total]
├── resume_text: "Abigail Rodriguez\nBronx, NY\n\nSUMMARY: I am a motivated..."
└── linkedin_text: "# Abigail Rodriguez\n\n**Student at City College**..."
```

---

#### STAGE 2 — Normalize

**Code:** [`tools/normalization.py`](passport_agent_v2/tools/normalization.py)

This is the most important stage. It converts raw text into typed, machine-readable evidence.

**Step 2A — Survey fields pass through cleanly**

Each survey answer becomes a `NormalizedField` at confidence 1.0:

```
"English - Spoken"  → { value: "Somewhat Comfortable", confidence: 1.0, source: "survey" }
"FY1 - Volunteered" → { value: "True",                 confidence: 1.0, source: "survey" }
```

**Step 2B — Document text → Evidence Facts**

Each paragraph from the resume and LinkedIn runs through `classify_evidence_span()`, which checks it against ~200 keyword patterns (in priority order) to assign five type tags:

| Dimension | Example Values | What It Means |
|-----------|---------------|---------------|
| `action_type` | `build_create`, `teach_support`, `coordinate_manage`, `analyze_apply`, `create_express`, `found_initiate`, `language_bridge` | What kind of activity? |
| `artifact_type` | `software_website`, `technical_project`, `creative_output`, `teaching_activity`, `coordination_role`, `analytical_project`, `volunteer_activity`, `founding_activity`, `bilingual_support` | What did they produce? |
| `context_type` | `general`, `community_service`, `community_nonprofit` | In what setting? |
| `initiative_level` | `leadership`, `project_execution`, `self_directed`, `service_support` | How much ownership? |
| `evidence_strength` | `direct`, `indirect` | How concrete is the evidence? |

The classifier is **ordered** — earlier patterns have priority. If a span matches both `"mentor"` (teaching_activity) and `"program manager"` (coordination_role), the first match wins. This means classification is deterministic and inspectable.

Example — a real span from Abigail Rodriguez's LinkedIn:

```
INPUT:  "Intern — My Bodega Online — Built a database of EBT stores
         across NYC metro area using Google Maps"

MATCHES: "Built" → build_create → checks for software terms →
         "database" found → artifact_type = "technical_project"
         → confidence = 0.90

OUTPUT: NormalizedEvidenceFact
        ├── action_type: "build_create"
        ├── artifact_type: "technical_project"
        ├── initiative_level: "project_execution"
        ├── relevant_pillars: ["CI", "CR", "CT"]
        └── confidence: 0.90
```

A typical student produces 20–50 evidence facts.

**Step 2C — Backfill missing survey fields**

For the 30–40 fields a student didn't answer, the normalizer infers values from evidence facts:

```
Missing: "Languages"           → scans docs for language names → "English, Spanish"
Missing: "FY1 Ever Volunteered" → checks for volunteer context  → "True" (conf 0.75)
Missing: "Community Feel"      → counts community facts        → 3 (conf 0.70)
Missing: "Know How To Pursue"  → counts coordination facts     → 7 (conf 0.65)
```

**Step 2D — Conservative defaults for what remains**

Unresolved fields get conservative (low/neutral) defaults marked `conservative: true, confidence: 0.40–0.45`:

```
"The Speaker inspired me..."  → 4.0 (midpoint, 1-7 scale)
"I made new friends"          → "False"
"I feel more engaged"         → "False"
```

**What comes out:**
```
NormalizedStudentProfile
├── normalized_fields: 74 fields total
│   ├── 42 from survey (confidence 1.0)
│   ├── 24 inferred from documents (confidence 0.55–0.80, was_missing: true)
│   └── 8 conservative defaults (confidence 0.40–0.45, conservative: true)
├── evidence_facts: 38 typed facts with 5-dimension tags each
├── unresolved_fields: 2 fields still missing
└── normalization_summary: { fields_from_survey: 42, fields_from_docs: 24, ... }
```

---

#### STAGE 3 — Score

**Code:** [`pillars/ec.py`](passport_agent_v2/pillars/ec.py), [`pillars/gc.py`](passport_agent_v2/pillars/gc.py), etc.

Six pure-math formulas. Each reads `normalized_fields` (and some read `evidence_facts`), applies weighted sub-dimensions, and outputs 0–100. Zero LLM calls. Same input always gives same output.

**EC example — 4 sub-dimensions:**

```
VERBAL (40%):     English level + Champion rating + Community Feel + text quality
WRITTEN (25%):    Open-ended response quality (word count + keyword signals)
INTERPERSONAL (20%): Listen score + Conflict score + bonus for high conflict mgmt
CROSS-CULTURAL (15%): Inclusion score + cultural understanding + multilingual bonus

EC = Verbal×0.40 + Written×0.25 + Interpersonal×0.20 + CrossCultural×0.15
   = 19.58 + 10.63 + 10.00 + 14.00
   = 54.21  → rounded to final score
```

**CT example — counts evidence facts by type:**

```
analytical_depth:   facts with artifact IN (analytical, technical, software)
                  → banded_score(count, [(0,5),(1,12),(2,17),(3,21),(4,25)])
academic_rigor:    mentions of "degree","gpa","honors","education" in docs
teaching:          facts with artifact IN (teaching_activity, bilingual_support)
coordination:      facts with artifact = coordination_role

CT = sum of four banded sub-scores
```

**What comes out — 6 pillar results:**
```
EC:  { score: 52, sub_scores: {Verbal: 19.6, Written: 10.6, ...}, source: "survey+docs" }
GC:  { score: 54, sub_scores: {D1_Empathy: 0.55, ...}, source: "survey+docs" }
RFF: { score: 63, sub_scores: {D1_SelfReflection: 0.37, ...}, source: "survey+docs" }
CR:  { score: 63, sub_scores: {C1: 50, C2: 76, C3: 85, C4: 100}, source: "survey+docs" }
CT:  { score: 83, depth_signal: "deep", source: "survey+docs" }
CI:  { score: 60, innovation_signal: "developing", source: "survey+docs" }
```

---

#### STAGE 4 — Rank

**Code:** [`tools/ranking.py`](passport_agent_v2/tools/ranking.py)

```
PathCredits_Score = mean(EC, GC, RFF, CR, CT, CI)
                  = (52 + 54 + 63 + 63 + 83 + 60) / 6
                  = 62.5

In cohort of 20 students → rank 7 → 65th percentile
```

Outputs JSON per student under `outputs/`. Batch mode produces ranked cohort summaries.

---

#### STAGE 5 — Hull Filter (Proposed)

Adapted from [`elliottdehn/open-jobs`](https://github.com/elliottdehn/open-jobs) `hull.py`.

```
Student profile     ┌──────────────────────────────────┐
────────────────────►           HULL FILTER             │
                    │                                   │
                    │  Streams 967K jobs, 21 GB Parquet │
                    │  Structured columns only           │
                    │  (no embeddings loaded — fast)     │
                    │                                   │
                    │  Hard eligibility filters:         │
                    │  • function = student's field      │
                    │  • level = Entry, Intern          │
                    │  • country = US                    │
                    │  • min_comp = 0 (keep unpaid)     │
                    │                                   │
                    │  Lexical recall:                   │
                    │  • title + alt_titles substring    │
                    │    match against skills[]          │
                    │  • +56% recall vs embeddings alone │
                    │                                   │
                    │  Deduplicate by (company, title)   │
                    │  (same role cross-posted to        │
                    │   multiple ATS)                    │
                    └──────────────────────────────────┘
                                       │
                                       ▼
                              ~1M → ~1,500 candidates
                              in ~15 seconds on CPU
```

The hull is intentionally **loose** — it eliminates only the 99.8% of jobs that are wrong function, wrong level, or wrong country. Everything in the hull is at least plausible. Ranking happens next.

---

#### STAGE 6 — Semantic Rank (Proposed)

Adapted from `rank.py`.

```
hull.json           ┌──────────────────────────────────┐
(1,500 jobs) ──────►│         SEMANTIC RANK            │
                    │                                   │
                    │  For each job:                     │
                    │  1. Embed JD text (all-MiniLM-L6) │
                    │  2. Build student query embedding: │
                    │     evidence facts text +          │
                    │     6-pillar summary               │
                    │  3. Cosine similarity              │
                    │  4. Structured bonus signals:      │
                    │     + skill Jaccard overlap        │
                    │     + level match (Entry/Intern)  │
                    │     + work_mode match              │
                    │     + artifact-type resonance      │
                    │     (e.g. technical_project +      │
                    │      engineering function = +0.10) │
                    │                                   │
                    │  Sort by composite score           │
                    └──────────────────────────────────┘
                                       │
                                       ▼
                              top ~50 matches
```

---

#### STAGE 7 — LLM Explanation (Proposed)

Adapted from `match.py`.

```
top 5 matches       ┌──────────────────────────────────┐
────────────────────►│         LLM EXPLAIN              │
                    │                                   │
                    │  5 concurrent calls, one per job   │
                    │  ThreadPoolExecutor(workers=5)     │
                    │                                   │
                    │  Prompt includes:                  │
                    │  • 6-pillar scores + depth signals │
                    │  • Key evidence facts              │
                    │  • Full JD excerpt (truncated)     │
                    │  • Structured fields (skills,     │
                    │    level, function, comp)          │
                    │                                   │
                    │  Returns JSON Schema-validated:    │
                    │  • match_score (0-100, calibrated) │
                    │  • verdict (strong|solid|stretch|weak)│
                    │  • strengths (evidence→job links)  │
                    │  • gaps (missing requirements)     │
                    │  • next_steps (actionable advice)  │
                    │                                   │
                    │  Resumable via JSONL sidecar       │
                    │  Thread-safe writes with lock      │
                    │  Killed run → continues where left │
                    └──────────────────────────────────┘
                                       │
                                       ▼
                          Per-job: score + verdict
                          + strengths + gaps + next_steps
```

Each next-step is specific and actionable: not "improve your resume" but "add your My Bodega Online database project with the words 'SQL' and 'data pipeline' — this directly addresses the data requirement in the Stripe role."

---

#### STAGE 8 — Output & Delivery

Two formats:

1. **JSON** — full machine-readable output under `outputs/{slug}.json`
2. **Self-contained HTML** — a single file with embedded data and vanilla JS for search, sort, radar chart, and apply links. No server, no login, no dependencies. A student can download their passport and open it anywhere.

### Semantic Matching Layer (Built)

The three-tier evidence classifier in [`tools/semantic.py`](passport_agent_v2/tools/semantic.py):

| Tier | Mode | How It Works | Cost |
|------|------|-------------|------|
| **1. Rules** | `semantic_mode=rules` | Deterministic keyword and pattern matching against 200+ phrases across 12 artifact types. No external dependencies. Classifies ~80% of spans. | $0 |
| **2. Local Retrieval** | `semantic_mode=local` | TF-IDF cosine similarity (default, deterministic) or `all-MiniLM-L6-v2` + ChromaDB against a curated concept library of labeled exemplars. Runs entirely on CPU. | $0 |
| **3. API Fallback** | `semantic_mode=api` | For spans still unresolved after local retrieval, calls OpenRouter API with a taxonomy-bounded JSON classifier. Validated against top local candidate — API result discarded if they disagree. Cache-backed — identical spans never call the API twice. | ~$0.01/student |

A span is only promoted to higher tiers if the lower tier returns `artifact_type == "unknown"` or a generic match that could benefit from disambiguation.

---

### Pairwise Distillation Pipeline (Proposed)

> Adapted from `langsort.py` + `btrank.py` in open-jobs. This is the single highest-impact piece not yet built.

The core problem: embedding similarity alone achieves ~68% accuracy on held-out preference judgments. A distilled model achieves ~90%. The difference is a **learn-once, apply-forever** linear model trained on pairwise comparisons.

**How it works:**

```
PHASE 1: Collect Judgments
──────────────────────────
  Staff or LLM compares pairs of (student, job):
  "For Abigail, which fits better?
   A: Content Creator @ Khan Academy
   B: Teaching Fellow @ Teach For America"
                               → Answer: A

  Only compares pairs NOT already transitively implied
  (gating recovers ~30-40% of budget from redundant comparisons)
  ~200 comparisons needed to saturate model accuracy


PHASE 2: Build Partial Order
─────────────────────────────
  Graph: Abigail×Khan > Abigail×TFA
         Abigail×TFA > Abigail×NYC-Ed
  Transitive closure: Abigail×Khan > Abigail×NYC-Ed (implied, no call needed)

  Topological sort honors every explicit decision
  Incomparable pairs remain as gaps to fill


PHASE 3: Distill into Linear Model
───────────────────────────────────
  For each compared pair:
    X = embedding(job_A) - embedding(job_B)
    y = 1 if A was preferred, 0 if B was preferred

  Fit logistic regression:
    P(A preferred over B) = sigmoid(w · (emb_A - emb_B))

  Save as taste.npz (a few KB)
  This model now scores ANY student against ANY job at zero LLM cost


PHASE 4: Apply to Corpus
─────────────────────────
  For each new (student, job) pair:
    score = w · job_embedding + student_bias

  Generalizes to future daily snapshots of job data
  Held-out accuracy: ~90% (vs ~68% embedding-only)
  Cost per student after training: $0
```

**Why pairwise beats absolute scoring:** LLMs bunch everything near 85 on 0–100 scales. Asking "which of these two is better?" produces much sharper discrimination. The Bradley-Terry model then aggregates thousands of binary choices into a smooth ranking without the bunching problem.

**Three ranking methods available from btrank.py:**

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| `gold` | Honor every decision as ground truth. Topological sort. Model only fills truly unknown pairs. | Staff judgments (trust the professionals) |
| `fuse` | Blend model scores with decisions. Softer — allows model to override weak or conflicting decisions. | LLM judgments (LLMs are fallible) |
| `bt` | Pure Bradley-Terry from win/loss counts. No embeddings used. | Lots of judgments but poor embeddings |

**Gating pattern:** Only compare pairs where the relationship is NOT already transitively implied. If A > B and B > C are known, then A > C is already determined — don't waste a comparison on it. This recovers ~30–40% of the judgment budget for genuinely new information.

---

### Daily Refresh Loop (Proposed)

Adapted from `download.py` + `stream.py`.

```
Cron: daily at 3am
──────────────────
  1. download.py → fetch latest open-jobs.parquet (resumable, 21 GB)
  2. For each active student: hull.py → rebuild filtered hull
  3. For each student: apply distilled taste.npz model → re-rank
  4. Students wake up to fresh recommendations daily
  5. New listings appear automatically — no manual curation

Cost: $0/day (model inference is CPU-only, dataset is CC0)
```

---

### Feedback Loops & Continuous Learning

The pipeline is designed to improve from every interaction. Six feedback mechanisms, ordered by implementation complexity:

---

#### Level 1 — Outcome-Based Pillar Weight Tuning

**Signal:** Did the student get an internship? Enroll in college? Graduate the program?

**Mechanism:** Replace the equal-weight PathCredits mean with learned weights.

```python
# After collecting outcomes for 100–200+ students:
# y = 1 if internship_placed, 0 if not
# X = [EC, GC, RFF, CR, CT, CI] for each student

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)

# model.coef_ tells you which pillars actually predict outcomes
# e.g., CR is 4× more predictive than GC → weight it higher

weights = model.coef_[0] / model.coef_[0].sum()
pathcredits = np.dot(pillar_scores, weights)
```

**What you need:** Outcome data already in SpeakHire's database — internship placement, college enrollment, program graduation.

---

#### Level 2 — Pairwise Preference Distillation

**Signal:** Staff (or LLM) pairwise comparisons: "For this student, Job A > Job B."

**Mechanism:** The full langsort.py → btrank.py pipeline described above. Train a logistic regression model on embedding differences that scores all future pairs at $0 cost.

**Accuracy gain:** ~68% (embedding-only) → ~90% (distilled model) on held-out decisions.

**What you need:** ~200 pairwise comparisons (15 minutes of staff time, or ~$2 in LLM costs).

---

#### Level 3 — Click-Through Relevance Feedback

**Signal:** Student actions on recommendations — applied, saved, clicked, ignored, dismissed.

**Mechanism:** Online learning on the embedding weight vector.

```python
# Implicit relevance labels:
#   applied = +1.0, saved = +0.5, clicked = +0.2
#   ignored = -0.1, dismissed = -0.5

# Online update:
# w_new = w_old + learning_rate * (actual - predicted) * embedding_diff
```

Each student interaction nudges the ranking weights. Over hundreds of students, the system learns which embedding dimensions predict real interest vs passive browsing.

---

#### Level 4 — Gap Resolution Impact Tracking

**Signal:** When a student follows advice ("add SQL keywords"), how much do their scores improve?

**Mechanism:** Track before/after score deltas per advice category.

```python
# After student updates resume based on advice:
# Before: CR=63, CI=60
# After:  CR=68, CI=66  (+5, +6)

# Aggregate across students:
gap_effectiveness = {
    "add_technical_keywords":  +5.2,   # biggest impact — suggest first
    "add_leadership_example":  +3.8,
    "add_volunteer_hours":     +2.1,
    "add_subject_focus":       +1.4,   # smallest impact — suggest last
}

# Prioritize advice by expected score improvement
```

---

#### Level 5 — Explanation Quality Rating

**Signal:** Student or staff ratings on explanation quality (★★★★☆).

**Mechanism:** A/B test prompt template variations. Track which explanation structure correlates with higher ratings. Feed high-rated explanation patterns back as few-shot examples in the match.py prompt.

---

#### Level 6 — Pillar Calibration Drift Detection

**Signal:** Score distributions across cohorts over time.

**Mechanism:** Weekly Kolmogorov-Smirnov test against historical baselines.

```python
# Run weekly:
for pillar in PILLARS:
    ks_stat, p_value = ks_2samp(current_scores[pillar],
                                 historical_baseline[pillar])
    if p_value < 0.01:
        alert(f"{pillar} drifted — possible causes:
               survey changed, new population, Champion rating shift")
```

Catches silent calibration drift before it affects ranking quality.

---

### Learning Architecture Summary

```
                    ┌────────────────────────────┐
                    │     FEEDBACK SIGNALS        │
                    │                             │
  Outcomes ─────────┤ internship_placed: 1/0     │──► logistic regression
  (from DB)         │ college_enrolled: 1/0      │    → pillar weights
                    │                             │
  Staff judgments ──┤ "Abigail×JobA > JobB"      │──► Bradley-Terry distill
  (pairwise)        │ (200 comparisons)           │    → embedding weights
                    │                             │
  Student actions ──┤ applied / saved / ignored  │──► online learning
  (click-through)   │ (per recommendation)       │    → re-rank weights
                    │                             │
  Gap resolution ───┤ before/after score deltas  │──► prioritize advice
  (resume updates)  │ (per advice item)          │    → highest-impact first
                    │                             │
  Ratings ──────────┤ ★★★★☆ on explanations     │──► A/B test prompts
  (explicit)        │ (per recommendation)       │    → best template wins
                    │                             │
  Score drift ──────┤ weekly KS test             │──► alert if p < 0.01
  (monitoring)      │ vs historical baseline     │    → recalibration needed
                    └────────────────────────────┘
```

---

## Directory Structure

```
passport generator/
├── README.md                          ← this file
├── Final_10.pdf                       ← SpeakHire evaluation framework doc
├── passport_agent_v2/                 ← current production pipeline
│   ├── main.py                        ← CLI entry point (single + batch)
│   ├── compare.py                     ← old-vs-new pipeline comparison
│   ├── batch_compare.py               ← batch comparison runner
│   ├── accuracy_audit.py              ← delta audit across cohorts
│   ├── calibration_report.py          ← score distribution reports
│   ├── pillars/                       ← deterministic scoring formulas
│   │   ├── formulas.py                ← shared normalizers and helpers
│   │   ├── ec.py                      ← Effective Communication
│   │   ├── gc.py                      ← Global Citizenship
│   │   ├── rff.py                     ← Ready for Future
│   │   ├── cr.py                      ← Career Readiness
│   │   ├── ct.py                      ← Critical Thinking
│   │   └── ci.py                      ← Creative Innovation
│   ├── tools/                         ← pipeline infrastructure
│   │   ├── ingest.py                  ← fixture loading, PDF extraction, fuzzy name matching
│   │   ├── normalization.py           ← evidence span classification, field backfilling
│   │   ├── semantic.py                ← 3-tier semantic matcher (rules/local/api)
│   │   ├── ranking.py                 ← PathCredits aggregation + cohort ranking
│   │   └── paths.py                   ← filesystem path constants
│   ├── models/
│   │   └── contracts.py               ← Pydantic models (NormalizedEvidenceFact, etc.)
│   ├── semantic/
│   │   └── concept_library.json       ← curated taxonomy of concept exemplars
│   ├── outputs/                       ← per-student JSON + batch summaries
│   ├── tests/                         ← pytest suite
│   └── synthetic_fixtures/            ← synthetic student data for testing
│       ├── agent1/outputs/            ← 6 synthetic student raw data files
│       └── student_data/              ← synthetic resumes + LinkedIn profiles
└── Passport_Agent_Actual_Test_Final_1/ ← original 5-agent pipeline (comparison baseline)
    └── Passport_Agent_Actual_Test/
        └── Passport_Agent_Actual/
            ├── agent1/                ← data loader (survey extraction)
            ├── agent2/                ← survey scorer
            ├── agent3/                ← document parser (resume + LinkedIn)
            ├── agent4/                ← LLM enrichment (Gemini, 22 calls/student)
            ├── agent5/                ← PDF report renderer
            ├── student_data/          ← 20 real student resumes + LinkedIn profiles
            └── run_pipeline.py        ← 5-agent orchestrator
```

---

## Quick Start

### Prerequisites

```bash
pip install pdfminer.six sentence-transformers chromadb scikit-learn numpy pydantic
```

For the API fallback tier, set your OpenRouter key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Score a Single Student

```bash
cd passport_agent_v2
python -m passport_agent_v2.main --student "Abigail Rodriguez"
```

Output: `outputs/abigail_rodriguez.json`

### Batch Score All Students

```bash
# Real students only, deterministic rules mode
python -m passport_agent_v2.main --batch all --rank-cohort

# With semantic retrieval for richer evidence extraction
python -m passport_agent_v2.main --batch all --semantic-mode local --rank-cohort

# Full API-augmented mode (cache-backed, costs ~$0.05 total)
python -m passport_agent_v2.main --batch all --semantic-mode api --rank-cohort
```

### Compare Old vs New Pipeline

```bash
# Single student
python -m passport_agent_v2.compare "Abigail Rodriguez"

# Batch comparison
python -m passport_agent_v2.batch_compare --batch-group real --semantic-mode rules
```

### Quality Assurance

```bash
# Accuracy audit (score deltas across pipelines)
python -m passport_agent_v2.accuracy_audit --batch-group real --semantic-mode rules

# Calibration report (score distributions, coverage)
python -m passport_agent_v2.calibration_report --batch-group real --semantic-mode rules

# Determinism test (same input → same output, 5 runs)
python -m pytest passport_agent_v2/tests/test_pipeline_output.py -v

# Semantic retrieval tests
python -m pytest passport_agent_v2/tests/test_semantic_retrieval.py -v
```

---

## CLI Reference

### `main.py`

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--student` | string | `"Abigail Rodriguez"` | Student name (fuzzy-matched to fixture folders) |
| `--batch` | `all` or integer | `0` (disabled) | Batch-process N students or all |
| `--batch-group` | `real`, `synthetic`, `merged` | `real` | Which fixture cohort to use |
| `--semantic-mode` | `rules`, `local`, `api` | `rules` | Evidence extraction tier |
| `--rank-cohort` | flag | off | Apply cohort percentile ranking |
| `--local-retrieval-threshold` | 0.0–1.0 | `0.72` | Minimum cosine similarity for local match |
| `--local-ambiguity-margin` | 0.0–1.0 | `0.04` | Required gap between top and runner-up match |
| `--api-model` | string | `openai/gpt-oss-20b:free` | OpenRouter model for API fallback |
| `--disable-api-cache` | flag | off | Skip cache reads (forces fresh API calls) |
| `--no-narratives` | flag | off | Omit reasoning strings from output |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | API key for tier-3 semantic classification |
| `PASSPORT_LOCAL_RETRIEVAL_BACKEND` | `tfidf` (default, deterministic) or `transformer` (ChromaDB) |

---

## Output Contract

Each student output JSON follows this schema:

```json
{
  "student_name": "Abigail Rodriguez",
  "email": "abigailrodriguez@crotonaihs.org",
  "scores": {
    "EC": {
      "score": 52.46,
      "sub_scores": { "Verbal": 19.58, "Written": 11.88, "Interpersonal": 10.0, "CrossCultural": 11.0 },
      "source": "survey+docs",
      "reasoning": "Abigail Rodriguez demonstrates competency in EC scoring 52.46."
    }
  },
  "pathcredits_score": 62.5,
  "ranking": { "rank": 7, "percentile": 65.0, "cohort_size": 20 },
  "evidence_facts": [
    {
      "span": "Built a database of EBT stores across NYC metro area",
      "action_type": "build_create",
      "artifact_type": "technical_project",
      "initiative_level": "project_execution",
      "context_type": "general",
      "relevant_pillars": ["CI", "CR", "CT"],
      "confidence": 0.90,
      "method": "rules",
      "classification_source": "rules"
    }
  ],
  "normalized_fields": {
    "Languages": { "value": "English, Spanish", "source": "docs", "confidence": 0.80, "was_missing": true }
  },
  "normalization_summary": {
    "semantic_mode": "rules",
    "model_calls": 0,
    "fields_from_survey": 42,
    "fields_from_documents": 24,
    "conservative_defaults": 8,
    "unresolved_fields": 2
  },
  "scoring_summary": {
    "auto_scored": true,
    "needs_review": false,
    "evidence_fact_count": 38,
    "unresolved_fields": 2
  },
  "opportunities": [
    {
      "company": "Khan Academy",
      "title": "Content Creator Intern — K-12",
      "url": "https://jobs.greenhouse.io/khanacademy/...",
      "match_score": 82,
      "verdict": "solid",
      "strengths": [
        "Your content creation at MetaBronx (poetry, community stories) directly mirrors the educational content this role produces"
      ],
      "gaps": [
        "Role wants K-12 subject expertise — your profile shows general education interest"
      ],
      "next_steps": [
        "Add a subject focus to your objective: 'seeking education internship in literacy and writing instruction'",
        "Create one sample worksheet or lesson outline as a portfolio piece (+4 CR expected)"
      ]
    }
  ]
}
```

### Scoring Summary Flags

| Flag | Meaning |
|------|---------|
| `auto_scored: true` | All scores computed without manual review |
| `needs_review: true` | Sparse evidence — scores are conservative, recommend human check |
| `review_reasons` | Why it needs review: `missing_documents`, `no_structured_evidence`, `sparse_evidence`, `limited_grounded_evidence`, `sparse_documents` |

---

## Design Principles

### Deterministic Scoring
The 6-pillar scoring path contains zero LLM calls. Every score is reproducible from the same input. The LLM is used only at the semantic matching tier (optional, for evidence classification) and in the opportunity explanation stage (for generating human-readable strengths/gaps/next-steps). The scoring itself is pure math.

### Conservative Defaults
When a field is missing and can't be inferred from documents, the system uses a conservative default (typically the 25th–40th percentile of observed values) and marks it `conservative: true, confidence: 0.4–0.6`. This prevents score inflation from optimistic guessing.

### Bounded LLM Use
Every LLM call in the system is:
- **Taxonomy-bounded**: The API classifier can only return concept IDs from the fixed concept library — it cannot invent new categories.
- **Validated**: API responses are checked against the top local retrieval candidate. If they disagree, the API result is discarded.
- **Cache-backed**: Identical spans produce cache hits, not new API calls.

### Resumability
Long-running operations (batch scoring, opportunity matching, pairwise judging) write progress to JSONL sidecars. A killed run resumes where it left off — completed work is never repeated. The open-jobs download is also resumable (picks up at the byte offset where the connection dropped).

### Learning Over Time
The system is designed to improve from every signal — outcomes, clicks, judgments, ratings. A static embedding ranker (~68% accuracy) becomes a distilled Bradley-Terry model (~90% accuracy) after ~200 pairwise comparisons, and continues to improve from click-through feedback. The more students use it, the better it gets.

---

## Implementation Roadmap

### Phase 1 — MVP (Built)
- [x] 6-pillar deterministic scoring pipeline
- [x] 3-tier semantic evidence extraction
- [x] Batch processing with cohort ranking
- [x] Old pipeline comparison and accuracy audit

### Phase 2 — Opportunity Matching (Next)
- [ ] `hull.py` adaptation — structured filtering of open-jobs dataset
- [ ] Semantic ranking with structured bonus signals
- [ ] `match.py` adaptation — per-role LLM explanation with JSON Schema validation
- [ ] Self-contained HTML student dashboard

### Phase 3 — Learn & Distill (Month 2)
- [ ] `langsort.py` pairwise judgment collection (LLM-powered)
- [ ] Transitive closure gating for comparison efficiency
- [ ] `btrank.py` Bradley-Terry distillation → `taste.npz` model
- [ ] `download.py` daily dataset refresh cron job
- [ ] Apply distilled model to daily snapshots ($0/student ongoing)

### Phase 4 — Feedback Loops (Month 3)
- [ ] Outcome-based pillar weight tuning (logistic regression on internship/college outcomes)
- [ ] Click-through relevance feedback (online learning on embedding weights)
- [ ] Gap resolution impact tracking (prioritize highest-impact advice)
- [ ] Explanation quality rating → prompt template A/B testing

### Phase 5 — Monitoring (Month 3+)
- [ ] Pillar calibration drift detection (weekly KS test)
- [ ] Staff judgment pipeline (replace LLM pairwise with human judges for higher quality)
- [ ] A/B test `gold` vs `fuse` vs `bt` ranking methods on real student outcomes

---

## FAQ

### Why not use a single LLM call to score everything?

LLMs produce bunched, uncalibrated scores on absolute scales — everything lands near 85. They're also non-deterministic: the same resume scored twice can vary by 10+ points. Deterministic formulas on structured evidence facts give reproducible, inspectable scores. The LLM is reserved for what it does well: generating human-readable explanations from structured data, and making pairwise comparisons ("which of these two fits better?") where binary choices produce sharper discrimination than absolute scoring.

### Why three tiers of semantic matching instead of just using an LLM?

Speed, cost, and determinism. Tier 1 (rules) classifies ~80% of evidence spans in microseconds. Tier 2 (local retrieval) catches niche projects that rules miss, on CPU. Tier 3 (API) is rarely reached and cache-backed. A single-tier LLM approach would cost ~$0.10/student and produce non-deterministic classifications.

### How does this handle students with sparse profiles?

The normalization layer detects sparse inputs and flags `needs_review: true`. Conservative defaults prevent inflated scores. The output still produces pillar scores and opportunity matches, but at lower confidence. The next-steps output becomes more useful here — it tells the student exactly what evidence to add and estimates the expected score improvement for each action.

### What makes a "good" PathCredits score?

The scoring is calibrated so that a score of 50 represents baseline competency across all pillars. 70+ is strong (top quartile). 85+ is exceptional. The formulas are designed so that no pillar can be gamed with a single strong signal — the dimensional weighting requires breadth. Once outcome-based tuning is active (Phase 4), the weights will reflect what actually predicts real-world success, not just what the rubric says.

### Can I use this without the open-jobs dataset?

Yes. The scoring pipeline is fully self-contained. The opportunity matching stage is dataset-agnostic — it needs any structured job feed with `title`, `description`, `function`, `level`, and `skills` fields. The architecture works with employer partner listings, another public dataset, or SpeakHire's own opportunity database.

### How much does it cost to run?

| Scenario | Per Student |
|----------|-------------|
| Scoring only, rules mode | $0 (CPU only) |
| Scoring + local retrieval | $0 (CPU only) |
| Scoring + API fallback | ~$0.01–0.05 |
| Full pipeline + opportunity matching (top-5 LLM explain) | ~$0.05–0.10 |
| After distillation (taste.npz model) | $0 (model inference is CPU-only) |

### Why 6 pillars instead of 5 or 7?

The 6 pillars map directly to SpeakHire's program evaluation framework (see [`Final_10.pdf`](Final_10.pdf) — "SpeakHire Student Evaluation Metrics"). Each pillar corresponds to a dimension the program already measures through surveys and session data. CT and CI are scored separately because the program distinguishes analytical thinking from creative innovation in its rubric.

### Why pairwise comparisons instead of absolute scoring?

LLMs bunch absolute scores near the middle (everything is an 85). Binary choice ("which fits better, A or B?") produces much sharper discrimination. The Bradley-Terry model then aggregates hundreds of binary choices into a smooth, well-separated ranking. And the distilled model (trained on those pairwise decisions) generalizes to new students and new jobs at zero additional LLM cost — a one-time training investment pays off forever.

### How many pairwise comparisons are needed?

~200 comparisons saturate model accuracy (held-out accuracy plateaus within 1% of its ceiling). That's ~15 minutes of staff time, or ~$2 in LLM costs. The gating pattern ensures every comparison adds genuinely new information — no budget is wasted on transitively implied pairs.

---

## External Datasets & Tools

| Resource | What It Provides | Components Borrowed | License |
|----------|-----------------|-------------------|---------|
| [`elliottdehn/open-jobs`](https://github.com/elliottdehn/open-jobs) | 967K live jobs, 16 ATS, daily refresh, precomputed embeddings, 34 LLM-extracted fields | `hull.py` (filtering), `rank.py` (semantic scoring), `match.py` (LLM explanation), `langsort.py` (pairwise judgments), `btrank.py` (Bradley-Terry distillation), `stream.py` (memory-bounded reads), `download.py` (resumable daily refresh) | CC0 |
| [`YangXu624/employer_matching`](https://github.com/YangXu624/employer_matching) | JD-to-competency embedding pipeline, rubric vector caching strategy | Vector caching pattern, rubric-aligned scoring structure | — |

---

## Related Documentation

- [`Final_10.pdf`](Final_10.pdf) — SpeakHire Student Evaluation Metrics framework (the requirements doc this pipeline implements)
- [`passport_agent_v2/docs/pipeline_comparison.md`](passport_agent_v2/docs/pipeline_comparison.md) — Old vs new pipeline rationale
- [`passport_agent_v2/docs/pipeline_implementation.md`](passport_agent_v2/docs/pipeline_implementation.md) — Implementation details per stage

---

Built for [SpeakHire](https://speakhire.org). Scoring logic is deterministic and inspectable. The LLM explains — it doesn't decide. The system learns from every outcome, every click, and every judgment.
