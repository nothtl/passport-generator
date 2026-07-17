# Location-Based Job & College Recommendations — Implementation Plan

> **Goal:** Fix three gaps (college diversity, internship relevance, goal job count) AND add location-aware recommendations so NY students see NY jobs/schools.

**Architecture:** Extract student location from resume text → filter parquet jobs by `city`/`location` column → filter colleges by `school.state` → pad all tiers to 5.

**Context:**
- Students: all in New York City / Bronx / Brooklyn (extracted from resume text)
- Parquet subset has `location`, `city`, `hq_city`, `hq_country_code` columns
- College Scorecard API supports `school.state=NY` filter
- Analyze.py `_build_candidate_jobs` already handles `level_filter`

---

## Task 1: Add location extraction util

**Files:** Create `recommender/extract/location_extractor.py`

**Step 1:** Write `extract_location(resume_text: str) -> dict`
```python
def extract_location(resume_text: str) -> dict:
    """Extract city and state from resume text. Returns {'city': '', 'state': ''}"""
    import re
    # NYC boroughs + common city patterns
    nyc = ['New York', 'Brooklyn', 'Bronx', 'Queens', 'Manhattan', 'Staten Island']
    text = resume_text[:500].lower()  # Check top of resume first
    
    # State patterns: "NY", "New York", etc
    state_map = {
        'ny': 'NY', 'new york': 'NY', 'nj': 'NJ', 'ct': 'CT', 'pa': 'PA',
        'ca': 'CA', 'tx': 'TX', 'fl': 'FL', 'il': 'IL', 'ma': 'MA',
    }
    
    city = ''
    state = ''
    
    # Check for NYC boroughs
    for borough in nyc:
        if borough.lower() in text:
            city = borough
            state = 'NY'
            break
    
    # Check for explicit state mentions
    if not state:
        for pattern, abbr in state_map.items():
            if re.search(rf'\b{pattern}\b', text):
                state = abbr
                break
    
    return {'city': city, 'state': state}
```

**Step 2:** Verify with `python -c "from recommender.extract.location_extractor import extract_location; print(extract_location('...'))"`

**Step 3:** Commit

---

## Task 2: Wire location into batch script

**Files:** Modify `recommender/scripts/batch_phase123.py`

**Step 1:** Import location extractor
```python
from recommender.extract.location_extractor import extract_location
```

**Step 2:** In `run_student()`, extract location after resume text:
```python
location = extract_location(resume_text)
student_state = location.get('state', '')
```

**Step 3:** Pass state to college matching:
```python
student_profile = {
    ...
    'preferred_state': student_state,  # ADD THIS
}
```

**Step 4:** Verify by running one student and checking `top_colleges` now shows NY schools.

**Step 5:** Commit

---

## Task 3: Fix college diversity — fetch more pages + state filter

**Files:** Modify `recommender/college/college_matcher.py`

**Step 1:** Increase `max_pages` from 3 to 6 to get ~600 schools across all states.

**Step 2:** When `preferred_state` is set, do TWO queries:
- First query WITH state filter (get preferred state schools)
- Second query WITHOUT state filter (get diverse national schools)
- Merge results, deduplicate, rank together

```python
# Query 1: preferred state schools
state_schools, err = client.fetch_schools(state=profile.preferred_state, per_page=100, max_pages=2)
# Query 2: national diversity (no state filter)
diverse_schools, _ = client.fetch_schools(per_page=100, max_pages=4, page=200)  # random offset
all_schools = state_schools + diverse_schools
```

**Step 3:** Verify Abigail now gets NY schools (CUNY, SUNY, NYU, Columbia etc).

**Step 4:** Commit

---

## Task 4: Fix internship relevance — filter to primary function

**Files:** Modify `recommender/analyze.py` (internship block ~line 1304)

**Step 1:** In the internship extraction block, add a function filter:
```python
if _needs_internships:
    chosen_func = chosen_function
    _intern_pool = [
        j for j in ranked_jobs
        if not j.get("_cross_function")
        and str(j.get("level", "")).lower() in ("intern", "internship")
        and str(j.get("function", "")).lower() in (chosen_func, _SUBSET_FUNCTION_MAP.get(chosen_func, chosen_func))
    ]
```

**Step 2:** This ensures a technology student only gets technology internships, not wedding coordinators from cross-function pools.

**Step 3:** Verify with `Alex Aquino` — his internships should now all be tech-related, not wedding coordinator / business dev.

**Step 4:** Commit

---

## Task 5: Pad goal jobs to 5

**Files:** Modify `recommender/scripts/batch_phase123.py`

**Step 1:** In the tier-padding section, add goal job padding:
```python
# Pad aspirational to 5 using ready_now overflow
aspirational = result.get('aspirational', [])
if len(aspirational) < 5:
    ready = result.get('ready_now', [])
    seen_t = {j['title'] for j in aspirational}
    for j in ready:
        if len(aspirational) >= 5:
            break
        if j['title'] not in seen_t:
            aspirational.append(dict(j))
            seen_t.add(j['title'])
    result['aspirational'] = aspirational
```

**Step 2:** Verify all students now show 5 goal jobs.

**Step 3:** Commit

---

## Task 6: Add location-based job filtering

**Files:** Modify `recommender/retrieve/retriever.py`

**Step 1:** Add `prefer_state: str = ""` parameter to `filter_job_records()`

**Step 2:** Add location scoring inside `filter_job_records`:
```python
# Location match bonus
location = str(job.get('location', job.get('city', ''))).lower()
if prefer_state and prefer_state.lower() in location:
    job['_quality_penalty'] = max(job.get('_quality_penalty', 0) - 0.15, 0)  # boost
```

**Step 3:** Wire `prefer_state` through from `analyze()` → `_build_candidate_jobs()` → `filter_job_records()`.

**Step 4:** Verify NY students get more NY-based job recommendations.

**Step 5:** Commit

---

## Task 7: Full rerun and verification

**Files:** Run `batch_phase123.py` with college cache cleared.

**Verification:** For each student check:
- 5 internships, all in student's function
- 5 goal jobs
- 5 colleges, with in-state schools appearing
- Jobs show location relevance

```python
# Verification assertions
for student in reports:
    assert len(student.internships) == 5
    assert len(student.aspirational) == 5
    assert len(student.top_colleges) == 5
    assert any(c['state'] == student.state for c in student.colleges)  # at least 1 in-state
```

---

## Risk / Tradeoffs

- **College API pagination** — fetching 600 schools may hit rate limits; use 1s delay between pages
- **Parquet location data** — `location`/`city` fields may be sparse for some functions
- **Student location detection** — regex-based; may miss locations in non-standard formats
- **Internship pool size** — filtering to function+intern level may leave <5 results; pad from attainable jobs as fallback