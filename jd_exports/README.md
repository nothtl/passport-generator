# JD Exports

27 CSV files with 169,757 entry-level US job postings.

## Format

| Column | Description |
|---|---|
| `title` | Job title |
| `company` | Company name |
| `level` | Intern, Junior, or Entry |
| `function` | Job function (technology, healthcare, etc.) |
| `skills` | Extracted skills (comma-separated) |
| `url` | Link to original posting |
| `jd_markdown` | Job description text (truncated to 1000 chars) |

## Files

One CSV per function. Use the function name from the pipeline output:

```
jd_exports/
  technology.csv      ← 1,915 tech JDs
  healthcare.csv      ← 11,439 healthcare JDs
  finance.csv         ← 2,145 finance JDs
  ...
```

## Usage

```python
import pandas as pd
df = pd.read_csv('jd_exports/technology.csv')
# Filter for specific skills
ai_jobs = df[df['skills'].str.contains('machine.learning|computer.vision', case=False)]
# Find companies hiring
print(ai_jobs[['title', 'company']].head(10))
```

## LinkedIn People Finder

For each JD, search LinkedIn for professionals with matching title + company:

```
1. Read title + company from CSV
2. Search: site:linkedin.com/in "{title}" "{company}"
3. Or use LinkedIn API / Recruiter search
4. Filter by skills match
```
