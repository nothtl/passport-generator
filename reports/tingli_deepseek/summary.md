# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 7.0m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 15/19
- rescue lane: 19/19
- avg confidence: 68%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 56 | 28 | 92s |
| Alan Li | finance |  | rescue | 82% | OK | 37 | 33 | 18s |
| Alex Aquino | technology | engineering | rescue | 72% | YES | 79 | 0 | 7s |
| Ayele Dounou | education | youth-programs | rescue | 58% | YES | 15 | 64 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 63 | 5 | 56s |
| Bianka Pena | healthcare | research | rescue | 64% | YES | 32 | 46 | 25s |
| Cristal Davidson | arts-media | content-creation | rescue | 63% | YES | 14 | 52 | 7s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 48 | 29 | 24s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 55% | YES | 15 | 60 | 17s |
| Francis Calderon | arts-media | graphic-design | rescue | 61% | YES | 12 | 11 | 18s |
| Ismatu Barry | education | classroom-support | rescue | 69% | YES | 42 | 80 | 21s |
| Iyana Rankin | arts-media | content-creation | rescue | 62% | YES | 28 | 51 | 9s |
| Jhan Motta | arts-media | photo-video | rescue | 67% | YES | 30 | 58 | 23s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 36 | 42 | 16s |
| Leila Titikpina | healthcare | clinical-support | rescue | 72% | YES | 45 | 39 | 24s |
| Leyli Hernandez | social-service | youth-support | rescue | 65% | YES | 23 | 35 | 16s |
| Mingyu Carl Huo | education | classroom-support | rescue | 82% | OK | 47 | 21 | 16s |
| Naim Bakere | education | community-education | rescue | 61% | YES | 27 | 35 | 8s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 76 | 4 | 16s |

## Benchmarks

### Abigail Rodriguez — PASS
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Instructor - Los Angeles', 'Paraeducator', 'Substitute Teacher']
- Core gaps: ['classroom-management', 'curriculum-delivery', 'teaching']
- Forbidden hits: None

### Leila Titikpina — FAIL
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Certified Nursing Assistant (CNA)', 'Teacher Assistant', 'Pharmacy Technician']
- Core gaps: []
- Forbidden hits: None
