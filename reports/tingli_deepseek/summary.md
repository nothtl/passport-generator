# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 6.0m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 24 | 41 | 61s |
| Alan Li | finance |  | rescue | 80% | OK | 70 | 0 | 14s |
| Alex Aquino | technology | engineering | rescue | 68% | YES | 79 | 0 | 6s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 36 | 41 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 63 | 5 | 62s |
| Bianka Pena | healthcare | clinical-support | rescue | 64% | YES | 80 | 0 | 20s |
| Cristal Davidson | arts-media | content-creation | rescue | 62% | YES | 14 | 52 | 7s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 35 | 42 | 23s |
| Emiliano Hernandez Cordero | education | youth-programs | rescue | 56% | YES | 15 | 60 | 13s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 10 | 10 | 12s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 36 | 44 | 17s |
| Iyana Rankin | sales |  | rescue | 62% | YES | 25 | 54 | 9s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 51 | 35 | 19s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 19 | 60 | 14s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | OK | 83 | 0 | 25s |
| Leyli Hernandez | social-service | youth-support | rescue | 61% | YES | 18 | 37 | 15s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 35 | 38 | 14s |
| Naim Bakere | education | classroom-support | rescue | 61% | YES | 27 | 37 | 8s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 51 | 29 | 15s |

## Benchmarks

### Abigail Rodriguez — FAIL
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Camp Instructor -Chicago', 'Instructor - Los Angeles', 'Paraeducator']
- Core gaps: ['classroom-management', 'curriculum-delivery', 'javascript', 'kahoot', 'python']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Refugee Health & Social Integration Intern (Summer 2026)', 'Certified Nursing Assistant (CNA)', 'Teacher Assistant']
- Core gaps: []
- Forbidden hits: None
