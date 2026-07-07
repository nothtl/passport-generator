# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 6.7m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 45 | 37 | 60s |
| Alan Li | finance |  | rescue | 82% | OK | 69 | 2 | 20s |
| Alex Aquino | technology | engineering | rescue | 69% | YES | 79 | 0 | 7s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 36 | 40 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 87 | 0 | 65s |
| Bianka Pena | healthcare | research | rescue | 67% | YES | 80 | 0 | 21s |
| Cristal Davidson | arts-media | content-creation | rescue | 63% | YES | 14 | 52 | 8s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 35 | 43 | 27s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 55% | YES | 8 | 68 | 16s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 10 | 11 | 26s |
| Ismatu Barry | education | classroom-support | rescue | 70% | YES | 35 | 45 | 18s |
| Iyana Rankin | arts-media | content-creation | rescue | 61% | YES | 28 | 51 | 10s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 33 | 53 | 19s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 30 | 50 | 21s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | OK | 60 | 41 | 23s |
| Leyli Hernandez | social-service | youth-support | rescue | 65% | YES | 18 | 37 | 18s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 28 | 40 | 16s |
| Naim Bakere | education | classroom-support | rescue | 62% | YES | 27 | 35 | 8s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 79 | 3 | 16s |

## Benchmarks

### Abigail Rodriguez — FAIL
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Paraeducator', 'Camp Instructor -Chicago', 'Instructor - Los Angeles']
- Core gaps: ['classroom-management', 'curriculum-delivery', 'javascript', 'kahoot', 'python']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Pharmacy Technician', 'Refugee Health & Social Integration Intern (Summer 2026)', 'Certified Nursing Assistant (CNA)']
- Core gaps: ['communication', 'data-entry']
- Forbidden hits: None
