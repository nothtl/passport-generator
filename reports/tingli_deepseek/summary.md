# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 6.2m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 48 | 35 | 62s |
| Alan Li | finance |  | rescue | 80% | OK | 70 | 0 | 16s |
| Alex Aquino | technology | engineering | rescue | 68% | YES | 79 | 0 | 8s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 36 | 41 | 9s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 63 | 5 | 62s |
| Bianka Pena | healthcare | clinical-support | rescue | 64% | YES | 67 | 11 | 19s |
| Cristal Davidson | arts-media | content-creation | rescue | 62% | YES | 14 | 52 | 7s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 88 | 80 | 27s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 56% | YES | 15 | 60 | 14s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 12 | 11 | 11s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 50 | 82 | 15s |
| Iyana Rankin | sales |  | rescue | 62% | YES | 28 | 51 | 9s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 43 | 44 | 22s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 27 | 53 | 17s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | OK | 41 | 48 | 21s |
| Leyli Hernandez | social-service | youth-support | rescue | 61% | YES | 18 | 37 | 13s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 40 | 27 | 16s |
| Naim Bakere | education | community-education | rescue | 61% | YES | 27 | 34 | 7s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 73 | 10 | 14s |

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
- Core gaps: ['data-entry']
- Forbidden hits: None
