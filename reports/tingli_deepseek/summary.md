# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 5.6m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 23 | 56 | 66s |
| Alan Li | finance |  | rescue | 82% | OK | 52 | 26 | 20s |
| Alex Aquino | technology | engineering | rescue | 69% | YES | 79 | 0 | 6s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 36 | 51 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 63 | 5 | 31s |
| Bianka Pena | healthcare | clinical-support | rescue | 67% | YES | 80 | 0 | 15s |
| Cristal Davidson | arts-media | content-creation | rescue | 63% | YES | 14 | 52 | 7s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 83 | 0 | 23s |
| Emiliano Hernandez Cordero | social-service | youth-support | rescue | 55% | YES | 36 | 40 | 16s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 11 | 11 | 17s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 51 | 82 | 17s |
| Iyana Rankin | arts-media | content-creation | rescue | 61% | YES | 27 | 53 | 9s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 39 | 45 | 16s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 24 | 56 | 14s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | OK | 44 | 39 | 21s |
| Leyli Hernandez | social-service | youth-support | rescue | 65% | YES | 16 | 38 | 14s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 12 | 10 | 13s |
| Naim Bakere | education | community-education | rescue | 62% | YES | 29 | 33 | 6s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 78 | 4 | 15s |

## Benchmarks

### Abigail Rodriguez — FAIL
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Adjunct Faculty: Entrepreneurship 2026', 'Camp Instructor -Chicago', 'Instructor - Los Angeles']
- Core gaps: ['classroom-management', 'javascript', 'kahoot', 'python', 'quizizz']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Refugee Health & Social Integration Intern (Summer 2026)', '2nd Grade Teacher – Small Charter School', 'Certified Nursing Assistant (CNA)']
- Core gaps: ['communication', 'reading-comprehension']
- Forbidden hits: None
