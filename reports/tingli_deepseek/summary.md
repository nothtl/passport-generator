# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 5.7m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 41 | 38 | 73s |
| Alan Li | finance |  | rescue | 80% | OK | 37 | 37 | 13s |
| Alex Aquino | technology | engineering | rescue | 68% | YES | 79 | 0 | 12s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 36 | 43 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 64 | 0 | 31s |
| Bianka Pena | healthcare | clinical-support | rescue | 64% | YES | 79 | 0 | 16s |
| Cristal Davidson | arts-media | content-creation | rescue | 62% | YES | 14 | 52 | 7s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 51 | 29 | 22s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 56% | YES | 8 | 69 | 17s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 11 | 10 | 15s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 42 | 80 | 15s |
| Iyana Rankin | sales |  | rescue | 62% | YES | 28 | 51 | 8s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 59 | 29 | 22s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 33 | 56 | 13s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | OK | 82 | 0 | 25s |
| Leyli Hernandez | social-service | youth-support | rescue | 61% | YES | 14 | 40 | 17s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 15 | 7 | 11s |
| Naim Bakere | education | community-education | rescue | 61% | YES | 27 | 34 | 7s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 72 | 10 | 12s |

## Benchmarks

### Abigail Rodriguez — PASS
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Part-time Career Coach, NeOn Works', 'University Supervisor', 'Bilingual Special Education Teacher (English/Spanish)']
- Core gaps: ['classroom-management', 'communication', 'english', 'lbs1', 'pel']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Home Care Aide/Caregiver - Everett', 'Peer Specialist - 170-01 Douglas Avenue', 'Direct Support Professional (DSP) - All Shifts Available']
- Core gaps: ['caregiving', 'meal-preparation', 'transportation']
- Forbidden hits: None
