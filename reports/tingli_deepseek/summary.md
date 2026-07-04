# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 7.6m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 15/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 86 | 74 | 143s |
| Alan Li | finance |  | rescue | 80% | OK | 34 | 39 | 16s |
| Alex Aquino | technology | engineering | rescue | 68% | YES | 79 | 0 | 9s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 15 | 65 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 63 | 5 | 31s |
| Bianka Pena | healthcare | clinical-support | rescue | 64% | YES | 67 | 11 | 19s |
| Cristal Davidson | arts-media | content-creation | rescue | 62% | YES | 14 | 52 | 9s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 60 | 29 | 21s |
| Emiliano Hernandez Cordero | education | youth-programs | rescue | 56% | YES | 15 | 61 | 15s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 12 | 11 | 17s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 43 | 79 | 33s |
| Iyana Rankin | sales |  | rescue | 62% | YES | 27 | 52 | 13s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 57 | 27 | 21s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 29 | 50 | 13s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | YES | 121 | 0 | 36s |
| Leyli Hernandez | social-service | youth-support | rescue | 61% | YES | 20 | 38 | 18s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 13 | 9 | 13s |
| Naim Bakere | education | classroom-support | rescue | 61% | YES | 29 | 33 | 8s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 76 | 4 | 13s |

## Benchmarks

### Abigail Rodriguez — PASS
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Adjunct Faculty: Entrepreneurship 2026', 'Supply Chain Coordinator Coach', 'Bilingual Special Education Teacher (English/Spanish)']
- Core gaps: ['teaching', 'english', 'lbs1', 'pel', 'speaking']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Home Care Aide/Caregiver - Everett', 'Peer Specialist - 170-01 Douglas Avenue', 'Direct Support Professional (DSP) - All Shifts Available']
- Core gaps: ['transportation', 'meal-preparation']
- Forbidden hits: None
