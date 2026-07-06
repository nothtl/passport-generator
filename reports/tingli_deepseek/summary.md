# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 5.8m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 78 | 38 | 73s |
| Alan Li | finance |  | rescue | 80% | OK | 34 | 39 | 13s |
| Alex Aquino | technology | engineering | rescue | 68% | YES | 79 | 0 | 14s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 15 | 64 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 63 | 5 | 30s |
| Bianka Pena | healthcare | clinical-support | rescue | 64% | YES | 76 | 80 | 18s |
| Cristal Davidson | arts-media | content-creation | rescue | 62% | YES | 14 | 52 | 7s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 39 | 37 | 24s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 56% | YES | 15 | 60 | 17s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 10 | 11 | 15s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 45 | 79 | 21s |
| Iyana Rankin | sales |  | rescue | 62% | YES | 28 | 51 | 9s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 40 | 44 | 19s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 27 | 62 | 12s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | OK | 44 | 41 | 19s |
| Leyli Hernandez | social-service | youth-support | rescue | 61% | YES | 24 | 32 | 17s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 52 | 58 | 14s |
| Naim Bakere | education | community-education | rescue | 61% | YES | 26 | 37 | 7s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 76 | 4 | 13s |

## Benchmarks

### Abigail Rodriguez — PASS
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Adjunct Faculty: Entrepreneurship 2026', 'Supply Chain Coordinator Coach', 'Bilingual Special Education Teacher (English/Spanish)']
- Core gaps: ['teaching', 'english', 'lbs1', 'pel', 'special-education']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Peer Specialist - 170-01 Douglas Avenue', 'Respite Care Provider', 'Direct Support Professional (DSP) - All Shifts Available']
- Core gaps: ['caregiving', 'light-housekeeping', 'meal-preparation', 'transportation']
- Forbidden hits: None
