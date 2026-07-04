# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 5.0m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 24 | 55 | 62s |
| Alan Li | finance |  | rescue | 80% | OK | 40 | 34 | 13s |
| Alex Aquino | technology | engineering | rescue | 68% | YES | 79 | 0 | 7s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 36 | 41 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 63 | 5 | 23s |
| Bianka Pena | healthcare | clinical-support | rescue | 64% | YES | 53 | 24 | 16s |
| Cristal Davidson | arts-media | content-creation | rescue | 62% | YES | 14 | 52 | 6s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 90 | 52 | 17s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 56% | YES | 41 | 36 | 17s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 11 | 11 | 14s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 63 | 82 | 19s |
| Iyana Rankin | sales |  | rescue | 62% | YES | 28 | 51 | 8s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 64 | 24 | 17s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 35 | 45 | 11s |
| Leila Titikpina | healthcare | pre-med-shadowing | rescue | 77% | OK | 53 | 41 | 16s |
| Leyli Hernandez | social-service | youth-support | rescue | 61% | YES | 20 | 36 | 14s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 49 | 69 | 12s |
| Naim Bakere | education | classroom-support | rescue | 61% | YES | 26 | 37 | 7s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 35 | 46 | 12s |

## Benchmarks

### Abigail Rodriguez — PASS
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Adjunct Faculty: Entrepreneurship 2026', 'University Supervisor', 'Bilingual Special Education Teacher (English/Spanish)']
- Core gaps: ['teaching', 'communication', 'english', 'lbs1', 'pel']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `pre-med-shadowing`)
- Lane: `rescue`
- Top jobs: ['Peer Specialist - 170-01 Douglas Avenue', 'Home Care Aide/Caregiver - Everett', 'Home Care Aide AM 8 to 12 hour Shifts']
- Core gaps: ['transportation']
- Forbidden hits: None
