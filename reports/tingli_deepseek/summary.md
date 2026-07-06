# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 6.6m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 14/19
- rescue lane: 19/19
- avg confidence: 69%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 53 | 29 | 61s |
| Alan Li | finance |  | rescue | 80% | OK | 59 | 18 | 17s |
| Alex Aquino | technology | engineering | rescue | 68% | YES | 79 | 0 | 10s |
| Ayele Dounou | social-service | youth-support | rescue | 58% | YES | 15 | 65 | 10s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 40 | 25 | 46s |
| Bianka Pena | healthcare | clinical-support | rescue | 64% | YES | 80 | 0 | 21s |
| Cristal Davidson | arts-media | content-creation | rescue | 62% | YES | 14 | 52 | 11s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 35 | 44 | 33s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 56% | YES | 33 | 41 | 17s |
| Francis Calderon | arts-media | content-creation | rescue | 63% | YES | 14 | 11 | 14s |
| Ismatu Barry | education | classroom-support | rescue | 72% | YES | 45 | 36 | 17s |
| Iyana Rankin | sales |  | rescue | 62% | YES | 28 | 51 | 13s |
| Jhan Motta | arts-media | photo-video | rescue | 68% | YES | 62 | 29 | 20s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 36 | 41 | 19s |
| Leila Titikpina | healthcare | clinical-support | rescue | 77% | OK | 84 | 51 | 26s |
| Leyli Hernandez | social-service | youth-support | rescue | 61% | YES | 24 | 33 | 16s |
| Mingyu Carl Huo | education | classroom-support | rescue | 90% | OK | 57 | 12 | 16s |
| Naim Bakere | education | community-education | rescue | 61% | YES | 27 | 34 | 11s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 76 | 4 | 16s |

## Benchmarks

### Abigail Rodriguez — PASS
- Function: `education` (subdomain: `youth-programs`)
- Lane: `rescue`
- Top jobs: ['Paraeducator', 'education (Entry)', 'Camp Instructor -Chicago']
- Core gaps: ['classroom-management', 'javascript', 'kahoot', 'python', 'quizizz']
- Forbidden hits: None

### Leila Titikpina — PASS
- Function: `healthcare` (subdomain: `clinical-support`)
- Lane: `rescue`
- Top jobs: ['Refugee Health & Social Integration Intern (Summer 2026)', '2nd Grade Teacher – Small Charter School', 'Certified Nursing Assistant (CNA)']
- Core gaps: ['reading-comprehension']
- Forbidden hits: None
