# DeepSeek v4 Flash — Recommender Reports

Generated 19 reports in 5.8m using DeepSeek v4 flash.
LLM config: mode=force, model=deepseek-chat, max_calls=9

## Stats
- needs_review: 15/19
- rescue lane: 19/19
- avg confidence: 68%

## Per-Student Results

| Student | Function | Subdomain | Lane | Conf | Review | Skills | Rejected | Time |
|---------|----------|-----------|------|------|--------|--------|----------|------|
| Abigail Rodriguez | education | youth-programs | rescue | 67% | YES | 49 | 30 | 59s |
| Alan Li | finance |  | rescue | 82% | OK | 15 | 59 | 18s |
| Alex Aquino | technology | engineering | rescue | 72% | YES | 79 | 0 | 6s |
| Ayele Dounou | education | youth-programs | rescue | 58% | YES | 36 | 43 | 8s |
| Benjamin Medrano | technology | it-support | rescue | 61% | YES | 58 | 5 | 48s |
| Bianka Pena | healthcare | research | rescue | 64% | YES | 59 | 21 | 19s |
| Cristal Davidson | arts-media | content-creation | rescue | 63% | YES | 14 | 52 | 6s |
| Devin Rhodie | technology | software | rescue | 95% | OK | 40 | 41 | 23s |
| Emiliano Hernandez Cordero | social-service | community-support | rescue | 55% | YES | 15 | 62 | 15s |
| Francis Calderon | arts-media | graphic-design | rescue | 61% | YES | 10 | 10 | 16s |
| Ismatu Barry | education | classroom-support | rescue | 69% | YES | 34 | 47 | 16s |
| Iyana Rankin | arts-media | content-creation | rescue | 62% | YES | 28 | 51 | 8s |
| Jhan Motta | arts-media | photo-video | rescue | 67% | YES | 57 | 34 | 21s |
| Khadim Ka | protective-service |  | rescue | 54% | YES | 36 | 42 | 17s |
| Leila Titikpina | healthcare | clinical-support | rescue | 72% | YES | 85 | 0 | 18s |
| Leyli Hernandez | social-service | youth-support | rescue | 65% | YES | 21 | 35 | 15s |
| Mingyu Carl Huo | education | classroom-support | rescue | 82% | OK | 12 | 7 | 14s |
| Naim Bakere | education | community-education | rescue | 61% | YES | 27 | 35 | 7s |
| Samuel Tavarez | finance |  | rescue | 85% | OK | 72 | 10 | 15s |

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
- Top jobs: ['Pharmacy Technician', 'Refugee Health & Social Integration Intern (Summer 2026)', 'Certified Nursing Assistant (CNA)']
- Core gaps: ['cpr-for-the-professional-rescuer', 'data-entry', 'lifeguard', 'record-keeping']
- Forbidden hits: None
