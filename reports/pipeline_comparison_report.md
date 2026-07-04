# SpeakHire Recommender — Pipeline Comparison Report

**Generated:** 2026-07-04 13:57

---

## Summary

| | Zip Pipeline | Our Pipeline |
|---|---|---|
| **Database** | 22 GB (full) | 307 MB subset + full streaming for thin functions |
| **Skill extraction** | Gemini-2.5-flash (67 O*NET skills) | JD vocabulary + DeepSeek evidence + O*NET importance |
| **Classification** | O*NET importance-weighted (923 occupations) | 3-signal ensemble + DeepSeek route judge + subdomain routing |
| **Job ranking** | GPT-4o-mini pairwise (langsort) | IDF scoring + keyword routing |
| **Youth eligibility** | ✅ LLM screening | ✅ LLM screening (DeepSeek) |
| **Tiered output** | ❌ Flat list | ✅ ready_now (2) + aspirational (3) |
| **Bridge summaries** | ❌ | ✅ Transferable skill paths |
| **Feedback loop** | ❌ | ✅ Like/dislike → preferences |
| **Offline capable** | ❌ Requires Gemini + OpenAI | ✅ Fast lane (deterministic) |
| **Cost/student** | ~$0.25 | ~$0.003 (**83x cheaper**) |
| **Occupation labels** | Misleading (Maids, Dishwashers) | Clean (education, healthcare) |

**Head-to-head: US 4 — ZIP 12 — TIE 3** (13/19 students getting good jobs)

---

## Per-Student Comparison

| Student | Our Function | Zip Top O*NET | Zip Top Job | Our Top Job | Winner |
|---------|-------------|---------------|-------------|-------------|--------|
| Abigail Rodriguez | education/youth-programs | Training and Development Manag | After School Instructor | Part-time Career Coach, NeOn Works | ZIP |
| Alan Li | finance/ | Office Clerks, General | Office Administrator | Payroll Clerk | ZIP |
| Alex Aquino | technology/engineering | ? | ? | GENERAL LABORER~S03015~5204 | TIE |
| Ayele Dounou | social-service/youth-support | ? | ? |  LA Galaxy, Youth Programs Coach -  | TIE |
| Benjamin Medrano | technology/it-support | Hotel, Motel, and Resort Desk  | Customer Service Representative Cal | Physical Therapy Technician | ZIP |
| Bianka Pena | healthcare/clinical-support | Training and Development Speci | Attendant Care Worker/Peer Support  | Lab - Medical Lab Technician/Medica | ZIP |
| Cristal Davidson | arts-media/content-creation | Media Programming Directors | Faculty Assistant and Program Coord | Junior Level Designer | ZIP |
| Devin Rhodie | technology/software | Electrical Engineers | Software Engineer (HPC) Linux & Scr | Graphic Design / Marketing  Intern  | ZIP |
| Emiliano Hernandez Cordero | education/youth-programs | Receptionists and Information  | Client Care Worker/Customer Care Wo | Sports Based Youth Development Spec | ZIP |
| Francis Calderon | arts-media/content-creation | Dishwashers | Teachers Aide Early Head Start - Fu | Junior Communication Specialist and | US |
| Ismatu Barry | education/classroom-support | Computer and Information Syste | Residence Program Specialist | Instructional Aide  | ZIP |
| Iyana Rankin | sales/ | First-Line Supervisors of Prod | Projects Coordinator | Leasing Consultant | ZIP |
| Jhan Motta | arts-media/photo-video | Graphic Designers | Talent Branding Specialist (Creativ | Graphic Designer KZ | ZIP |
| Khadim Ka | protective-service/ | Maids and Housekeeping Cleaner | Front Desk Monitor  | Loss Prevention Associate | US |
| Leila Titikpina | healthcare/clinical-support | Maids and Housekeeping Cleaner | Attendant Care Worker/Peer Support  | Peer Specialist - 170-01 Douglas Av | US |
| Leyli Hernandez | social-service/youth-support | First-Line Supervisors of Reta | Youth Advocate | Amazing Athletes Youth Sports Coach | ZIP |
| Mingyu Carl Huo | education/classroom-support | ? | ? | Lower School Teacher | TIE |
| Naim Bakere | education/community-education | First-Line Supervisors of Hous | Coordinator, People | Preschool Aide | US |
| Samuel Tavarez | finance/ | Personal Financial Advisors | Student Account Representative | Credit Analyst I | ZIP |

---

## Key Improvements Over Zip

1. **Tiered output**: Students see 2 jobs they can do TODAY and 3 to work TOWARD
2. **Bridge summaries**: When a ready-now job differs from aspirational goals, the pipeline explains how it builds transferable skills
3. **Feedback loop**: Students like/dislike jobs → preferences shape future recommendations
4. **83x cheaper**: $0.003 vs $0.25 per student
5. **Clean labels**: education/classroom-support vs Maids and Housekeeping Cleaners
6. **Offline fallback**: Fast lane works without any API calls
7. **Aspiration routing**: ideal_careers input weighted above past experience for students

## Remaining Gaps

1. **Job explanations**: Auto-generated vs GPT-4o-mini specific summaries
2. **Very sparse resumes** (Francis: 3 skills, Alex: messy engineering): classification needs more signal
3. **Survey data not connected**: 73 agent1 fields (SMART goals, aspirations) exist but never reach recommender
4. **Only 2 benchmarks**: Need 19 to track quality over time

## Files

- Reports:  (19 JSON + summary.md)
- Pipeline: 
- Corpus:  (307 MB)
- Full parquet:  (21 GB)