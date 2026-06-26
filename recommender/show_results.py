from recommender.extract.skill_extractor import extract_skills_from_text
from recommender.match.role_matcher import match_role
from recommender.retrieve.retriever import retrieve_jds
from recommender.profile.aggregator import aggregate_skills, aggregate_passport
from recommender.profile.gap_analyzer import analyze_gaps
import json

# ========== Abigail ==========
abigail_resume = """Abigail Rodriguez
Graphic Design & Content Creator for MetaBronx 2021 - Present
I write and create content on issues that affect the community.
I train younger students on how to navigate the internship application process
and I'm part of the program manager.

Volunteer for SPEAKHIRE 2021-2022
I help students with their remote learning and mentor them to be successful.

Volunteer for Teen Outreach Program (TOP) at Crotona High School
I write and record original stories for children. I help with outreach activities
in the community like cleaning up the local parks and creating and posting flyers.

Intern at My Bodega Online
Built a database of EBT stores/warehouses across NYC metro area using Google Maps.

Program Manager at MetaBronx
Organization of information, analysis of data for decision making. Manages new
intern and staff intake and actively mentors and socializes youth participants.

Peer Group Connection Volunteer at Crotona International High School
Teacher-run volunteer program matching upper grades with younger students for
networking and mentoring. Created interactive programs using Kahoot, Padlet."""

skills_a = extract_skills_from_text(abigail_resume)
best_a = match_role(skills_a)
jds_a = retrieve_jds(best_a["function"], best_a["level"], skills_a, top_k=5)
all_skills_a = aggregate_skills(jds_a, best_a["matched_skills"], best_a["missing_skills"])
passport_a = aggregate_passport(jds_a)

print("=" * 60)
print("ABIGAIL RODRIGUEZ")
print("=" * 60)
print()
print("Stage 1 -- Extracted skills ({}):".format(len(skills_a)))
for s in skills_a:
    print("  {}".format(s))
print()

print("Stage 2 -- Role ranking:")

with open("recommender/match/lookup_table.json") as f:
    roles = json.load(f)
scored = []
for k, role in roles.items():
    reqs = {r["name"].lower(): r.get("importance", 1) for r in role["required_skills"]}
    match_w = sum(reqs[s] for s in skills_a if s in reqs)
    total_w = sum(reqs.values())
    scored.append((role["function"], round(match_w / total_w * 100, 1)))
scored.sort(key=lambda x: x[1], reverse=True)
for func, pct in scored[:3]:
    print("  {}: {}%".format(func, pct))
print("  -> Winner: {} at {}%".format(best_a["function"], best_a["match_pct"]))
print()

print("Stage 3 -- JDs retrieved: {}".format(len(jds_a)))
for i, jd in enumerate(jds_a[:3]):
    title = str(jd.get("title", "?"))[:60]
    comp = jd.get("company", "") or "?"
    skills = jd.get("skills", [])
    if hasattr(skills, "tolist"):
        skills = list(skills)
    print("  {}. {} @ {}".format(i + 1, title, comp))
    if skills:
        print("     skills: {}".format(skills[:5]))
print()

print("Stage 4 -- Role profile:")
print("  Role: {} ({})".format(best_a["function"], best_a["level"]))
print("  Match: {}%".format(best_a["match_pct"]))
print()
print("  Full skill breakdown ({} skills from market):".format(len(all_skills_a)))
for s in sorted(all_skills_a, key=lambda x: (-x.frequency, x.skill)):
    mark = "YES" if s.student_has else " NO"
    print("    [{:>3}] {:30s} (found in {} JDs)".format(mark, s.skill, s.frequency))
print()
print("  Matched skills: {}".format(best_a["matched_skills"]))
print("  Missing skills: {}".format(best_a["missing_skills"]))
print()
print("  Ideal passport: {}".format(passport_a))

# ========== Leila ==========
leila_resume = """Leila Titikpina
Ithaca College, BS in Health Science, Graduated May 2026

Level 4 CNYMRC -- Volunteer
Managed the Flu POD on campus with 520 students and staff vaccinated.

Platinum Home Care -- Home Health Aide
Completed 75 hours of training focused on assisting patients at home.

Woodhull Community Care -- Case Management Intern
Created training for case management including Epic and VMware."""

skills_l = extract_skills_from_text(leila_resume)
best_l = match_role(skills_l)
jds_l = retrieve_jds(best_l["function"], best_l["level"], skills_l, top_k=5)
all_skills_l = aggregate_skills(jds_l, best_l["matched_skills"], best_l["missing_skills"])
passport_l = aggregate_passport(jds_l)

print()
print("=" * 60)
print("LEILA TITIKPINA")
print("=" * 60)
print()
print("Stage 1 -- Extracted skills ({}):".format(len(skills_l)))
for s in skills_l:
    print("  {}".format(s))
print()

print("Role ranking:")
scored_l = []
for k, role in roles.items():
    reqs = {r["name"].lower(): r.get("importance", 1) for r in role["required_skills"]}
    match_w = sum(reqs[s] for s in skills_l if s in reqs)
    total_w = sum(reqs.values())
    scored_l.append((role["function"], round(match_w / total_w * 100, 1)))
scored_l.sort(key=lambda x: x[1], reverse=True)
for func, pct in scored_l[:3]:
    print("  {}: {}%".format(func, pct))
print("  -> Winner: {} at {}%".format(best_l["function"], best_l["match_pct"]))
print()

print("JDs retrieved: {}".format(len(jds_l)))
print()
print("Full skill breakdown ({} skills from market):".format(len(all_skills_l)))
for s in sorted(all_skills_l, key=lambda x: (-x.frequency, x.skill)):
    mark = "YES" if s.student_has else " NO"
    print("    [{:>3}] {:30s} (found in {} JDs)".format(mark, s.skill, s.frequency))
print()
print("Matched skills: {}".format(best_l["matched_skills"]))
print("Missing skills: {}".format(best_l["missing_skills"]))
print()
print("Ideal passport: {}".format(passport_l))
