import json, os, sys, time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

os.environ["RECOMMENDER_DISABLE_EMBEDDINGS"] = "1"
os.environ["DEEPSEEK_API_KEY"] = ""

from recommender.analyze import analyze

BASE = r"C:\Users\Tingli\AppData\Local\Temp\opencode\students"
OUT = r"C:\Users\Tingli\AppData\Local\Temp\opencode\students"

students = [
    ("Alan Li",
     os.path.join(BASE, "Speakhire_Recommender_speakhire-recommender_output_work_Alan Li_resume.txt"),
     os.path.join(BASE, "Alan Li_Alan Li - LinkedIn.md"),
     "Computer Science student seeking tech internship"),
    ("Benjamin Medrano",
     os.path.join(BASE, "Speakhire_Recommender_speakhire-recommender_output_work_Benjamin Medrano_resume.txt"),
     "",
     "Bilingual customer service & IT support"),
    ("Bianka Pena",
     os.path.join(BASE, "Speakhire_Recommender_speakhire-recommender_output_work_Bianka Pena_resume.txt"),
     os.path.join(BASE, "Bianka Pena_Bianka Pena - LinkedIn.md"),
     ""),
    ("Jhan Motta",
     os.path.join(BASE, "Speakhire_Recommender_speakhire-recommender_output_work_Jhan Motta_resume.txt"),
     os.path.join(BASE, "Jhan Motta_Jhan Motta - LinkedIn.md"),
     ""),
    ("Samuel Tavarez",
     os.path.join(BASE, "Speakhire_Recommender_speakhire-recommender_output_work_Samuel Tavarez_resume.txt"),
     os.path.join(BASE, "Samuel Tavarez_Samuel Tavarez - LinkedIn.md"),
     ""),
]

results = []

for name, resume_path, linkedin_path, headline in students:
    resume_text = open(resume_path, encoding='utf-8').read()
    linkedin_text = open(linkedin_path, encoding='utf-8').read() if linkedin_path and os.path.exists(linkedin_path) else ""

    t0 = time.time()
    r = analyze(resume_text=resume_text, linkedin_text=linkedin_text, headline_text=headline, top_k=5)
    elapsed = time.time() - t0

    fn = r.get('function', '?')
    sd = r.get('subdomain', '?')
    lane = r.get('lane_used', '?')
    exp_lvl = r.get('experience_level', '?')
    exp_mo = r.get('experience_months', 0)

    jobs_info = []
    for j in r.get("jobs", [])[:5]:
        bd = j.get("job_score_breakdown", {})
        jobs_info.append({
            "title": j.get("title", ""),
            "company": j.get("company", ""),
            "fit": j.get("fit", 0),
            "job_type": j.get("job_type", ""),
            "skill_overlap": bd.get("skill_overlap", 0),
            "function_match": bd.get("function_match", 0),
            "goal_alignment": bd.get("goal_alignment", 0),
            "why": j.get("why", ""),
        })

    results.append({"name": name, "function": fn, "subdomain": sd, "lane": lane,
                    "exp_mo": exp_mo, "exp_level": exp_lvl,
                    "jobs": jobs_info, "elapsed": round(elapsed, 1)})

    # Print progress
    print(f"[{elapsed:.1f}s] {name}: {fn}/{sd} | exp={exp_mo}mo/{exp_lvl} | lane={lane}")
    for i, j in enumerate(jobs_info):
        print(f"  {i+1}. fit={j['fit']:>3} [{j['job_type']:>8}] sk={j['skill_overlap']:.2f} | {j['title'][:50]} @ {j['company'][:20]}")

with open(os.path.join(OUT, "batch_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nDone. Results saved to {OUT}/batch_results.json")
