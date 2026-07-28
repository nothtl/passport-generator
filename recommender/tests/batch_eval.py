import json, os, sys, time, re

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

os.environ["RECOMMENDER_DISABLE_EMBEDDINGS"] = "1"
os.environ["DEEPSEEK_API_KEY"] = ""

from recommender.analyze import analyze

STUDENT_DIR = r"C:\Users\Tingli\AppData\Local\Temp\opencode\students"

def find_file(prefix: str, suffix: str) -> str | None:
    for f in os.listdir(STUDENT_DIR):
        if suffix in f and f.startswith(prefix):
            return os.path.join(STUDENT_DIR, f)
    for f in os.listdir(STUDENT_DIR):
        name_lower = f.lower()
        if suffix.lower() in name_lower and prefix.lower() in name_lower:
            return os.path.join(STUDENT_DIR, f)
    return None

def find_report(name: str) -> dict | None:
    for f in os.listdir(STUDENT_DIR):
        if name in f and f.endswith('.json') and 'reports' in f:
            with open(os.path.join(STUDENT_DIR, f), encoding='utf-8') as fh:
                return json.load(fh)
    return None

students = ["Alan Li", "Benjamin Medrano", "Bianka Pena", "Jhan Motta", "Samuel Tavarez"]

for name in students:
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    resume_path = None
    linkedin_path = None

    # Try to find resume
    for f in os.listdir(STUDENT_DIR):
        if name.lower().replace(' ', '') in f.lower().replace(' ', ''):
            full = os.path.join(STUDENT_DIR, f)
            if 'resume.txt' in f.lower():
                resume_path = full
            elif 'resume' in f.lower() and f.endswith('.md'):
                resume_path = full
            elif 'linkedin' in f.lower() and f.endswith('.md'):
                linkedin_path = full

    if not resume_path:
        # Try alternative filenames
        for f in os.listdir(STUDENT_DIR):
            if name in f and ('resume' in f.lower() or 'linkedin' in f.lower()):
                full = os.path.join(STUDENT_DIR, f)
                if not resume_path:
                    resume_path = full
                elif not linkedin_path:
                    linkedin_path = full

    if not resume_path:
        print(f"  ERROR: No resume found")
        continue

    resume_text = open(resume_path, encoding='utf-8').read()
    linkedin_text = open(linkedin_path, encoding='utf-8').read() if linkedin_path else ""

    t0 = time.time()
    r = analyze(resume_text=resume_text, linkedin_text=linkedin_text, top_k=5)
    elapsed = time.time() - t0

    print(f"  ({elapsed:.1f}s)  Function: {r.get('function')}  Subdomain: {r.get('subdomain')}  Lane: {r.get('lane_used')}  Exp: {r.get('experience_months')}mo/{r.get('experience_level')}")

    # Current jobs
    print(f"\n  CURRENT JOBS (top 5):")
    for i, j in enumerate(r.get("jobs", [])[:5]):
        bd = j.get("job_score_breakdown", {})
        print(f"    {i+1}. fit={j.get('fit',0):>3} [{j.get('job_type','?'):>8}] sk={bd.get('skill_overlap',0):.2f} fm={bd.get('function_match',0):.2f} goal={bd.get('goal_alignment',0):.2f} study={bd.get('study_alignment',0):.2f}")
        print(f"       {j.get('title','')[:55]} @ {j.get('company','')[:25]}")

    # Old report
    old = find_report(name)
    if old:
        print(f"\n  OLD REPORT:")
        print(f"    Zone: {old.get('zone')}  Education: {old.get('education_level')}")
        old_jobs = old.get("top_jobs", [])[:5]
        for i, j in enumerate(old_jobs):
            ms = j.get("blended_score", j.get("match_score", "?"))
            print(f"    {i+1}. score={ms} [{j.get('family','?'):>20}] {j.get('title','')[:55]} @ {j.get('company','')[:25]}")
    else:
        print(f"\n  OLD REPORT: Not found")
