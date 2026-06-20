"""
PathCredits — All 6 Competency Scoring Pipeline
================================================
Scores all 6 competencies for either the 14-student pilot or 199-student master sheet.

Usage:
    python run_all_competencies.py             # 14-student pilot
    python run_all_competencies.py --master    # 199-student master

LLM scoring (Claude Haiku — cheapest):
    Set ANTHROPIC_API_KEY in environment to enable.
    Falls back to algorithmic/keyword scoring if key is absent.

Text fields scored by LLM (per competency):
    EC  — 5 fields: written quality, stay-in-touch, discussion depth, self-leadership, skills
    GC  — 1 field:  culture feel (cultural identity)
    RFF — 5 fields: adjectives, career skills, SMART goal, hope to gain, ideal job
    CR  — 2 fields: C3 NACE extraction, C4 resume confirmation (+ passport description)
    CI, CT — pre-computed in source data, no LLM needed

Outputs:
    outputs/pathcredits_14_scored.csv     (pilot)
    outputs/pathcredits_199_scored.csv    (master)
"""

import os, sys, re, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Mode ──────────────────────────────────────────────────────────────────────
RUN_MASTER = "--master" in sys.argv

BASE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.join(BASE, "..")

DATA_14     = os.path.join(ROOT, "data", "STUDOR_DATA_6_FEATURES.xlsx")
DATA_MASTER = os.path.join(ROOT, "data", "FULL_199_students_MASTER_SHEET.csv")
OUT_DIR     = os.path.join(ROOT, "outputs")

INPUT_PATH  = DATA_MASTER if RUN_MASTER else DATA_14
OUTPUT_PATH = os.path.join(OUT_DIR, "pathcredits_199_scored.csv" if RUN_MASTER else "pathcredits_14_scored.csv")

print(f"Mode: {'MASTER (199)' if RUN_MASTER else 'PILOT (14)'}")

# ── LLM setup ─────────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
USE_LLM = bool(ANTHROPIC_KEY)
LLM_MODEL = "claude-haiku-4-5-20251001"

if USE_LLM:
    try:
        import anthropic
        _llm = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        print(f"LLM: ON  ({LLM_MODEL})")
    except ImportError:
        USE_LLM = False
        print("LLM: OFF (pip install anthropic)")
else:
    print("LLM: OFF (set ANTHROPIC_API_KEY to enable full scoring)")

# ── Import prompt registries ──────────────────────────────────────────────────
sys.path.insert(0, BASE)
from ec_llm_prompts  import PROMPT_REGISTRY       as EC_PROMPTS
from gc_llm_prompts  import GC_PROMPT_REGISTRY    as GC_PROMPTS
from rff_llm_prompts import RFF_PROMPT_REGISTRY   as RFF_PROMPTS
from cr_llm_prompts  import CR_PROMPT_REGISTRY    as CR_PROMPTS

# ── Load data ─────────────────────────────────────────────────────────────────
if INPUT_PATH.endswith(".xlsx"):
    df = pd.read_excel(INPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH, encoding="latin1", low_memory=False)

print(f"Loaded: {len(df)} students, {len(df.columns)} columns")

# ── Column resolver (pilot vs master name differences) ───────────────────────
def col(name, *fallbacks):
    for c in [name] + list(fallbacks):
        if c in df.columns:
            return c
    raise KeyError(f"Column not found: {name!r}")

COL_EVER_VOL  = col("FY1_Ever_Volunteered",  "FY1 Ever Volunteered.1", "FY1 Ever Volunteered")
COL_HOURS_VOL = col("FY1_Hours_Volunteered", "FY1 Hours Volunteered.1", "FY1 Hours Volunteered")

# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def norm17(s):   return (pd.to_numeric(s, errors="coerce").fillna(4) - 1) / 6
def norm15(s):   return (pd.to_numeric(s, errors="coerce").fillna(3) - 1) / 4
def norm010(s):  return  pd.to_numeric(s, errors="coerce").fillna(5) / 10
def norm110(s):  return (pd.to_numeric(s, errors="coerce").fillna(5) - 1) / 9

def to_binary(s):
    def _c(v):
        if pd.isna(v): return 0
        sv = str(v).strip().lower()
        if sv in ("1","true","yes","y","1.0"): return 1
        if sv in ("0","false","no","n","0.0"): return 0
        try: return 1 if float(sv) > 0 else 0
        except: return 0
    return s.apply(_c)

def text_substance(s):
    """Fallback scorer: blank=0, <10=0.3, 10-100=0.6, >100=1.0"""
    def _sc(v):
        if pd.isna(v): return 0.0
        t = str(v).strip()
        if not t: return 0.0
        if len(t) < 10: return 0.30
        if len(t) <= 100: return 0.60
        return 1.00
    return s.apply(_sc)

def hours_parse(s):
    def _p(v):
        if pd.isna(v): return 0
        sv = str(v).strip()
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", sv)
        if m: return (int(m.group(1)) + int(m.group(2))) / 2
        try: return float(sv)
        except: return 0
    return s.apply(_p)

def champions_count(s):
    def _c(v):
        if pd.isna(v): return 0
        sv = str(v).strip()
        try: return int(float(sv))
        except: return len([x for x in sv.split(",") if x.strip()])
    return s.apply(_c)

def is_multilingual(s):
    kw = ["and","&","/",",","french","spanish","creole","fulani",
          "yoruba","haitian","portuguese","mandarin","arabic","hindi"]
    return s.apply(lambda v: 1 if any(k in str(v).lower() for k in kw) else 0)

def english_level(s):
    m = {"native":5,"very comfortable":4,"comfortable":3,"somewhat":2,"not":1}
    def _m(v):
        t = str(v).lower() if not pd.isna(v) else ""
        for k, sc in m.items():
            if k in t: return sc
        return 3
    return s.apply(_m)

def tag_score(s):
    def _t(v):
        if pd.isna(v): return 0
        items = [x.strip() for x in re.split(r"[,\n]", str(v)) if x.strip()]
        return min(len(set(i.lower() for i in items if len(i) > 3)), 6) / 6
    return s.apply(_t)

# ═══════════════════════════════════════════════════════════════════════════════
# LLM CALLER
# ═══════════════════════════════════════════════════════════════════════════════

_cache = {}

def llm_call(prompt_text, expect_json=True, default=None):
    """Send prompt to Claude Haiku, return parsed JSON or plain text."""
    key = prompt_text[:200]
    if key in _cache:
        return _cache[key]
    for attempt in range(3):
        try:
            resp = _llm.messages.create(
                model=LLM_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt_text}]
            )
            raw = resp.content[0].text.strip()
            result = json.loads(raw) if expect_json else raw
            _cache[key] = result
            return result
        except Exception:
            if attempt < 2: time.sleep(1 + attempt)
    _cache[key] = default
    return default

def score_col(series, registry, prompt_key, score_field="score", default=3):
    """Score a pandas Series using an LLM prompt. Returns Series of scores (1-5)."""
    tpl = registry[prompt_key]
    scores = []
    for i, v in enumerate(series):
        if not USE_LLM:
            scores.append(default)
            continue
        if pd.isna(v) or str(v).strip() == "":
            scores.append(1)
            continue
        prompt = tpl.format(text=str(v)[:800])
        result = llm_call(prompt, expect_json=True, default={score_field: default})
        scores.append(int(result.get(score_field, default)) if result else default)
        time.sleep(0.05)
        if (i + 1) % 20 == 0:
            print(f"    [{prompt_key}] {i+1}/{len(series)}")
    return pd.Series(scores, index=series.index)

def score_col_struct(series, registry, prompt_key, default_obj=None):
    """Like score_col but returns full JSON dict per row (for multi-field prompts)."""
    tpl = registry[prompt_key]
    results = []
    for i, v in enumerate(series):
        if not USE_LLM or pd.isna(v) or str(v).strip() == "":
            results.append(default_obj or {})
            continue
        prompt = tpl.format(notes_text=str(v)[:1200])
        result = llm_call(prompt, expect_json=True, default=default_obj or {})
        results.append(result or default_obj or {})
        time.sleep(0.05)
        if (i + 1) % 20 == 0:
            print(f"    [{prompt_key}] {i+1}/{len(series)}")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 1 · EFFECTIVE COMMUNICATOR  (max 100)
# Scored by: ec_llm_prompts.py — 5 text fields
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── EC: Effective Communicator ───")

eng_15    = english_level(df["English - Spoken"])
champ_rat = norm110(df["Rate your Career Pathways Champion (1-10)"])
comm_feel = norm17(df["Community Feel (Quant)"])

# Text fields — LLM or substance fallback
if USE_LLM:
    print("  Scoring EC text fields...")
    sugg_llm   = score_col(df["Any suggestions to make the Foundational Year a better experience"],
                            EC_PROMPTS, "written_comm_quality", default=3)
    touch_llm  = score_col(df["Did you find a way to stay in touch"], EC_PROMPTS, "written_comm_quality", default=3)
    cross_llm  = score_col(df["Did you learn something about other careers from other Career Cohorts"],
                            EC_PROMPTS, "written_comm_quality", default=3)
    skill_llm  = score_col(df["What are three skills you have that will help you in your future career"],
                            EC_PROMPTS, "written_comm_quality", default=3)
    written_avg_llm = pd.concat([sugg_llm, touch_llm, cross_llm], axis=1).mean(axis=1)
    sugg_depth = sugg_llm
    skill_sub  = (skill_llm - 1) / 4
else:
    written_avg_llm = pd.concat([
        text_substance(df["Any suggestions to make the Foundational Year a better experience"]),
        text_substance(df["Did you find a way to stay in touch"]),
        text_substance(df["Did you learn something about other careers from other Career Cohorts"]),
    ], axis=1).mean(axis=1) * 4 + 1
    sugg_depth = written_avg_llm
    skill_sub  = text_substance(df["What are three skills you have that will help you in your future career"])

V = norm15(eng_15)*10 + champ_rat*10 + comm_feel*5 + norm15(sugg_depth)*5
W = norm15(written_avg_llm)*15 + (skill_sub if not USE_LLM else (skill_llm-1)/4)*5

I_s = (norm17(df["Listen to others.1"])*10 +
       norm17(df["Deal with conflicts - conflict management"])*10 +
       (pd.to_numeric(df["Deal with conflicts - conflict management"], errors="coerce").fillna(4) > 3).astype(int)*5)

C_s = (norm17(df["Include others who are different - diversity and inclusion"])*10 +
       to_binary(df["After meeting Champions, I better understand people who are different from me"])*5 +
       is_multilingual(df["Languages"])*3 +
       norm15(english_level(df["English - Spoken"]))*2)

EC_score = (V + W + I_s + C_s).clip(0, 100)
df["EC_Score"]       = EC_score.round(2)
df["EC_V_verbal"]    = V.round(2)
df["EC_W_written"]   = W.round(2)
df["EC_I_interpers"] = I_s.round(2)
df["EC_C_crosscult"] = C_s.round(2)
print(f"  mean={EC_score.mean():.1f}  std={EC_score.std():.1f}  min={EC_score.min():.1f}  max={EC_score.max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2 · GLOBAL CITIZEN  (max 100)
# Scored by: gc_llm_prompts.py — 1 text field (Culture Feel)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── GC: Global Citizen ───")

# Empathy (30%)
pre   = [norm17(df[c]) for c in ["Pre Empathy","Pre Humble","Pre Listen",
                                   "Pre Include Others Who Are Different",
                                   "Pre Deal with Conflicts","Pre Lead With Authenticity"]]
post  = [norm17(df[c]) for c in ["Listen to others.1",
                                   "Include others who are different - diversity and inclusion",
                                   "Deal with conflicts - conflict management",
                                   "Reflect if you have been in a similar situation as someone you are trying to help non positional leadership"]]
Empathy = sum(p*0.05 for p in pre) + sum(p*0.175 for p in post)

# Community (20%) — Culture Feel via LLM or substance
if USE_LLM:
    print("  Scoring GC Culture Feel...")
    cf_scores = score_col(df["Culture Feel"], GC_PROMPTS, "culture_feel", default=3)
    cult_feel = (cf_scores - 1) / 4
else:
    cult_feel = text_substance(df["Culture Feel"])

Community = norm17(df["Pre Community Connected"])*0.15 + norm17(df["Community Feel (Quant)"])*0.40 + cult_feel*0.45

# Cultural (20%)
Cultural = (norm17(df["I understand how my cultural values can shape my career choices.1"])*0.30 +
            to_binary(df["After meeting Champions, I better understand people who are different from me"])*0.30 +
            to_binary(df["Were you introduced to diverse career professionals"])*0.40)

# Network (20%)
Network = (to_binary(df["Meeting with my Champions during school helped me feel like I belong in school"])*0.15 +
           to_binary(df["This SPEAKHIRE Foundational Year helped me understand the value of building a strong network"])*0.15 +
           to_binary(df["I feel more engaged in school and participate more than before"])*0.15 +
           to_binary(df["I made new friends during the Foundational Year"])*0.15 +
           pd.to_numeric(df["How many individuals do you know who work in the career you are interested in"],
                         errors="coerce").fillna(0).clip(0,5)/5 * 0.40)

# Volunteering (10%)
Volunteering = (to_binary(df[COL_EVER_VOL])*0.35 +
                hours_parse(df[COL_HOURS_VOL]).clip(0,60)/60 * 0.65)

GC_score = (Empathy*0.30 + Community*0.20 + Cultural*0.20 + Network*0.20 + Volunteering*0.10)*100
GC_score = GC_score.clip(0, 100)
df["GC_Score"]        = GC_score.round(2)
df["GC_Empathy"]      = (Empathy*100).round(2)
df["GC_Community"]    = (Community*100).round(2)
df["GC_Cultural"]     = (Cultural*100).round(2)
df["GC_Network"]      = (Network*100).round(2)
df["GC_Volunteering"] = (Volunteering*100).round(2)
print(f"  mean={GC_score.mean():.1f}  std={GC_score.std():.1f}  min={GC_score.min():.1f}  max={GC_score.max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3 · CREATIVE INNOVATOR — pre-computed
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── CI: Creative Innovator (pre-computed) ───")
df["CI_Score"] = pd.to_numeric(df["Creative_Innovator_Score"], errors="coerce").round(2)
print(f"  mean={df['CI_Score'].mean():.1f}  std={df['CI_Score'].std():.1f}  min={df['CI_Score'].min():.1f}  max={df['CI_Score'].max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4 · CRITICAL THINKER — pre-computed
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── CT: Critical Thinker (pre-computed) ───")
df["CT_Score"] = pd.to_numeric(df["Critical Thinker Score (0-100)"], errors="coerce").round(2)
print(f"  mean={df['CT_Score'].mean():.1f}  std={df['CT_Score'].std():.1f}  min={df['CT_Score'].min():.1f}  max={df['CT_Score'].max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5 · REFLECTIVE & FUTURE FOCUSED  (max 100)
# Scored by: rff_llm_prompts.py — 5 text fields
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── RFF: Reflective & Future Focused ───")

if USE_LLM:
    print("  Scoring RFF text fields...")
    adj_n   = (score_col(df["What are three adjectives that describe the person you are and why"],
                          RFF_PROMPTS, "adjectives") - 1) / 4
    skill_n = (score_col(df["What are three skills you have that will help you in your future career"],
                          RFF_PROMPTS, "career_skills") - 1) / 4
    smart_n = (score_col(df["SMART GOAL"], RFF_PROMPTS, "smart_goal") - 1) / 4
    hope_n  = (score_col(df["What do you hope to gain by going through this program"],
                          RFF_PROMPTS, "hope_gain") - 1) / 4
    job_n   = (score_col(df["If you do not have a job, what is your ideal future career job"],
                          RFF_PROMPTS, "ideal_job") - 1) / 4
else:
    adj_n   = text_substance(df["What are three adjectives that describe the person you are and why"])
    skill_n = text_substance(df["What are three skills you have that will help you in your future career"])
    smart_n = text_substance(df["SMART GOAL"])
    hope_n  = text_substance(df["What do you hope to gain by going through this program"])
    job_n   = text_substance(df["If you do not have a job, what is your ideal future career job"])

SR = (adj_n*0.30 + skill_n*0.30 +
      tag_score(df["What do you hope to gain by going through this program"])*0.20 +
      tag_score(df["Hope to Gain"])*0.20)

GS = (smart_n*0.40 +
      text_substance(df["Set your SMART Goal"])*0.35 +
      text_substance(df["Remember the SMART Goal you set - next round"])*0.25)

FC = (norm17(df["Know How To Pursue Careers"])*0.40 +
      to_binary(df["I feel more prepared for my future career"])*0.30 +
      job_n*0.30)

CP = (norm17(df["I feel ready and prepared for college"])*0.20 +
      norm17(df["FY1 Feel College Ready and Prepped"])*0.20 +
      pd.to_numeric(df["I feel I am now a stronger candidate for college and careers"], errors="coerce").fillna(5)/10 * 0.20 +
      pd.to_numeric(df["I feel I am now more prepared for college"], errors="coerce").fillna(5)/10 * 0.20 +
      to_binary(df["FY helped realize doing well connects to my career goals"])*0.20)

ST = ((norm010(df["The Speaker inspired me to think more about my future career"]) +
       norm010(df["The Speaker helped me think about my future career pathway"]) +
       norm010(df["The Speaker was a relatable role model"]) +
       norm010(df["The topic inspired me to think more about my future career"]) +
       norm010(df["The topic helped me think about my future career pathway"])) / 5)

RFF_score = (SR*0.25 + GS*0.20 + FC*0.25 + CP*0.15 + ST*0.15)*100
RFF_score = RFF_score.clip(0, 100)
df["RFF_Score"]          = RFF_score.round(2)
df["RFF_SelfReflection"] = (SR*100).round(2)
df["RFF_GoalSetting"]    = (GS*100).round(2)
df["RFF_FutureCareer"]   = (FC*100).round(2)
df["RFF_CollegePrep"]    = (CP*100).round(2)
df["RFF_SpeakerTopic"]   = (ST*100).round(2)
print(f"  mean={RFF_score.mean():.1f}  std={RFF_score.std():.1f}  min={RFF_score.min():.1f}  max={RFF_score.max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6 · CAREER READY  (max 100)
# Scored by: cr_llm_prompts.py — C3 NACE extraction + C4 resume confirmation
# Adapted from Rakshana's Gemini pipeline (career_ready_pipeline.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── CR: Career Ready ───")

# C1 — Pre-Program Exposure
ever_vol_c1 = to_binary(df[COL_EVER_VOL])
hours_c1    = hours_parse(df[COL_HOURS_VOL]).clip(0,60)/60
had_intern  = to_binary(df["Mentee_Had_Internship"])
had_job     = to_binary(df["Mentee_Had_Job"])
C1 = (ever_vol_c1*0.25 + hours_c1*0.25 + had_intern*0.25 + had_job*0.25)*100

# C2 — Foundation Building (cohort-relative)
sessions_raw = pd.to_numeric(df["Total_Sessions_Attended"], errors="coerce").fillna(0)
cohort_max   = max(sessions_raw.max(), 1)
sessions_n   = (sessions_raw / cohort_max).clip(0, 1)
champ_n      = champions_count(df["Connected_Champions"]).clip(0,6)/6
C2 = (sessions_n*0.65 + champ_n*0.35)*100

# C3 — Skills Developed (LLM NACE extraction or keyword fallback)
NACE_KW = {
    "Communication":        ["communicat","present","speak","articulate","express","verbal","email","writing"],
    "Critical Thinking":    ["analyz","research","problem","solution","evaluat","assess","critical","think"],
    "Teamwork":             ["team","collaborat","group","together","peer","partner","cooperat"],
    "Leadership":           ["lead","initiative","manag","organiz","guide","motivat","mentor"],
    "Professionalism":      ["professional","punctual","responsib","deadline","work ethic","reliab","commit"],
    "Career Development":   ["career","goal","pathway","network","interview","job","future","opportunit","skill"],
}

def nace_keyword_score(v):
    if pd.isna(v): return 0
    sv = str(v).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", sv) or len(sv) < 10: return 0
    found = sum(1 for kws in NACE_KW.values() if any(kw in sv for kw in kws))
    return round(found / len(NACE_KW) * 100, 1)

if USE_LLM:
    print("  Scoring CR C3/C4 with LLM...")
    c3_results = score_col_struct(df["CPC_Resume_Text"], CR_PROMPTS, "c3_nace",
                                   default_obj={"score": 0, "nace_categories": [], "dominant_skills": []})
    C3 = pd.Series([r.get("score", 0) for r in c3_results], index=df.index)
    df["CR_C3_NACE_categories"] = [", ".join(r.get("nace_categories", [])) for r in c3_results]
    df["CR_C3_dominant_skills"] = [", ".join(r.get("dominant_skills", [])) for r in c3_results]

    c4_results = score_col_struct(df["CPC_Resume_Text"], CR_PROMPTS, "c4_resume",
                                   default_obj={"score": 0, "resume_built": False, "confidence": "none", "key_evidence": None})
    C4 = pd.Series([r.get("score", 0) for r in c4_results], index=df.index)
    df["CR_C4_resume_built"]    = [r.get("resume_built", False) for r in c4_results]
    df["CR_C4_confidence"]      = [r.get("confidence", "none") for r in c4_results]
    df["CR_C4_key_evidence"]    = [r.get("key_evidence", "") for r in c4_results]
else:
    C3 = df["CPC_Resume_Text"].apply(nace_keyword_score)
    # Resume keyword fallback for C4
    RESUME_KW = ["resume","cv","objective","experience","skill","education","award","activit","format","bullet"]
    def resume_kw_score(v):
        if pd.isna(v): return 0
        sv = str(v).strip().lower()
        if re.fullmatch(r"\d+(\.\d+)?", sv) or len(sv) < 10: return 0
        hits = sum(1 for kw in RESUME_KW if kw in sv)
        if hits == 0: return 0
        if hits <= 1: return 33
        if hits <= 3: return 67
        return 100
    C4 = df["CPC_Resume_Text"].apply(resume_kw_score).astype(float)

CR_score = ((C1 + C2 + C3 + C4) / 4).clip(0, 100)
df["CR_Score"]         = CR_score.round(2)
df["CR_C1_Exposure"]   = C1.round(2)
df["CR_C2_Foundation"] = C2.round(2)
df["CR_C3_Skills"]     = C3.round(2)
df["CR_C4_Resume"]     = C4.round(2)
print(f"  mean={CR_score.mean():.1f}  std={CR_score.std():.1f}  min={CR_score.min():.1f}  max={CR_score.max():.1f}")
print(f"  C3==C4 in {(df['CR_C3_Skills']==df['CR_C4_Resume']).sum()}/{len(df)} rows")

# ── Passport descriptions (LLM only) ──────────────────────────────────────────
if USE_LLM:
    print("  Generating passport descriptions...")
    passports = []
    for _, row in df.iterrows():
        nace = row.get("CR_C3_NACE_categories", "")
        hrs  = f"({int(hours_parse(pd.Series([row[COL_HOURS_VOL]])).iloc[0])}h)" if hours_parse(pd.Series([row[COL_HOURS_VOL]])).iloc[0] > 0 else ""
        prompt = CR_PROMPTS["passport"].format(
            sessions_attended=int(pd.to_numeric(row.get("Total_Sessions_Attended", 0), errors="coerce") or 0),
            champion_count=champions_count(pd.Series([row["Connected_Champions"]])).iloc[0],
            volunteered="Yes" if to_binary(pd.Series([row[COL_EVER_VOL]])).iloc[0] else "No",
            hours_str=hrs,
            had_internship="Yes" if to_binary(pd.Series([row["Mentee_Had_Internship"]])).iloc[0] else "No",
            had_job="Yes" if to_binary(pd.Series([row["Mentee_Had_Job"]])).iloc[0] else "No",
            nace_competencies=nace or "not recorded",
            resume_confirmed=row.get("CR_C4_resume_built", False),
            career_ready_score=round(row.get("CR_Score", 0)),
            top_label=f"Top {100 - int(df['CR_Score'].rank(pct=True).loc[row.name]*100)+1}%"
        )
        desc = llm_call(prompt, expect_json=False, default="Career Ready score computed from program data.")
        passports.append(desc)
        time.sleep(0.05)
    df["CR_passport_description"] = passports

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
base_cols = [
    "Student Name",
    "EC_Score","EC_V_verbal","EC_W_written","EC_I_interpers","EC_C_crosscult",
    "GC_Score","GC_Empathy","GC_Community","GC_Cultural","GC_Network","GC_Volunteering",
    "CI_Score",
    "CT_Score",
    "RFF_Score","RFF_SelfReflection","RFF_GoalSetting","RFF_FutureCareer","RFF_CollegePrep","RFF_SpeakerTopic",
    "CR_Score","CR_C1_Exposure","CR_C2_Foundation","CR_C3_Skills","CR_C4_Resume",
]
extra = [c for c in ["CR_C3_NACE_categories","CR_C3_dominant_skills",
                      "CR_C4_resume_built","CR_C4_confidence","CR_C4_key_evidence",
                      "CR_passport_description"] if c in df.columns]

out = df[base_cols + extra].copy()
out["PathCredits_Score"] = ((out["EC_Score"]+out["GC_Score"]+out["CI_Score"]+
                              out["CT_Score"]+out["RFF_Score"]+out["CR_Score"])/6).round(2)
out = out.sort_values("PathCredits_Score", ascending=False).reset_index(drop=True)
out.index += 1
out.index.name = "Rank"
out.to_csv(OUTPUT_PATH)
print(f"\n✓ Saved → {OUTPUT_PATH}")

# ── Print table ───────────────────────────────────────────────────────────────
print("\n" + "="*80)
disp = ["Student Name","EC_Score","GC_Score","CI_Score","CT_Score","RFF_Score","CR_Score","PathCredits_Score"]
print(out[disp].to_string())

print("\n── Score Summary ──")
for c in ["EC_Score","GC_Score","CI_Score","CT_Score","RFF_Score",
          "CR_Score","CR_C3_Skills","CR_C4_Resume","PathCredits_Score"]:
    s = out[c]
    print(f"  {c:30s}  mean={s.mean():.1f}  std={s.std():.1f}  min={s.min():.1f}  max={s.max():.1f}")
print(f"\n  LLM used: {USE_LLM}")
