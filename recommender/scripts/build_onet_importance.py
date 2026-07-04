"""Build O*NET importance-weighted skill index from Essential/Transferable/Knowledge skills."""
import json, os, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ONET_DIR = os.path.join(HERE, "..", "data", "onet_full")
OUT = os.path.join(HERE, "..", "data", "onet_importance.json")

SKILL_FILES = {
    "Essential Skills.xlsx": "essential",
    "Transferable Skills.xlsx": "transferable",
    "Knowledge.xlsx": "knowledge",
}

def build():
    index = {}  # {soc: {skill_name: importance, ...}}
    skill_names = set()

    for fname, source in SKILL_FILES.items():
        path = os.path.join(ONET_DIR, fname)
        if not os.path.exists(path):
            print(f"SKIP: {fname} not found")
            continue
        df = pd.read_excel(path)
        # Filter to Importance scores only
        im_df = df[df["Scale ID"] == "IM"]
        for _, row in im_df.iterrows():
            soc = str(row["O*NET-SOC Code"]).strip()
            skill = str(row["Element Name"]).strip()
            value = float(row["Data Value"])
            if not soc or not skill:
                continue
            index.setdefault(soc, {})[skill] = round(value, 2)
            skill_names.add(skill)

    # Add alternate titles from Occupation Data
    occ_path = os.path.join(ONET_DIR, "Occupation Data.xlsx")
    titles = {}
    if os.path.exists(occ_path):
        odf = pd.read_excel(occ_path)
        for _, row in odf.iterrows():
            soc = str(row.get("O*NET-SOC Code", "")).strip()
            title = str(row.get("Title", "")).strip()
            if soc and title:
                titles[soc] = title

    result = {
        "occupations": {soc: {"title": titles.get(soc, ""), "skills": skills}
                        for soc, skills in index.items()},
        "skill_names": sorted(skill_names),
        "num_occupations": len(index),
        "num_skills": len(skill_names),
        "sources": list(SKILL_FILES.values()),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Built importance index: {len(index)} occupations, {len(skill_names)} skills")
    print(f"Written to {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
    return result

if __name__ == "__main__":
    build()
