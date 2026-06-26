from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RECOMMENDER_DIR = os.path.dirname(_SCRIPT_DIR)
_CORPUS_OUTPUT = os.path.join(_SCRIPT_DIR, "our_jobs.parquet")
_ONET_DIR = os.path.join(_RECOMMENDER_DIR, "data", "onet")
_ONET_ZIP_URL = "https://www.onetcenter.org/dl_files/database/db_30_3_excel.zip"


_TARGET_FUNCTIONS = [
    "marketing", "education", "healthcare", "ops", "support", "design",
    "sales", "skilled-trade", "technology", "food-service", "administrative",
    "finance", "logistics", "hospitality", "manufacturing",
]


def corpus_exists() -> bool:
    """Only return True when ALL expected function parquets exist."""
    for func in _TARGET_FUNCTIONS:
        if not os.path.exists(os.path.join(_SCRIPT_DIR, f"{func}.parquet")):
            return False
    return True


def onet_data_exists() -> bool:
    path = os.path.join(_ONET_DIR, "Occupation Data.xlsx")
    return os.path.exists(path)


def download_and_build_corpus() -> dict:
    steps = []

    if not onet_data_exists():
        _download_onet_files()
        steps.append("onet_downloaded")

    rows: list[dict] = []
    rows += _build_onet_rows()
    steps.append(f"onet_rows={len(rows)}")

    oj_rows = _download_open_jobs_rows()
    rows += oj_rows
    steps.append(f"open_jobs_rows={len(oj_rows)}")

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(_CORPUS_OUTPUT), exist_ok=True)

        for func in df["function"].dropna().unique():
            func_df = df[df["function"] == func]
            path = os.path.join(_SCRIPT_DIR, f"{func}.parquet")
            func_df.to_parquet(path, index=False)
        size_mb = round(os.path.getsize(_CORPUS_OUTPUT) / (1024 * 1024), 1) if os.path.exists(_CORPUS_OUTPUT) else 0
        steps.append(f"total_rows={len(rows)}")
        steps.append(f"disk_mb={size_mb}")

    return {
        "status": "ok",
        "corpus_path": _CORPUS_OUTPUT,
        "total_rows": len(rows),
        "steps": steps,
    }


def _download_onet_files() -> None:
    import urllib.request
    import zipfile
    import io
    import shutil

    os.makedirs(_ONET_DIR, exist_ok=True)
    print(f"Downloading O*NET 30.3 from {_ONET_ZIP_URL} ...")
    req = urllib.request.Request(_ONET_ZIP_URL)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    print(f"Extracting to {_ONET_DIR} ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith(".xlsx") and "db_30_3_excel/" in name:
                basename = os.path.basename(name)
                with zf.open(name) as src:
                    dest = os.path.join(_ONET_DIR, basename)
                    if os.path.exists(dest):
                        continue
                    with open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    print(f"O*NET files ready in {_ONET_DIR}")


def _build_onet_rows() -> list[dict]:
    rows = []
    for entry in _ONET_OCCUPATIONS:
        rows.append({
            "id": f"onet_{entry['function'].replace(' ', '_').lower()}",
            "ats": "onet",
            "company": "",
            "title": f"{entry['function']} ({entry['level']})",
            "url": "",
            "jd_markdown": entry["description"],
            "level": entry["level"],
            "function": entry["function"],
            "skills": entry["skills"],
            "country_code": "US",
            "jd_embedding": None,
        })
    return rows


def _download_open_jobs_rows() -> list[dict]:
    import time as _time

    max_retries = 2
    source = os.environ.get(
        "OPEN_JOBS_URL",
        "https://download.jobscream.com/open-jobs.parquet",
    )
    target_functions = [
        "marketing", "education", "healthcare", "support",
        "ops", "design", "sales", "hr", "skilled-trade",
        "technology", "food-service", "logistics", "hospitality",
        "administrative", "finance", "manufacturing",
    ]
    target_levels = ["intern", "entry", "junior", ""]

    for attempt in range(max_retries + 1):
        try:
            import fsspec
            import pyarrow.parquet as pq
            import pandas as pd

            if attempt > 0:
                wait = 2 ** attempt
                print(f"Retry {attempt}/{max_retries} after {wait}s ...")
                _time.sleep(wait)

            print(f"Streaming and filtering {source} ...")
            fs = fsspec.filesystem("http", headers={"User-Agent": "Mozilla/5.0"})
            fh = fs.open(source, "rb")
            pf = pq.ParquetFile(fh)

            total_rows = pf.metadata.num_rows
            print(f"  {total_rows:,} rows, {pf.metadata.num_row_groups} row groups")

            matched = []
            for i in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(i, columns=[
                    "id", "ats", "company", "title", "url", "jd_markdown",
                    "level", "function", "skills", "country_code",
                ])
                df = table.to_pandas()
                df = df[df["function"].str.lower().isin(target_functions)]
                df = df[df["level"].str.lower().isin(target_levels)]
                df = df[df["country_code"].str.upper().isin(["US", ""])]
                matched.append(df)
                print(f"  {i+1}/{pf.metadata.num_row_groups}, matched: {sum(len(r) for r in matched):>6,}", end="\r")

            if matched:
                result = pd.concat(matched, ignore_index=True)
                fh.close()
                print(f"\nDownloaded {len(result):,} rows (filtered from {total_rows:,})")
                return result.to_dict("records")
            fh.close()
            print(f"\nNo rows matched ({total_rows:,} scanned)")
            return []
        except Exception as e:
            print(f"\nAttempt {attempt}: streaming failed: {e}")
            if attempt == max_retries:
                import traceback
                traceback.print_exc()
                print("All retries exhausted. Corpus will be built from O*NET data only.")
                return []
    return []


_ONET_OCCUPATIONS: list[dict] = [
    {
        "function": "marketing",
        "level": "Entry",
        "description": "Plan and implement marketing campaigns, create social media content, write newsletters, design promotional materials, and coordinate events.",
        "skills": ["content creation", "social media management", "graphic design", "writing", "public speaking", "photography", "event planning", "data entry", "customer service"],
    },
    {
        "function": "education",
        "level": "Entry",
        "description": "Mentor and tutor students, develop educational activities, facilitate after-school programs, and support classroom instruction.",
        "skills": ["mentoring", "youth engagement", "teaching", "writing", "public speaking", "program management", "event planning", "data entry", "content creation"],
    },
    {
        "function": "healthcare",
        "level": "Entry",
        "description": "Provide patient care, assist with intake and record-keeping, coordinate health screening events, and support clinical operations.",
        "skills": ["healthcare", "community outreach", "data entry", "customer service", "public speaking", "writing", "program management", "volunteer coordination", "mentoring"],
    },
    {
        "function": "ops",
        "level": "Entry",
        "description": "Coordinate volunteer programs, plan events, manage logistics, track budgets, and communicate with stakeholders.",
        "skills": ["volunteer coordination", "event planning", "community outreach", "program management", "fundraising", "writing", "data entry", "public speaking", "mentoring"],
    },
    {
        "function": "support",
        "level": "Entry",
        "description": "Respond to customer inquiries via phone, email, and chat. Process orders, resolve complaints, maintain customer records.",
        "skills": ["customer service", "data entry", "writing", "public speaking", "content creation", "social media management"],
    },
    {
        "function": "design",
        "level": "Entry",
        "description": "Create visual content for digital and print media, design graphics for campaigns, produce photography and video content.",
        "skills": ["graphic design", "content creation", "photography", "social media management", "writing", "data entry", "public speaking"],
    },
    {
        "function": "sales",
        "level": "Entry",
        "description": "Assist customers, process transactions, meet sales targets, manage inventory, and build client relationships in retail or B2B environments.",
        "skills": ["sales", "customer service", "communication", "inventory management", "data entry", "time management", "leadership"],
    },
    {
        "function": "technology",
        "level": "Entry",
        "description": "Develop software, debug issues, write code, collaborate on technical projects, and support IT systems and infrastructure.",
        "skills": ["software & technical", "problem solving", "data analysis", "communication", "project management", "time management"],
    },
    {
        "function": "skilled-trade",
        "level": "Entry",
        "description": "Perform construction, electrical, plumbing, HVAC, welding, or mechanical work. Read blueprints, operate tools and heavy equipment.",
        "skills": ["trades & physical", "certifications", "problem solving", "time management", "inventory management"],
    },
    {
        "function": "food-service",
        "level": "Entry",
        "description": "Prepare food, cook meals, serve customers, maintain kitchen cleanliness, manage inventory, and follow food safety protocols.",
        "skills": ["food service", "customer service", "time management", "certifications", "inventory management"],
    },
    {
        "function": "administrative",
        "level": "Entry",
        "description": "Provide office support, manage calendars, answer phones, file documents, enter data, and coordinate meetings and travel.",
        "skills": ["data entry", "scheduling", "communication", "customer service", "writing", "time management", "problem solving"],
    },
    {
        "function": "finance",
        "level": "Entry",
        "description": "Process financial transactions, maintain ledgers, reconcile accounts, prepare reports, and support budgeting and accounting operations.",
        "skills": ["budgeting & finance", "data entry", "data analysis", "problem solving", "communication", "time management"],
    },
    {
        "function": "logistics",
        "level": "Entry",
        "description": "Coordinate shipping and receiving, manage warehouse inventory, operate forklifts, plan delivery routes, and track shipments.",
        "skills": ["logistics & driving", "inventory management", "time management", "customer service", "certifications"],
    },
    {
        "function": "hospitality",
        "level": "Entry",
        "description": "Welcome guests, manage front desk, coordinate events, handle reservations, and provide excellent customer experiences in hotels and tourism.",
        "skills": ["customer service", "communication", "time management", "event planning", "languages", "scheduling"],
    },
    {
        "function": "manufacturing",
        "level": "Entry",
        "description": "Operate production machinery, assemble products, perform quality checks, maintain safety standards, and follow lean manufacturing processes.",
        "skills": ["manufacturing", "problem solving", "time management", "certifications", "inventory management", "trades & physical"],
    },
]


if __name__ == "__main__":
    result = download_and_build_corpus()
    print(result)
