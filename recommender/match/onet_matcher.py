"""
O*NET-based occupation matching via Task Statement keyword indexing.

Bridges formal O*NET language → informal resume language by using
18,796 O*NET Task Statements (actual job activities) as the vocabulary.

Build: 2-second index, <1ms per query. No models, no downloads, no API.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
_INDEX_PATH = os.path.join(_DATA, "onet_occ_index.json")

# Caches
_occ_index: dict[str, Any] | None = None  # {occs: [...], keywords: {...}, num_occs: N}

_STOPS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "or", "not",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "who", "whom", "which", "what", "when",
    "where", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "if", "then", "else", "about",
    "up", "out", "over", "under", "again", "further", "once", "here",
    "there", "activities", "may", "work", "use", "using", "equipment",
    "information", "develop", "prepare", "provide", "materials",
    "determine", "ensure", "evaluate", "maintain", "programs", "procedures",
    "perform", "reports", "conduct", "research", "control", "plan", "test",
    "monitor", "direct", "review", "plans", "tools", "required", "including",
    "related", "knowledge", "principles", "techniques", "methods", "needs",
    "appropriate", "necessary", "standards", "results", "processes", "services",
    "records", "record", "supervise", "inspect", "specifications", "clean",
    "coordinate", "identify", "quality", "customer",
}

# Occupation title → function mapping (word-boundary aware regex)
_FUNC_MAP: list[tuple[str, str]] = [
    (r"\bSoftware\b", "technology"), (r"\bDeveloper", "technology"),
    (r"\bEngineer(?!$)", "technology"), (r"\bComputer\b", "technology"),
    (r"\bData\b", "technology"), (r"\bNetwork", "technology"),
    (r"\bWeb\b", "technology"), (r"\bIT\b", "technology"),
    (r"\bDatabase", "technology"), (r"\bProgrammer", "technology"),
    (r"\bMarketing\b", "marketing"), (r"\bSales\b", "sales"),
    (r"\bTeacher", "education"), (r"\bInstructor", "education"),
    (r"\bEducation", "education"), (r"\bTraining\b", "education"),
    (r"\bProfessor", "education"), (r"\bLibrarian", "education"),
    (r"\bSchool\b", "education"), (r"\bCurriculum\b", "education"),
    (r"\bNurse", "healthcare"), (r"\bMedical\b", "healthcare"),
    (r"\bHealth\b", "healthcare"), (r"\bClinical\b", "healthcare"),
    (r"\bPhysician", "healthcare"), (r"\bTherapist", "healthcare"),
    (r"\bDental\b", "healthcare"), (r"\bSurgeon", "healthcare"),
    (r"\bPharmacy", "healthcare"), (r"\bVeterinar", "healthcare"),
    (r"\bCustomer Service\b", "support"), (r"\bSupport\b", "support"),
    (r"\bDesigner", "design"), (r"\bDesign\b", "design"),
    (r"\bGraphic\b", "design"), (r"\bArtist", "design"),
    (r"\bManager", "ops"), (r"\bOperations\b", "ops"),
    (r"\bDirector\b", "ops"), (r"\bSupervisor", "ops"),
    (r"\bAdministrative\b", "administrative"), (r"\bOffice\b", "administrative"),
    (r"\bClerk", "administrative"), (r"\bReceptionist", "administrative"),
    (r"\bSecretar", "administrative"),
    (r"\bAccountant", "finance"), (r"\bFinancial\b", "finance"),
    (r"\bAuditor", "finance"), (r"\bBookkeeping\b", "finance"),
    (r"\bBudget", "finance"), (r"\bTax\b", "finance"),
    (r"\bChef\b", "food-service"), (r"\bCook(?!ing)", "food-service"),
    (r"\bRestaurant\b", "food-service"), (r"\bKitchen\b", "food-service"),
    (r"\bFood\b", "food-service"), (r"\bBarista", "food-service"),
    (r"\bWait", "food-service"), (r"\bHost(?!ler)", "food-service"),
    (r"\bBartender", "food-service"), (r"\bDishwasher", "food-service"),
    (r"\bConstruction\b", "skilled-trade"), (r"\bElectrician", "skilled-trade"),
    (r"\bPlumber", "skilled-trade"), (r"\bCarpenter", "skilled-trade"),
    (r"\bWelder", "skilled-trade"), (r"\bHVAC\b", "skilled-trade"),
    (r"\bRoofer", "skilled-trade"), (r"\bDrywall\b", "skilled-trade"),
    (r"\bMason", "skilled-trade"), (r"\bPainter", "skilled-trade"),
    (r"\bPipefitter", "skilled-trade"), (r"\bMechanic", "skilled-trade"),
    (r"\bDriver\b", "logistics"), (r"\bTruck\b", "logistics"),
    (r"\bLogistics\b", "logistics"), (r"\bWarehouse\b", "logistics"),
    (r"\bDelivery\b", "logistics"), (r"\bForklift\b", "logistics"),
    (r"\bManufacturing\b", "manufacturing"), (r"\bProduction\b", "manufacturing"),
    (r"\bAssembly\b", "manufacturing"), (r"\bMachine\b", "manufacturing"),
    (r"\bMachinist", "manufacturing"), (r"\bQuality Control\b", "manufacturing"),
    (r"\bHospitality\b", "hospitality"), (r"\bHotel\b", "hospitality"),
    (r"\bMotel\b", "hospitality"), (r"\bResort\b", "hospitality"),
    (r"\bLawyer", "legal"), (r"\bAttorney", "legal"),
    (r"\bParalegal", "legal"), (r"\bJudge\b", "legal"),
    (r"\bLegal\b", "legal"), (r"\bLaw\b", "legal"),
    (r"\bWriter", "arts-media"), (r"\bAuthor\b", "arts-media"),
    (r"\bJournalist", "arts-media"), (r"\bReporter", "arts-media"),
    (r"\bEditor\b", "arts-media"), (r"\bProducer", "arts-media"),
    (r"\bActor", "arts-media"), (r"\bMusician", "arts-media"),
    (r"\bPhotographer", "arts-media"), (r"\bDancer", "arts-media"),
    (r"\bAthlete", "arts-media"), (r"\bCoach\b", "arts-media"),
    (r"\bMedia\b", "arts-media"), (r"\bFilm\b", "arts-media"),
    (r"\bBroadcast", "arts-media"), (r"\bPublic Relations\b", "arts-media"),
    (r"\bPolice\b", "protective-service"), (r"\bFirefighter", "protective-service"),
    (r"\bSecurity\b", "protective-service"), (r"\bCorrection", "protective-service"),
    (r"\bDetective", "protective-service"), (r"\bGuard\b", "protective-service"),
    (r"\bEMT\b", "protective-service"), (r"\bParamedic", "protective-service"),
    (r"\bDispatche", "protective-service"),
    (r"\bJanitor", "building-grounds"), (r"\bCleaner", "building-grounds"),
    (r"\bMaid", "building-grounds"), (r"\bHousekeep", "building-grounds"),
    (r"\bLandscap", "building-grounds"), (r"\bGroundskeeper", "building-grounds"),
    (r"\bPest Control\b", "building-grounds"),
    (r"\bHairdresser", "personal-care"), (r"\bBarber", "personal-care"),
    (r"\bCosmetolog", "personal-care"), (r"\bChildcare\b", "personal-care"),
    (r"\bChild Care\b", "personal-care"), (r"\bNanny", "personal-care"),
    (r"\bFuneral", "personal-care"), (r"\bAnimal Care", "personal-care"),
    (r"\bManicurist", "personal-care"), (r"\bEsthetician", "personal-care"),
    (r"\bFarm", "agriculture"), (r"\bAgricultur", "agriculture"),
    (r"\bForest", "agriculture"), (r"\bFisher", "agriculture"),
    (r"\bLogger", "agriculture"),
    (r"\bScientist", "science"), (r"\bChemist", "science"),
    (r"\bBiologist", "science"), (r"\bPhysicist", "science"),
    (r"\bResearcher", "science"), (r"\bResearch\b", "science"),
    (r"\bLaboratory\b", "science"), (r"\bLab\b", "science"),
    (r"\bStatistician", "science"), (r"\bEconomist", "science"),
    (r"\bPsychologist", "science"), (r"\bSociologist", "science"),
    (r"\bHistorian", "science"), (r"\bGeographer", "science"),
    (r"\bSocial Work", "social-service"), (r"\bCounselor", "social-service"),
    (r"\bClergy", "social-service"), (r"\bProbation", "social-service"),
    (r"\bRehabilitation\b", "social-service"), (r"\bCommunity\b", "social-service"),
]


_FUNC_BONUS_WORDS: dict[str, list[str]] = {
    "administrative": ["calendars", "ledgers", "invoices", "postage", "principal", "agendas", "minutes", "appointment", "memos", "mailing", "accounts", "mail", "bookkeeping", "payment", "documents", "account", "affix", "transactions", "file", "answer", "correspondence", "telephones", "proofread", "expedite", "inquiries"],
    "agriculture": ["seed", "seedlings", "agricultural", "harvesting", "livestock", "planting", "farmers", "logging", "lands", "thinning", "seeds", "insects", "forest", "farm", "forestry", "weed", "insect", "growers", "competing", "planted", "nursery", "reforestation", "grazing", "trees", "weeds"],
    "arts-media": ["broadcast", "cues", "dramatic", "stories", "shot", "rehearsal", "roles", "broadcasts", "edit", "live", "actors", "plays", "acting", "writers", "productions", "story", "producers", "scripts", "films", "news", "audition", "footage", "filming", "audience", "musical"],
    "building-grounds": ["carpets", "scrubbing", "squeegees", "driveways", "mow", "mops", "cleaners", "representations", "insecticides", "infestation", "mowers", "lawns", "trimmers", "sewers", "shovels", "conserve", "landscaping", "sidewalks", "scrub", "brooms", "basins", "mud", "drainage", "vacuuming", "ditches"],
    "design": ["decoration", "illustrations", "artwork", "ideas", "creations", "looks", "sketch", "characters", "success", "intended", "concepts", "render", "illustration", "animation", "script", "arrangement", "brochures", "designs", "sketches", "art", "shows", "venues", "backgrounds", "drawings", "three"],
    "education": ["internship", "advisers", "classroom", "class", "homework", "bibliographies", "campus", "moderate", "syllabi", "classrooms", "curricula", "teacher", "textbooks", "experiential", "handouts", "vocational", "faculty", "lessons", "supplement", "provides", "academic", "persevere", "challenging", "teaching", "lectures"],
    "finance": ["quickbooks", "payroll", "journal entries", "reconciled", "bookkeeping", "general ledger", "invoicing", "accountants", "securities", "income", "valuation", "investment", "owed", "influences", "traders", "informal", "capital", "audit", "debt", "liabilities", "investments", "returns", "quantify", "trading", "deductions", "assets", "stocks", "finance", "financial", "tax", "estate", "theory"],
    "food-service": ["kitchen", "utensils", "coffee", "cook", "drinks", "sandwiches", "kitchens", "salads", "foods", "desserts", "silverware", "menus", "cooking", "dishes", "cooks", "drink", "beverages", "breads", "garnish", "soups", "meats", "cooked", "served", "taste", "pastries"],
    "healthcare": ["immunizations", "caregivers", "respiratory", "wounds", "electrocardiograms", "patient", "ekgs", "intravenous", "patients", "therapy", "urine", "diagnosis", "nursing", "pain", "medication", "acute", "operative", "surgery", "treatment", "medical", "vital", "medicine", "interventions", "chronic", "physicians"],
    "hospitality": ["guests", "reservations", "checking in", "checked in", "check ins", "cleaned rooms", "amenities", "restocked", "opera", "guest", "check-in", "checkout", "reservation", "housekeeping", "front desk", "concierge"],
    "legal": ["westlaw", "deposition", "discovery", "litigation", "notary", "wills", "briefs", "statutes", "appeals", "facts", "trial", "witnesses", "disputes", "legislation", "search", "court", "law", "decisions", "file", "cases", "contracts", "gather", "legal", "documents", "clients", "public", "meet", "analyze", "data"],
    "logistics": ["loaded", "warehouse", "plugs", "tank", "weighing", "loads", "navigation", "contents", "gps", "delivered", "spark", "drivers", "unload", "capacities", "bulbs", "connecting", "safely", "tire", "railroad", "lower", "transport", "vehicles", "trucks", "global", "tires"],
    "manufacturing": ["spindles", "jammed", "handwheels", "coolants", "arbors", "tensions", "guides", "conveyors", "speeds", "sharpen", "extruded", "cams", "strokes", "feed", "workpieces", "tend", "stops", "turn", "machine", "holding", "setup", "dies", "hoppers", "thread", "start"],
    "marketing": ["strategy", "pricing", "forecasting", "analyzing", "buying", "satisfaction", "developers", "text", "demand", "promotion", "markets", "marketing", "communications", "promotional", "trade", "company", "trends", "sales", "integrate", "affecting", "commercial", "strategies", "advertising", "vendors", "surveys"],
    "ops": ["disciplinary", "hires", "budgets", "hire", "promotions", "employee", "worker", "recruit", "grievances", "employees", "investors", "harassment", "vacancies", "ranges", "experienced", "complaints", "staffing", "manage", "managers", "operational", "company", "timely", "oversee", "policies", "transfers"],
    "personal-care": ["caskets", "pallbearers", "beards", "razors", "scalp", "sanitize", "casket", "cremations", "funerals", "burial", "floral", "bereaved", "shampoo", "clippers", "lotions", "funeral", "nail", "caring", "shave", "scissors", "flowers", "liquid", "skin", "homes", "laundry"],
    "protective-service": ["tampering", "secured", "suspects", "police", "fingerprints", "apprehend", "patrol", "violators", "survivors", "disturbances", "inmates", "correctional", "inmate", "prohibited", "persons", "crimes", "guard", "suspicious", "officers", "crime", "search", "occurrences", "prisoners", "entrances", "valuables"],
    "sales": ["warranties", "reorder", "prospective", "dealers", "offers", "expand", "profitability", "retailers", "territories", "quotes", "emphasize", "trade", "quote", "sales", "leads", "behalf", "markets", "purchased", "credit", "advertised", "complicated", "drops", "price", "features", "contracts"],
    "science": ["experiments", "samples", "laboratory", "lab", "analyzed", "findings", "data", "protocols", "pcr", "elisa", "assay", "pipette", "microscope", "cell culture", "specimen", "economics", "colleges", "theories", "fact", "universities", "historical", "interpreting", "phenomena", "scientific", "geographic", "manuscripts", "bioinformatics", "satellites", "fieldwork", "imagery", "interviewers", "science", "habitat", "applied", "species"],
    "skilled-trade": ["pistons", "ceiling", "paneling", "carburetors", "ignition", "tile", "insulated", "couplings", "hangers", "distortion", "scaffolding", "reassemble", "scaffolds", "freshly", "float", "shoring", "carburetor", "caulking", "cover", "plumbing", "ceilings", "ladders", "holes", "walls", "engine"],
    "social-service": ["dependencies", "overcoming", "crises", "counseling", "referral", "advocate", "options", "family", "client", "consultation", "education", "housing", "share", "informed", "community", "psychologists", "supporting", "mental", "goals", "dealing", "understanding", "engage", "refer", "making", "eligibility"],
    "support": ["resolved", "calls", "billing", "returns", "customer service", "helped", "assisted", "answered", "troubleshoot", "ticketing", "helpdesk", "zendesk", "freshdesk", "service desk", "causes", "shipping", "issue", "details", "departments", "complaints", "resolve", "arrange", "service", "orders", "changes", "customers", "recommend", "keep", "problems"],
    "technology": ["logical", "user", "engineering", "qualification", "prototype", "prototypes", "engineers", "network", "validation", "server", "users", "software", "configure", "manufacturing", "patent", "system", "technologies", "hardware", "analysts", "integration", "theoretical", "interoperability", "nano", "backup", "interface"],
    "unmapped": ["pipelines", "protractors", "climb", "rigging", "derricks", "shapes", "exteriors", "patch", "stretch", "sides", "furnace", "hammers", "solvents", "crane", "cranes", "hand", "cable", "imperfections", "poles", "signal", "alteration", "warn", "plot", "horizontal", "flows"],
}


_FUNC_BONUS_WORDS: dict[str, list[str]] = {
    "administrative": ["calendars", "ledgers", "invoices", "postage", "principal", "agendas", "minutes", "appointment", "memos", "mailing", "accounts", "mail", "bookkeeping", "payment", "documents", "account", "affix", "transactions", "file", "answer", "correspondence", "telephones", "proofread", "expedite", "inquiries"],
    "agriculture": ["seed", "seedlings", "agricultural", "harvesting", "livestock", "planting", "farmers", "logging", "lands", "thinning", "seeds", "insects", "forest", "farm", "forestry", "weed", "insect", "growers", "competing", "planted", "nursery", "reforestation", "grazing", "trees", "weeds"],
    "arts-media": ["logos", "branding", "illustrator", "photoshop", "indesign", "canva", "figma", "sketch", "wireframe", "broadcast", "cues", "dramatic", "stories", "shot", "rehearsal", "roles", "broadcasts", "edit", "live", "actors", "plays", "acting", "writers", "productions", "story", "producers", "scripts", "films", "news", "audition", "footage", "filming", "audience", "musical"],
    "building-grounds": ["carpets", "scrubbing", "squeegees", "driveways", "mow", "mops", "cleaners", "representations", "insecticides", "infestation", "mowers", "lawns", "trimmers", "sewers", "shovels", "conserve", "landscaping", "sidewalks", "scrub", "brooms", "basins", "mud", "drainage", "vacuuming", "ditches"],
    "design": ["decoration", "illustrations", "artwork", "ideas", "creations", "looks", "sketch", "characters", "success", "intended", "concepts", "render", "illustration", "animation", "script", "arrangement", "brochures", "designs", "sketches", "art", "shows", "venues", "backgrounds", "drawings", "three"],
    "education": ["internship", "advisers", "classroom", "class", "homework", "bibliographies", "campus", "moderate", "syllabi", "classrooms", "curricula", "teacher", "textbooks", "experiential", "handouts", "vocational", "faculty", "lessons", "supplement", "provides", "academic", "persevere", "challenging", "teaching", "lectures"],
    "finance": ["quickbooks", "payroll", "journal entries", "reconciled", "bookkeeping", "general ledger", "invoicing", "accountants", "securities", "income", "valuation", "investment", "owed", "influences", "traders", "informal", "capital", "audit", "debt", "liabilities", "investments", "returns", "quantify", "trading", "deductions", "assets", "stocks", "finance", "financial", "tax", "estate", "theory"],
    "food-service": ["kitchen", "utensils", "coffee", "cook", "drinks", "sandwiches", "kitchens", "salads", "foods", "desserts", "silverware", "menus", "cooking", "dishes", "cooks", "drink", "beverages", "breads", "garnish", "soups", "meats", "cooked", "served", "taste", "pastries"],
    "healthcare": ["immunizations", "caregivers", "respiratory", "wounds", "electrocardiograms", "patient", "ekgs", "intravenous", "patients", "therapy", "urine", "diagnosis", "nursing", "pain", "medication", "acute", "operative", "surgery", "treatment", "medical", "vital", "medicine", "interventions", "chronic", "physicians"],
    "hospitality": ["guests", "reservations", "checking in", "checked in", "check ins", "cleaned rooms", "amenities", "restocked", "opera", "guest", "check-in", "checkout", "reservation", "housekeeping", "front desk", "concierge"],
    "legal": ["westlaw", "deposition", "discovery", "litigation", "notary", "wills", "briefs", "statutes", "appeals", "facts", "trial", "witnesses", "disputes", "legislation", "search", "court", "law", "decisions", "file", "cases", "contracts", "gather", "legal", "documents", "clients", "public", "meet", "analyze", "data"],
    "logistics": ["loaded", "warehouse", "plugs", "tank", "weighing", "loads", "navigation", "contents", "gps", "delivered", "spark", "drivers", "unload", "capacities", "bulbs", "connecting", "safely", "tire", "railroad", "lower", "transport", "vehicles", "trucks", "global", "tires"],
    "manufacturing": ["spindles", "jammed", "handwheels", "coolants", "arbors", "tensions", "guides", "conveyors", "speeds", "sharpen", "extruded", "cams", "strokes", "feed", "workpieces", "tend", "stops", "turn", "machine", "holding", "setup", "dies", "hoppers", "thread", "start"],
    "marketing": ["strategy", "pricing", "forecasting", "analyzing", "buying", "satisfaction", "developers", "text", "demand", "promotion", "markets", "marketing", "communications", "promotional", "trade", "company", "trends", "sales", "integrate", "affecting", "commercial", "strategies", "advertising", "vendors", "surveys"],
    "ops": ["disciplinary", "hires", "budgets", "hire", "promotions", "employee", "worker", "recruit", "grievances", "employees", "investors", "harassment", "vacancies", "ranges", "experienced", "complaints", "staffing", "manage", "managers", "operational", "company", "timely", "oversee", "policies", "transfers"],
    "personal-care": ["caskets", "pallbearers", "beards", "razors", "scalp", "sanitize", "casket", "cremations", "funerals", "burial", "floral", "bereaved", "shampoo", "clippers", "lotions", "funeral", "nail", "caring", "shave", "scissors", "flowers", "liquid", "skin", "homes", "laundry"],
    "protective-service": ["tampering", "secured", "suspects", "police", "fingerprints", "apprehend", "patrol", "violators", "survivors", "disturbances", "inmates", "correctional", "inmate", "prohibited", "persons", "crimes", "guard", "suspicious", "officers", "crime", "search", "occurrences", "prisoners", "entrances", "valuables"],
    "sales": ["warranties", "reorder", "prospective", "dealers", "offers", "expand", "profitability", "retailers", "territories", "quotes", "emphasize", "trade", "quote", "sales", "leads", "behalf", "markets", "purchased", "credit", "advertised", "complicated", "drops", "price", "features", "contracts"],
    "science": ["experiments", "samples", "laboratory", "lab", "analyzed", "findings", "data", "protocols", "pcr", "elisa", "assay", "pipette", "microscope", "cell culture", "specimen", "economics", "colleges", "theories", "fact", "universities", "historical", "interpreting", "phenomena", "scientific", "geographic", "manuscripts", "bioinformatics", "satellites", "fieldwork", "imagery", "interviewers", "science", "habitat", "applied", "species"],
    "skilled-trade": ["pistons", "ceiling", "paneling", "carburetors", "ignition", "tile", "insulated", "couplings", "hangers", "distortion", "scaffolding", "reassemble", "scaffolds", "freshly", "float", "shoring", "carburetor", "caulking", "cover", "plumbing", "ceilings", "ladders", "holes", "walls", "engine"],
    "social-service": ["dependencies", "overcoming", "crises", "counseling", "referral", "advocate", "options", "family", "client", "consultation", "education", "housing", "share", "informed", "community", "psychologists", "supporting", "mental", "goals", "dealing", "understanding", "engage", "refer", "making", "eligibility"],
    "support": ["resolved", "calls", "billing", "returns", "customer service", "helped", "assisted", "answered", "troubleshoot", "ticketing", "helpdesk", "zendesk", "freshdesk", "service desk", "causes", "shipping", "issue", "details", "departments", "complaints", "resolve", "arrange", "service", "orders", "changes", "customers", "recommend", "keep", "problems"],
    "technology": ["logical", "user", "engineering", "qualification", "prototype", "prototypes", "engineers", "network", "validation", "server", "users", "software", "configure", "manufacturing", "patent", "system", "technologies", "hardware", "analysts", "integration", "theoretical", "interoperability", "nano", "backup", "interface"],
    "unmapped": ["pipelines", "protractors", "climb", "rigging", "derricks", "shapes", "exteriors", "patch", "stretch", "sides", "furnace", "hammers", "solvents", "crane", "cranes", "hand", "cable", "imperfections", "poles", "signal", "alteration", "warn", "plot", "horizontal", "flows"],
}


def _build_index() -> dict:
    """Build keyword→occupation index from O*NET Task Statements."""
    import pandas as pd

    onet = os.path.join(_HERE, "..", "data", "onet")
    tasks_df = pd.read_excel(os.path.join(onet, "Task Statements.xlsx"))

    occs = []
    keyword_df = Counter()
    keyword_occs: dict[str, set[int]] = {}

    for (soc, title), group in tasks_df.groupby(["O*NET-SOC Code", "Title"]):
        occ_idx = len(occs)
        occs.append({"title": title, "soc": soc})
        all_words = set()
        for t in group["Task"].dropna():
            all_words.update(
                w for w in re.findall(r"\b[a-z][a-z]+\b", t.lower())
                if w not in _STOPS
            )
        for w in all_words:
            keyword_df[w] += 1
            keyword_occs.setdefault(w, set()).add(occ_idx)

    return {
        "occs": occs,
        "num_occs": len(occs),
        "keyword_df": dict(keyword_df),
        "keyword_occs": {k: list(v) for k, v in keyword_occs.items()},
    }


def _load_index() -> dict:
    """Load or build the occupation index."""
    global _occ_index
    if _occ_index is None:
        if os.path.exists(_INDEX_PATH):
            with open(_INDEX_PATH, encoding="utf-8") as f:
                _occ_index = json.load(f)
        else:
            _occ_index = _build_index()
            with open(_INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump(_occ_index, f)
    return _occ_index


def _occ_to_function(title: str) -> str | None:
    """Map O*NET occupation title to function category."""
    # Check systems blacklist first
    for pattern, func in _FUNC_MAP:
        if re.search(pattern, title):
            if func == "technology" and pattern == r"\bSystems\b":
                if any(bad in title for bad in _SYSTEMS_BLACKLIST):
                    continue
            return func
    return None


def match_onet_occupations(
    text: str,
    top_k: int = 5,
) -> list[dict]:
    """Match resume text against 923 O*NET occupations via TF-IDF keyword overlap.

    Returns list of {title, soc, score} dicts.
    """
    index = _load_index()
    occs = index["occs"]
    num_occs = index["num_occs"]
    keyword_df = index["keyword_df"]
    keyword_occs = index["keyword_occs"]

    # Tokenize resume
    words = [w for w in re.findall(r"\b[a-z][a-z]+\b", text.lower()) if w not in _STOPS]
    wc = Counter(words)

    # Score each occupation by TF-IDF keyword overlap
    occ_scores = Counter()
    for w, count in wc.items():
        w_key = str(w)
        if w_key in keyword_occs and keyword_df[w_key] > 2:
            idf = math.log(num_occs / keyword_df[w_key])
            tf = 1 + math.log(count)
            weight = tf * idf
            for occ_idx in keyword_occs[w_key]:
                occ_scores[occ_idx] += weight

    # Apply function-specific bonus: distinctive words boost their function
    resume_words = set(wc.keys())
    resume_lower = text.lower()
    func_bonus = {}
    for func, bonus_words in _FUNC_BONUS_WORDS.items():
        # Match both single words and multi-word phrases
        matches = set()
        for bw in bonus_words:
            if ' ' in bw:
                if bw in resume_lower:
                    matches.add(bw)
            elif bw in resume_words:
                matches.add(bw)
        if matches:
            func_bonus[func] = 1 + 1.0 * len(matches)  # 100% boost per distinctive word
    
    if func_bonus:
        for occ_idx in list(occ_scores.keys()):
            func = _occ_to_function(occs[occ_idx]["title"])
            if func and func in func_bonus:
                occ_scores[occ_idx] *= func_bonus[func]

    # Apply function-specific bonus: distinctive words boost their function
    resume_words = set(wc.keys())
    resume_lower = text.lower()
    func_bonus = {}
    for func, bonus_words in _FUNC_BONUS_WORDS.items():
        matches = set()
        for bw in bonus_words:
            if " " in bw:
                if bw in resume_lower: matches.add(bw)
            elif bw in resume_words: matches.add(bw)
        if matches:
            func_bonus[func] = 1 + 1.0 * len(matches)
    if func_bonus:
        for occ_idx in list(occ_scores.keys()):
            func = _occ_to_function(occs[occ_idx]["title"])
            if func and func in func_bonus:
                occ_scores[occ_idx] *= func_bonus[func]

    # Rank and return top-k
    results = []
    for occ_idx, score in occ_scores.most_common(top_k):
        occ = occs[occ_idx]
        func = _occ_to_function(occ["title"])
        results.append({
            "title": occ["title"],
            "soc": occ["soc"],
            "score": round(score, 1),
            "function": func,
        })

    return results


def match_role_onet(text: str) -> dict | None:
    """Match resume text to best-fit function using O*NET occupation data.

    Returns dict compatible with the old match_role format:
    {function, match_pct, onet_title, alternatives, ...}
    """
    occs = match_onet_occupations(text, top_k=15)
    if not occs:
        return None

    # Score functions: best occupation + breadth bonus.
    # Breadth only matters as a tiebreaker when scores are close.
    func_best_score: dict[str, float] = {}
    func_best_title: dict[str, str] = {}
    func_breadth: dict[str, int] = {}
    for occ in occs:
        func = occ["function"]
        if func:
            if func not in func_best_score or occ["score"] > func_best_score[func]:
                func_best_score[func] = occ["score"]
                func_best_title[func] = occ["title"]
            # Count additional occupations with meaningful scores
            if occ["score"] > 8:
                func_breadth[func] = func_breadth.get(func, 0) + 1

    # Combine: best score + small breadth bonus (only for close calls)
    func_score = {}
    for func in func_best_score:
        base = func_best_score[func]
        breadth = func_breadth.get(func, 1)
        # Each additional occupation beyond the first adds 10% of its score
        func_score[func] = base * (1 + 0.1 * (breadth - 1))

    if not func_best_score:
        return None

    # Rank functions by score (best occupation + breadth bonus)
    ranked = sorted(func_score.items(), key=lambda x: -x[1])

    # Confidence: what fraction of total signal goes to each function?
    total_signal = sum(s for _, s in ranked)
    if total_signal == 0:
        total_signal = 1

    best_func, best_score_val = ranked[0]
    best_title = func_best_title[best_func]
    raw_pct = best_score_val / total_signal * 100
    match_pct = min(round(raw_pct), 95) if len(ranked) == 1 else round(raw_pct)

    alternatives = []
    for func, score_val in ranked[1:]:
        pct = round(score_val / total_signal * 100)
        if pct >= 5:
            alternatives.append({
                "function": func,
                "match_pct": pct,
                "onet_title": func_best_title[func],
            })

    return {
        "function": best_func,
        "level": "Entry",
        "match_pct": match_pct,
        "onet_title": best_title,
        "top_occupations": [
            {"title": o["title"], "score": o["score"], "function": o["function"]}
            for o in occs[:5]
        ],
        "alternatives": alternatives[:5],
    }
