from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passport_agent_v2.tools.ingest import StudentBundle

# =========================================================================
# Skill patterns: each skill has multiple regex patterns.
# Expanded from 19 → 45+ skills. Each pattern matches how people
# actually write on resumes, not how the corpus labels skills.
# =========================================================================

SKILL_PATTERNS: dict[str, list[str]] = {
    # ── Creative & Design ──
    "graphic design": [
        r"\bgraphic design", r"\bcanva\b", r"\badobe creative",
        r"\bphotoshop\b", r"\billustrator\b", r"\bvisual design",
        r"\bindesign\b", r"\bfigma\b",
    ],
    "photography": [
        r"\bphotograph", r"\bvisual artist", r"\bphotoshop",
        r"\bphoto\s?(shoot|edit)", r"\blightroom\b", r"\bdslr\b",
    ],
    "content creation": [
        r"\bcontent creat", r"\bwrote (and|&) creat", r"\bblog post",
        r"\bnewsletter", r"\bcontent (writing|development|strategy)",
    ],
    "social media management": [
        r"\bsocial media", r"\binstagram\b", r"\btiktok\b", r"\btwitter\b",
        r"\blinkedin", r"\bfacebook\b", r"\bsocial platform",
        r"\bsocial media (account|page|presence|content|marketing|campaign)",
    ],
    "video editing": [
        r"\bvideo edit", r"\bvideograph", r"\bpremiere pro\b",
        r"\bfinal cut\b", r"\bdavinci\b", r"\bcapcut\b", r"\bafter effects\b",
    ],

    # ── Writing & Communication ──
    "writing": [
        r"\bwrite\b", r"\bwrote\b", r"\bwritten", r"\bauthor\b",
        r"\bcopywriting", r"\bcopywrite", r"\bblog\b", r"\bjournalis",
    ],
    "public speaking": [
        r"\bpublic speak", r"\bpresentation\b", r"\bpresented\b",
        r"\bpublic speaking", r"\btoastmaster", r"\bkeynote\b",
        r"\bgave (a |the )?(talk|presentation|speech)",
    ],

    # ── Teaching & Mentoring ──
    "teaching": [
        r"\bteach", r"\binstruct", r"\bdigital historian",
        r"\btrain(ing|ed)?\b(?!.*(?:student))", r"\btutor",
        r"\b(coach|coaching)\b", r"\blesson plan", r"\bcurriculum",
        r"\bworkshop (facilitat|deliver|led|lead|ran|run)",
        r"\b(led|ran|run|facilitat).*workshop",
    ],
    "mentoring": [
        r"\bmentor", r"\bpeer group", r"\btutor",
        r"\b(coach|coaching).*(student|peer|youth|junior)",
    ],
    "youth engagement": [
        r"\byouth", r"\bteen", r"\bhigh school student",
        r"\bwork with.*(student|youth|teen)", r"\bafter.school",
        r"\bsummer camp", r"\bcamp counselor",
    ],

    # ── Community & Outreach ──
    "community outreach": [
        r"\bcommunity outrea", r"\boutreach", r"\bpantry\b",
        r"\bfood (distri|drive|bank)", r"\bpark clean", r"\bcommunity (service|event|program)",
    ],
    "volunteer coordination": [
        r"\bvolunteer coord", r"\brecruit.*volunteer",
        r"\bvolunteer.*manage", r"\b(coordinate|organize|manage).*volunteer",
    ],
    "fundraising": [
        r"\bfundrais", r"\bdonor\b", r"\bgrant writ",
        r"\b(grant|proposal) writing", r"\bdevelopment associate",
    ],

    # ── Program & Project Management ──
    "program management": [
        r"\bprogram manag", r"\bcoordinate.*program", r"\bmanaged.*program",
        r"\bprogram coordinat", r"\bprogram director",
    ],
    "project management": [
        r"\bproject manag", r"\blead.*project", r"\bmanaged.*project",
        r"\bproject coordinat", r"\bproject lead",
    ],
    "event planning": [
        r"\bevent plan", r"\bevent coord", r"\borganiz.*event",
        r"\bflyers", r"\bevent (host|execution|logistics)",
        r"\b(hosted|organized|coordinated|planned|ran).*event",
    ],
    "leadership": [
        r"\bleadership", r"\bco-chair\b", r"\bexecutive board",
        r"\b(president|treasurer|secretary|vice president|vp)\b",
        r"\b(board member|board of directors)", r"\bteam lead",
    ],

    # ── Data & Technical ──
    "data entry": [
        r"\bdata entry\b", r"\bdatabase\b", r"\bgoogle maps\b",
        r"\bexcel\b", r"\bspreadsheet", r"\bdata (input|collection|management)",
        r"\brecord.?keeping", r"\bdata.*(entry|enter)",
    ],
    "data analysis": [
        r"\bdata analy", r"\banalytics\b", r"\bvisualization",
        r"\bpower\s?bi\b", r"\btableau\b", r"\blooker\b",
        r"\b(pivot table|vlookup|sql|python|r\b).*data",
    ],
    "software & technical": [
        r"\bsoftware (development|engineering|developer)\b",
        r"\b(website|app|application|platform) (built|created|developed|designed|launched)",
        r"\b(front.?end|back.?end|full.?stack)\b",
        r"\b(javascript|python|java|react|node|html|css|sql|api)\b",
    ],

    # ── Healthcare ──
    "healthcare": [
        r"\b(healthcare|health care|health aide|health science)\b",
        r"\bpatient\b", r"\bclinical\b", r"\bmedical\b",
        r"\bhospital\b", r"\bhome health", r"\bvaccination", r"\bvaccinated",
        r"\b(EMR|EHR|electronic (medical|health) record)\b",
        r"\bepic\b(?!.*games)", r"\bcerner\b", r"\bHIPAA\b",
        r"\b(phlebotom|venipuncture|blood draw)", r"\bvital sign",
        r"\bpatient care", r"\bCNA\b", r"\bRN\b", r"\bLPN\b",
        r"\bmedical assistant", r"\bnursing\b", r"\bhome health aide",
        r"\bcertified nursing", r"\bclinical assistant",
        r"\bcase manag", r"\bflu (pod|clinic|vaccination|immunization)",
        r"\bvaccine\b", r"\bimmunization",
    ],

    # ── Customer Service & Sales ──
    "customer service": [
        r"\bcustomer service", r"\bclient.*service", r"\bfront desk\b",
        r"\breception", r"\b(assist|help|serve).*(customer|client|guest|patient)",
        r"\b(customer|client|guest|patient).*(assist|help|serve|support)",
        r"\bpoint.of.sale\b", r"\bPOS system\b", r"\bcash register",
        r"\b(retail|store|shop|boutique).*associate",
    ],
    "sales": [
        r"\bsales associate", r"\bsales representative", r"\bsales rep\b",
        r"\bsales floor", r"\b(retail|inside|outside|b2b|b2c) sales\b",
        r"\b(upsell|cross.sell|revenue|quota|commission)\b",
        r"\b(credit card|membership|loyalty).*(sign.?up|enroll)",
    ],

    # ── Operations & Logistics ──
    "inventory management": [
        r"\binventory", r"\bstock(ing|ed|room)?\b", r"\bsupply chain",
        r"\bwarehouse\b", r"\b(stock|inventory).*(manage|control|track|audit|organize)",
    ],
    "scheduling": [
        r"\bschedul", r"\bappointment", r"\bcalendar management",
        r"\b(book|coordinate|manage).*(schedule|appointment)",
    ],

    # ── Finance ──
    "budgeting & finance": [
        r"\bbudget", r"\baccounting\b", r"\bbookkeeping",
        r"\b(manage|track|oversee).*(budget|finance|money|fund)",
        r"\b(handle|process).*(payment|transaction|cash)",
        r"\bquickbooks\b", r"\baccounts? (payable|receivable)\b",
    ],

    # ── Marketing ──
    "marketing": [
        r"\bmarket(ing)?\b(?!.*(?:supermarket|farmers market))",
        r"\b(brand|branding|brand management)\b",
        r"\b(SEO|SEM|PPC|email marketing|digital marketing)\b",
        r"\b(campaign|ad campaign|marketing campaign)",
    ],

    # ── Translation & Languages ──
    "languages": [
        r"\bbilingual\b", r"\btranslat(e|or|ion)\b", r"\binterpret(er|ation)?\b",
        r"\bspanish\b", r"\bfrench\b", r"\barabic\b", r"\bmandarin\b",
        r"\bchinese\b", r"\bcantonese\b", r"\bportuguese\b",
        r"\b(bengali|urdu|vietnamese|hindi|wolof|korean|japanese|russian)\b",
    ],

    # ── Soft/Transferable Skills ──
    "communication": [
        r"\bcommunicat", r"\b(collaborat|teamwork|interpersonal)\b",
        r"\b(verbal|written|oral) communication", r"\bpeople skills\b",
    ],
    "problem solving": [
        r"\bproblem.solv", r"\bcritical think", r"\btroubleshoot",
        r"\broot cause", r"\b(resolve|resolved|resolution).*(issue|problem|conflict)",
    ],
    "time management": [
        r"\btime manag", r"\bprioriti(z|s)ation", r"\bdeadline",
        r"\bmulti.?task", r"\b(juggle|balance).*(multiple|many|several)",
    ],

    # ── Certifications ──
    "certifications": [
        r"\b(CPR|BLS|AED|first aid) (certif|cert|training)",
        r"\b(certified|certification).*(nursing|medical|cna|phlebotom|emt)",
        r"\b(OSHA|ServSafe|food handler|driver.?s? license|CDL)",
    ],

    # ── Food Service & Hospitality ──
    "food service": [
        r"\b(line cook|cook|chef|sous chef|pastry chef|prep cook)\b",
        r"\b(kitchen|grill|fry|saute|bake|baking|culinary)\b",
        r"\b(server|waitstaff|waiter|waitress|host|hostess|busser|busboy)\b",
        r"\b(barista|bartender|bar back|bar management)\b",
        r"\b(restaurant|cafe|cafeteria|diner|fast food|fine dining)\b",
        r"\b(food prep|food preparation|food service|food handling)\b",
        r"\bmenu\b", r"\bcatering\b", r"\bplating\b", r"\bfood (cost|safety|quality)",
    ],

    # ── Manufacturing & Production ──
    "manufacturing": [
        r"\b(assembly|production) (line|worker|operator|associate)\b",
        r"\b(machine operator|CNC|fabrication|machining)\b",
        r"\b(quality control|quality assurance|QA|inspector)\b",
        r"\b(manufacturing|production) (plant|facility|floor|environment)\b",
        r"\b(packaging|packer|picker|material handler)\b",
        r"\b(lean manufacturing|six sigma|5S|kaizen|continuous improvement)\b",
    ],

    # ── Administrative & Clerical ──
    "administrative": [
        r"\b(admin|administrative|executive) assistant\b",
        r"\b(office|clerk|clerical|secretary|receptionist)\b",
        r"\b(filing|typing|data entry|document (management|preparation|processing))\b",
        r"\b(calendar management|expense report|travel arrang|meeting (coordination|scheduling))\b",
        r"\b(phone|telephone).*(answering|handling|operator)",
        r"\b(office (management|manager|administration|support))\b",
    ],

    # ── Childcare ──
    "childcare": [
        r"\b(nanny|babysitter|daycare|child care|childcare)\b",
        r"\b(early childhood|preschool|toddler|infant care)\b",
        r"\b(au pair|playgroup|children.*program)\b",
    ],

    # ── Physical / Trade ──
    "trades & physical": [
        r"\b(construction|carpentry|plumbing|electrical|HVAC|welding|masonry)\b",
        r"\b(drywall|roofing|painting|concrete|framing|flooring|insulation)\b",
        r"\b(forklift|machinery|equipment).*(operator|operation|operating)",
        r"\bheavy (equipment|lifting|machinery)", r"\bOSHA\b",
        r"\b(mechanic|auto repair|diesel|appliance repair|millwright|machinist)\b",
        r"\b(power tool|hand tool|blueprint|job site|work site)\b",
    ],

    # ── Logistics & Driving ──
    "logistics & driving": [
        r"\blogistics\b", r"\bdelivery driver\b", r"\bshipping",
        r"\b(route|dispatch|fleet) (plan|manage|coordinate|driver)",
        r"\bCDL\b", r"\bdriving\b", r"\bdriver.?s? license\b",
        r"\b(courier|freight|truck|OTR|over.the.road|short.haul|long.haul)\b",
        r"\b(warehouse|distribution|fulfillment) (worker|associate|center)\b",
    ],
}

# Term sets for additional context detection
CREATIVE_TERMS = {"graphic", "design", "content", "poetry", "music", "choir", "story", "stories", "flyers", "photography", "photoshop", "visual", "artist"}
SOFTWARE_TERMS = {"website", "app", "application", "database", "software", "platform", "excel", "google maps", "canva", "kahoot", "padlet", "piktochart"}
COMMUNITY_TERMS = {"community", "nonprofit", "volunteer", "church", "pantry", "outreach", "cleanup", "food distribution", "food drive"}


# =========================================================================
# Stage 2: Keyword-overlap skill matching
# For each unmatched resume span, checks word overlap against expanded
# skill keyword lists. Fast (no model load), catches synonyms and
# context clues that Stage 1 regex misses.
# =========================================================================

# Expanded keyword lists for each skill — includes synonyms,
# related tools/software, and context phrases
_SKILL_KEYWORDS: dict[str, set[str]] = {
    "healthcare": {
        "patient", "patients", "clinical", "clinic", "hospital", "medical",
        "nursing", "nurse", "vaccination", "vaccinated", "vaccine", "immunization",
        "flu pod", "flu clinic", "health", "home health", "health aide",
        "emr", "ehr", "epic", "cerner", "vmware", "hipaa",
        "medical record", "medical records", "case management", "case manager",
        "phlebotomy", "phlebotomist", "vital sign", "vital signs",
        "cna", "rn", "lpn", "medical assistant", "patient care",
        "blood draw", "venipuncture", "specimen", "diagnosis",
        "home care", "caregiver", "hospice", "rehabilitation",
    },
    "teaching": {
        "train", "training", "trained", "teach", "teaching", "teacher",
        "instruct", "instructor", "tutor", "tutoring", "coach", "coaching",
        "lesson plan", "lesson planning", "curriculum", "workshop",
        "facilitation", "facilitator", "education", "educational",
        "student", "students", "classroom", "course", "courses",
    },
    "program management": {
        "program manager", "program management", "program coordinator",
        "program coordination", "managed program", "coordinated program",
        "oversaw program", "ran program", "pod management", "clinic management",
        "operations management", "initiative lead", "program lead",
        "flu pod", "flu clinic", "vaccination clinic", "immunization clinic",
        "health fair", "screening event", "health screening",
        "managed the pod", "managed a clinic", "coordinated clinic",
        "ran the clinic", "oversaw clinic",
    },
    "customer service": {
        "customer service", "client service", "patient service",
        "front desk", "reception", "receptionist",
        "assisting patients", "assisting clients", "assisting customers",
        "patient intake", "patient registration", "customer support",
        "client interaction", "patient interaction", "guest service",
    },
    "data entry": {
        "data entry", "database", "databases", "spreadsheet", "spreadsheets",
        "excel", "google sheets", "record keeping", "records management",
        "data collection", "data input", "data management",
        "electronic record", "electronic records", "documentation",
        "filing", "clerical", "administrative",
        "epic emr", "epic ehr", "emr system", "ehr system",
        "electronic medical record", "electronic health record",
        "vmware", "cerner", "meditech",
    },
    "leadership": {
        "leadership", "team lead", "team leader", "supervisor", "supervised",
        "manager", "managed team", "president", "treasurer", "secretary",
        "executive board", "board member", "director", "vice president",
        "co-chair", "chair", "head of", "led team", "leading",
    },
    "communication": {
        "communication", "communicated", "collaboration", "collaborated",
        "teamwork", "team player", "interpersonal", "people skills",
        "verbal communication", "written communication",
        "presented", "liaised", "coordinated with", "partnered",
    },
    "event planning": {
        "event", "events", "organizing event", "event planning",
        "event coordination", "hosted event", "event logistics",
        "flyers", "fundraiser", "fundraising event", "community event",
        "planning event", "coordinated event", "organized event",
    },
    "sales": {
        "sales", "selling", "upsell", "upselling", "cross-sell",
        "revenue", "quota", "commission", "credit card signup",
        "credit card sign-ups", "membership enrollment",
        "retail associate", "sales associate", "sales representative",
        "cash register", "pos", "point of sale", "store associate",
    },
    "marketing": {
        "marketing", "brand", "branding", "seo", "sem", "ppc",
        "digital marketing", "social media marketing", "email marketing",
        "campaign", "ad campaign", "content marketing",
        "market research", "google analytics",
    },
    "social media management": {
        "social media", "instagram", "tiktok", "facebook", "twitter",
        "linkedin", "social platform", "social account",
        "social media strategy", "social media content",
        "grew following", "followers", "engagement",
    },
    "content creation": {
        "content creation", "content creator", "content writing",
        "blog", "blogging", "newsletter", "newsletters",
        "copywriting", "copywriter", "wrote content",
    },
    "writing": {
        "writing", "writer", "wrote", "written", "author", "authored",
        "journalism", "journalist", "blogging", "blog posts",
        "copywriting", "copy", "editor", "editing",
    },
    "project management": {
        "project management", "project manager", "project coordination",
        "project lead", "managed project", "led project",
        "project planning", "project execution", "timeline",
        "deliverables", "stakeholder", "agile", "scrum",
    },
    "mentoring": {
        "mentor", "mentored", "mentoring", "mentorship",
        "peer advisor", "peer counselor", "big brother", "big sister",
        "coaching", "coach", "guided", "guidance",
    },
    "volunteer coordination": {
        "volunteer coordination", "volunteer coordinator",
        "managed volunteer", "coordinated volunteer",
        "recruit volunteer", "volunteer management",
        "volunteer scheduling", "volunteer recruitment",
    },
    "community outreach": {
        "community outreach", "outreach", "food drive", "food pantry",
        "community service", "community engagement", "park clean",
        "neighborhood", "community event", "community program",
    },
    "data analysis": {
        "data analysis", "data analyst", "analytics", "analyzed data",
        "visualization", "power bi", "tableau", "looker",
        "sql", "python", "r language", "statistical",
        "data science", "metrics", "dashboard",
    },
    "scheduling": {
        "scheduling", "schedule", "scheduled", "appointment",
        "appointments", "calendar", "booking", "coordinated schedule",
        "shift planning", "roster",
    },
    "budgeting & finance": {
        "budget", "budgeting", "budgeted", "accounting", "bookkeeping",
        "quickbooks", "accounts payable", "accounts receivable",
        "financial", "finance", "processed payment", "processed transactions",
        "cash handling", "cash management", "reconciliation",
    },
    "graphic design": {
        "graphic design", "graphic designer", "visual design",
        "canva", "photoshop", "illustrator", "indesign", "figma",
        "adobe creative", "adobe suite", "design software",
        "typography", "layout", "branding", "logo design",
    },
    "photography": {
        "photography", "photographer", "photo shoot", "photo editing",
        "lightroom", "dslr", "mirrorless", "camera",
        "visual content", "photo", "photos", "portrait",
    },
    "video editing": {
        "video editing", "video editor", "videography", "videographer",
        "premiere pro", "final cut", "davinci", "capcut",
        "after effects", "video production", "motion graphics",
    },
    "public speaking": {
        "public speaking", "public speaker", "presentation", "presentations",
        "presented", "keynote", "toastmasters", "speech", "speeches",
        "gave talk", "gave a talk", "public address",
    },
    "problem solving": {
        "problem solving", "problem solver", "troubleshoot", "troubleshooting",
        "critical thinking", "root cause", "resolved issue", "resolved issues",
        "conflict resolution", "analytical", "debugging",
    },
    "time management": {
        "time management", "multitask", "multitasking", "multi-tasking",
        "prioritize", "prioritization", "deadline", "deadlines",
        "juggling", "balancing multiple", "organized", "organization",
    },
    "languages": {
        "bilingual", "spanish", "french", "arabic", "mandarin",
        "chinese", "cantonese", "portuguese", "bengali", "urdu",
        "vietnamese", "hindi", "wolof", "korean", "japanese", "russian",
        "translate", "translation", "translator", "interpret", "interpreter",
    },
    "fundraising": {
        "fundraising", "fundraiser", "donor", "donation", "donations",
        "grant writing", "grant writer", "development associate",
        "capital campaign", "sponsorship", "philanthropy",
    },
    "inventory management": {
        "inventory", "stocking", "stock room", "stock management",
        "supply chain", "warehouse", "inventory control",
        "merchandise", "stock audit", "replenishment",
    },
    "logistics & driving": {
        "logistics", "delivery", "driver", "driving", "shipping",
        "dispatch", "dispatcher", "route planning", "fleet",
        "cdl", "truck", "transportation", "supply chain",
    },
    "certifications": {
        "cpr", "bls", "aed", "first aid", "certified", "certification",
        "osha", "servsafe", "food handler", "driver license",
        "drivers license", "cdl", "emt", "paramedic",
    },
    "youth engagement": {
        "youth", "teen", "teenager", "high school student",
        "after school", "summer camp", "camp counselor",
        "youth program", "youth group", "young people",
    },
    "trades & physical": {
        "construction", "carpentry", "plumbing", "plumber",
        "electrical", "electrician", "hvac", "welding", "welder",
        "masonry", "forklift", "heavy equipment", "machinery",
        "osha", "blueprint", "job site",
    },
    "software & technical": {
        "software", "developer", "development", "coding", "programming",
        "javascript", "python", "java", "react", "node", "nodejs",
        "html", "css", "api", "app development", "web development",
        "full stack", "front end", "back end", "database", "sql",
        "git", "github", "cloud", "aws", "azure",
    },
}

# Build keyword index: word → set of skills
_KEYWORD_INDEX: dict[str, set[str]] = {}
for _skill, _keywords in _SKILL_KEYWORDS.items():
    for _kw in _keywords:
        _KEYWORD_INDEX.setdefault(_kw.lower(), set()).add(_skill)

_STAGE2_MIN_OVERLAP = 1  # minimum keyword hits to match a skill


def _split_into_spans(text: str) -> list[str]:
    """Split resume text into meaningful spans for matching."""
    spans = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or len(para) < 20:
            continue
        spans.append(para)
    return spans


def _stage1_matched_spans(text: str) -> set[str]:
    """Return the set of lowercase spans that Stage 1 regex already matched."""
    matched_lines: set[str] = set()
    for line in text.splitlines():
        line_lower = line.strip().lower()
        if len(line_lower) < 20:
            continue
        for patterns in SKILL_PATTERNS.values():
            if any(re.search(pat, line_lower) for pat in patterns):
                matched_lines.add(line_lower)
                break
    return matched_lines


def extract_skills_semantic(text: str) -> list[str]:
    """Stage 2: keyword-overlap skill extraction for spans regex missed.

    Splits resume into paragraphs, tokenizes each, and checks word overlap
    against expanded skill keyword lists. Runs on ALL spans, not just
    unmatched ones — because a span may trigger one skill via regex
    but still contain other skills via keywords.

    Fast — no model load, no embeddings, <10ms.
    """
    if not text:
        return []

    all_spans = _split_into_spans(text)
    if not all_spans:
        return []

    # For each span, count keyword hits per skill
    skill_hits: dict[str, int] = {}
    for span in all_spans:
        span_lower = span.lower()
        span_skills: dict[str, int] = {}
        # Tokenize span into words, bigrams, and trigrams
        # Strip punctuation from tokens
        raw_words = span_lower.split()
        span_words = [w.strip('.,;:!?()[]{}""''-') for w in raw_words]
        span_words = [w for w in span_words if len(w) >= 2]
        all_tokens = set(span_words)
        for i in range(len(span_words) - 1):
            all_tokens.add(f"{span_words[i]} {span_words[i+1]}")
        for i in range(len(span_words) - 2):
            all_tokens.add(f"{span_words[i]} {span_words[i+1]} {span_words[i+2]}")

        for token in all_tokens:
            if token in _KEYWORD_INDEX:
                for skill in _KEYWORD_INDEX[token]:
                    span_skills[skill] = span_skills.get(skill, 0) + 1

        # Accumulate: at least _STAGE2_MIN_OVERLAP hits in this span
        for skill, hits in span_skills.items():
            if hits >= _STAGE2_MIN_OVERLAP:
                skill_hits[skill] = skill_hits.get(skill, 0) + 1

    return sorted(skill_hits.keys())


def extract_skills_from_text(text: str, use_semantic: bool = True) -> list[str]:
    """Extract named skills from resume/LinkedIn text.

    Stage 1: regex patterns (fast, covers ~60-70% of skills)
    Stage 2: embedding matching (slower, catches synonyms and context, ~20-30% more)

    Set use_semantic=False for Stage 1 only (faster, offline).
    """
    if not text:
        return []

    # Stage 1: regex
    lowered = text.lower()
    matched: set[str] = set()
    for skill, patterns in SKILL_PATTERNS.items():
        if any(re.search(pat, lowered) for pat in patterns):
            matched.add(skill)

    # Stage 2: semantic (optional, catches what regex missed)
    if use_semantic:
        try:
            semantic_skills = extract_skills_semantic(text)
            matched.update(semantic_skills)
        except Exception:
            # Stage 2 fails gracefully — embeddings not available, etc.
            pass

    return sorted(matched)


def extract_skills_from_bundle(bundle: StudentBundle) -> list[str]:
    """Extract skills from a StudentBundle (passport_agent_v2 format)."""
    resume_text = getattr(bundle, "resume_text", "") or ""
    linkedin_text = getattr(bundle, "linkedin_text", "") or ""
    combined = f"{resume_text}\n{linkedin_text}"
    return extract_skills_from_text(combined)
