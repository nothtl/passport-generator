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
        r"\bled (the |a |development |data )?(team|initiative)", r"\bcross.functional",
        r"\bworked (with|alongside)", r"\bpartnered with",
    ],
    "teamwork": [
        r"\bteamwork\b", r"\bteam player\b",
        r"\bled (the |a )?team", r"\bcollaborat",
        r"\bcross.functional team", r"\bworked with.*team",
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
# Stage 2: Section-aware + known-term extraction
# Reads the Technical Skills section directly, matches against a
# curated list of ~300 common technical tools/frameworks/platforms,
# and extracts role-based skills (CTO → leadership).
# Instant — no model, no embeddings.
# =========================================================================

# Known technical terms that indicate real skills when found in text
_KNOWN_TECH_TERMS: set[str] = {
    # AI / ML
    "agentic ai", "artificial intelligence", "machine learning", "deep learning",
    "computer vision", "nlp", "natural language processing", "neural network",
    "transformer", "llm", "large language model", "generative ai", "gen ai",
    "reinforcement learning", "data science", "predictive modeling",
    "yolo", "yolov5", "yolov8", "yolov11", "segformer", "resnet", "resnet-101",
    "imagenet", "object detection", "image segmentation", "image classification",
    "semantic segmentation", "instance segmentation", "ocr", "pose estimation",
    # Frameworks & Tools
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "opencv",
    "hugging face", "spacy", "nltk", "langchain", "llamaindex", "crewai",
    "autogen", "semantic kernel", "fastapi", "flask", "django", "streamlit",
    "gradio", "ros", "ros2", "zed", "zed 2i", "realsense",
    # Cloud & Infrastructure
    "aws", "azure", "gcp", "google cloud", "firebase", "vercel", "netlify",
    "azure functions", "cosmos db", "lambda", "s3", "ec2", "dynamodb",
    "kubernetes", "docker", "terraform", "ci/cd", "github actions", "jenkins",
    "cloud infrastructure", "microservices", "serverless",
    # Languages & Databases
    "python", "javascript", "typescript", "java", "go", "golang", "rust",
    "c++", "c#", "sql", "postgresql", "mongodb", "redis", "graphql",
    "rest api", "restful", "node.js", "nodejs", "react", "angular", "vue",
    "next.js", "nextjs", "tailwind", "html", "css",
    # Full Stack & Architecture
    "full stack", "full-stack", "frontend", "front-end", "backend", "back-end",
    "system design", "software architecture", "technical architecture",
    "api design", "database design", "distributed systems",
    # Data & Analytics
    "data analysis", "data analytics", "data engineering", "data pipeline",
    "etl", "power bi", "tableau", "looker", "snowflake", "databricks",
    "big data", "data warehouse", "data lake",
    # DevOps & MLOps
    "devops", "mlops", "continuous integration", "continuous deployment",
    "monitoring", "logging", "prometheus", "grafana", "elk stack",
    # Embedded & Hardware
    "embedded systems", "iot", "raspberry pi", "arduino", "fpga",
    "robotics", "autonomous systems", "autonomous vehicle",
    "uav", "drone", "slam", "lidar", "stereo camera", "imu",
    # Security
    "cybersecurity", "penetration testing", "owasp", "encryption",
    "authentication", "authorization", "oauth", "jwt",
    # Business & Management
    "agile", "scrum", "kanban", "jira", "confluence", "notion",
    "product management", "project management", "stakeholder management",
    "okr", "kpi", "roadmap", "sprint planning",
    # Design
    "figma", "sketch", "adobe xd", "ui/ux", "user research",
    "wireframing", "prototyping", "design system",
    # Other
    "git", "github", "gitlab", "bitbucket", "linux", "unix",
    "bash", "shell scripting", "regex", "api integration",
    "saas", "crm", "salesforce", "erp", "sap",
    "blockchain", "web3", "solidity", "smart contract",
}

# Job titles that imply specific skills
_ROLE_SKILL_MAP: dict[str, str] = {
    "cto": "technical leadership",
    "chief technology officer": "technical leadership",
    "vp of engineering": "technical leadership",
    "tech lead": "technical leadership",
    "technical architect": "software architecture",
    "solution architect": "software architecture",
    "co-founder": "entrepreneurship",
    "founder": "entrepreneurship",
    "ceo": "entrepreneurship",
    "chief executive": "entrepreneurship",
    "product manager": "product management",
    "engineering manager": "engineering management",
    "scrum master": "agile",
    "devops engineer": "devops",
    "ml engineer": "ai & machine learning",
    "data scientist": "data science",
    "data engineer": "data engineering",
    "research scientist": "research",
}

# Terms that indicate context/domain, NOT a person's skill
_CONTEXT_TERMS: set[str] = {
    "youth", "immigrant", "first-generation", "nonprofit",
    "medical insurance", "insurance firms", "claim", "claims",
    "fitness", "nutrition", "food", "ingredient",
    "student", "students", "children", "elderly",
    "patient", "patients", "customer", "customers",
    "restaurant", "retail", "hospitality",
}


def _extract_skills_section(text: str) -> str:
    """Extract the Technical Skills / Skills section from a resume."""
    # Common section headers
    patterns = [
        r'(?:Technical\s+)?Skills?\s*:?\s*(.+?)(?:\n\n|\n[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:\||-|–|:|\n))',
        r'(?:Technical\s+)?Skills?\s*:?\s*\n((?:\s*[•\-\*]\s*.+\n?)+)',
        r'(?:Technical\s+)?Skills?\s*:?\s*(.+?)(?:\n\s*\n|\Z)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1).strip()
            if len(section) > 10:
                return section
    return ""


def _split_terms(text: str) -> list[str]:
    """Split a comma/pipe/separator-delimited skills string into individual terms."""
    terms = []
    # Split on commas, pipes, bullets
    for chunk in re.split(r'[,;|\n•\-\*]', text):
        # Also split on "and" / "&" for compound lists
        sub_chunks = re.split(r'\s+(?:and|&)\s+', chunk.strip())
        for s in sub_chunks:
            s = s.strip().rstrip('.').lower()
            if len(s) > 2:
                terms.append(s)
    return terms


def extract_skills_section_aware(text: str) -> list[str]:
    """Stage 2: Section-aware extraction.

    Reads the Technical Skills section directly and matches against
    a curated list of known technical terms. Also extracts role-based
    skills from job titles (CTO, Co-Founder, etc.).
    Instant — no model, no embeddings, <5ms.
    """
    found: set[str] = set()
    lowered = text.lower()

    # 1. Extract from Technical Skills section
    skills_section = _extract_skills_section(text)
    if skills_section:
        for term in _split_terms(skills_section):
            if term in _KNOWN_TECH_TERMS:
                found.add(term)

    # 2. Also scan full text for known tech terms (catches terms in prose)
    #    but only if they appear in high-signal positions
    for term in _KNOWN_TECH_TERMS:
        if term in lowered:
            # Check if this term is used as a skill, not context
            # Look for it near action verbs or in tech context
            idx = lowered.find(term)
            surrounding = lowered[max(0, idx-50):idx+len(term)+50]
            # Accept if near technical signals
            tech_signals = ['built', 'developed', 'integrated', 'designed',
                          'implemented', 'deployed', 'engineered', 'coded',
                          'programmed', 'architected', 'led', 'managed']
            if any(sig in surrounding for sig in tech_signals):
                found.add(term)

    # 3. Extract role-based skills
    for line in text.split('\n'):
        line_lower = line.strip().lower()
        for role_title, skill in _ROLE_SKILL_MAP.items():
            if role_title in line_lower:
                found.add(skill)

    # 4. Map extracted terms to our skill categories
    mapped = _map_to_categories(found)
    return sorted(mapped | found)


def _map_to_categories(terms: set[str]) -> set[str]:
    """Map specific tech terms to our broader skill categories."""
    mapping = {
        "software & technical": {
            "python", "javascript", "typescript", "java", "go", "golang", "rust",
            "c++", "c#", "sql", "react", "angular", "vue", "node.js", "nodejs",
            "fastapi", "flask", "django", "rest api", "api design", "graphql",
            "docker", "kubernetes", "terraform", "ci/cd", "github actions",
            "full stack", "full-stack", "frontend", "front-end", "backend", "back-end",
            "microservices", "serverless", "git", "github", "linux",
            "software architecture", "technical architecture", "system design",
        },
        "ai & machine learning": {
            "agentic ai", "artificial intelligence", "machine learning", "deep learning",
            "computer vision", "nlp", "natural language processing", "neural network",
            "transformer", "llm", "large language model", "generative ai",
            "data science", "yolo", "yolov8", "yolov11", "segformer", "resnet",
            "tensorflow", "pytorch", "object detection", "image segmentation",
            "autonomous systems", "autonomous vehicle",
        },
        "data analysis": {
            "data analysis", "data analytics", "power bi", "tableau", "looker",
            "snowflake", "databricks", "data engineering", "data pipeline",
        },
        "cloud infrastructure": {
            "aws", "azure", "gcp", "google cloud", "azure functions", "cosmos db",
            "lambda", "s3", "ec2", "cloud infrastructure", "firebase",
        },
        "project management": {
            "agile", "scrum", "kanban", "jira", "sprint planning",
            "product management", "roadmap",
        },
        "leadership": {
            "technical leadership", "cto", "tech lead", "vp of engineering",
            "engineering manager",
        },
    }
    result: set[str] = set()
    for category, keywords in mapping.items():
        if terms & keywords:
            result.add(category)
    return result

def extract_skills_semantic(text: str) -> list[str]:
    """Stage 2: Section-aware + keyword overlap.

    Runs both approaches and merges:
    - Section-aware: reads Technical Skills section, matches ~300 tech terms
    - Quick phrase match: catches common non-tech skill phrases in prose
    Instant — no model, no embeddings, <10ms.
    """
    skills: set[str] = set()

    # 2a: Section-aware (tech skills, role titles)
    skills.update(extract_skills_section_aware(text))

    # 2b: Quick phrase match for non-tech context skills
    skills.update(_extract_phrase_skills(text))

    return sorted(skills)


# Quick phrase → skill mapping for non-tech contexts
_PHRASE_SKILL_MAP: dict[str, str] = {
    # Food service
    "food handler": "certifications",
    "servsafe": "certifications",
    "food safety": "certifications",
    "prepped": "food service",
    "prep cook": "food service",
    "grill station": "food service",
    "line cook": "food service",
    "sous chef": "food service",
    "pastry chef": "food service",
    "plated": "food service",
    "espresso": "food service",
    "barista": "food service",
    "restaurant": "food service",
    "kitchen": "food service",
    "dinner rush": "time management",
    "culinary": "food service",
    # Leadership / supervision
    "opened/closed store": "leadership",
    "opened and closed": "leadership",
    "supervised": "leadership",
    "shift supervisor": "leadership",
    "managed team": "leadership",
    "in charge of": "leadership",
    # Teaching / training
    "trained new": "teaching",
    "trained staff": "teaching",
    "onboarded": "teaching",
    # Finance
    "handled cash": "budgeting & finance",
    "cash register": "budgeting & finance",
    "processed payments": "budgeting & finance",
    "processed transactions": "budgeting & finance",
    # Inventory
    "placed weekly supply orders": "inventory management",
    "supply orders": "inventory management",
    "inventoried": "inventory management",
    "ordered supplies": "inventory management",
    "stocked": "inventory management",
    "managed inventory": "inventory management",
    # Trades
    "forklift": "trades & physical",
    "scissor lift": "trades & physical",
    "osha": "certifications",
    "blueprint": "trades & physical",
    "drywall": "trades & physical",
    "framed walls": "trades & physical",
    "poured concrete": "trades & physical",
    "construction": "trades & physical",
    "job site": "trades & physical",
    # Logistics
    "cdl": "logistics & driving",
    "cdl class": "certifications",
    "otr driver": "logistics & driving",
    "truck driver": "logistics & driving",
    "pre-trip": "logistics & driving",
    "pre-trip inspection": "logistics & driving",
    "eld logs": "logistics & driving",
    "delivery driver": "logistics & driving",
    "route planning": "logistics & driving",
    "vehicle maintenance": "logistics & driving",
    "clean driving record": "logistics & driving",
    "freight": "logistics & driving",
    "dispatch": "logistics & driving",
    "miles per week": "time management",
    "on-time delivery": "time management",
    "on time delivery": "time management",
    "shipping": "inventory management",
    "receiving": "inventory management",
    "warehouse": "inventory management",
    # Administrative
    "medical records": "data entry",
    "patient scheduling": "scheduling",
    "insurance verification": "administrative",
    "filing system": "administrative",
    "organized office": "administrative",
    "office supplies": "administrative",
    "data entry": "data entry",
    "answered phones": "customer service",
    "front desk": "customer service",
    "customer service": "customer service",
    "client service": "customer service",
    # Communication & Teamwork
    "coordinated with": "communication",
    "communicated": "communication",
    "team player": "communication",
    "led team": "communication",
    "led the team": "communication",
    "led a team": "communication",
    "managed team": "communication",
    "led development": "communication",
    "cross-functional": "communication",
    "collaborated": "communication",
    "collaboration": "communication",
    "worked with": "communication",
    "worked alongside": "communication",
    "partnered with": "communication",
}


def _extract_phrase_skills(text: str) -> set[str]:
    """Catch non-tech skills from common action phrases in prose."""
    found: set[str] = set()
    lowered = text.lower()
    for phrase, skill in _PHRASE_SKILL_MAP.items():
        if phrase in lowered:
            found.add(skill)
    return found


def extract_skills_from_text(text: str, use_semantic: bool = True) -> list[str]:
    """Extract named skills from resume/LinkedIn text.

    Stage 1: regex patterns (35 categories, fast, covers general skills)
    Stage 2: section-aware + known-term extraction (instant, catches tech/role skills)

    Set use_semantic=False for Stage 1 only.
    """
    if not text:
        return []

    # Stage 1: regex patterns for general skills
    lowered = text.lower()
    matched: set[str] = set()
    for skill, patterns in SKILL_PATTERNS.items():
        if any(re.search(pat, lowered) for pat in patterns):
            matched.add(skill)

    # Stage 2: section-aware (catches tech terms, role skills, filters context)
    if use_semantic:
        try:
            semantic_skills = extract_skills_semantic(text)
            matched.update(semantic_skills)

            # Filter false positives: when strong tech signals exist,
            # suppress context-based Stage 1 matches
            strong_tech = {'ai & machine learning', 'software & technical',
                          'software architecture', 'technical leadership',
                          'cloud infrastructure', 'computer vision'}
            if matched & strong_tech:
                # These are likely context words, not the person's skills
                context_prone = {'healthcare', 'youth engagement', 'community outreach',
                                'social media management', 'photography', 'logistics & driving'}
                # Only remove if they appear ONLY via Stage 1 (not confirmed by Stage 2)
                for skill in context_prone:
                    if skill in matched and skill not in semantic_skills:
                        matched.discard(skill)
        except Exception:
            pass

    return sorted(matched)


def extract_skills_from_bundle(bundle: StudentBundle) -> list[str]:
    """Extract skills from a StudentBundle (passport_agent_v2 format)."""
    resume_text = getattr(bundle, "resume_text", "") or ""
    linkedin_text = getattr(bundle, "linkedin_text", "") or ""
    combined = f"{resume_text}\n{linkedin_text}"
    return extract_skills_from_text(combined)
