"""College major recommendation - maps skills and interests to college majors.

Phase 3c: Uses CIP (Classification of Instructional Programs) codes and
O*NET education data to recommend college majors aligned with student profile.
"""

from __future__ import annotations

from recommender.config import get_college

_CFG = get_college()

# CIP code to major name mapping (most common 2-digit codes)
_CIP_MAJORS: dict[str, dict] = {
    "11": {
        "name": "Computer and Information Sciences",
        "majors": ["Computer Science", "Information Technology", "Software Engineering", "Data Science", "Cybersecurity"],
        "careers": ["Software Developer", "Data Analyst", "IT Specialist", "Systems Administrator"],
    },
    "14": {
        "name": "Engineering",
        "majors": ["Mechanical Engineering", "Electrical Engineering", "Civil Engineering", "Chemical Engineering", "Biomedical Engineering"],
        "careers": ["Engineer", "Project Manager", "Research Scientist", "Technical Consultant"],
    },
    "26": {
        "name": "Biological and Biomedical Sciences",
        "majors": ["Biology", "Biochemistry", "Neuroscience", "Microbiology", "Genetics"],
        "careers": ["Research Scientist", "Lab Technician", "Healthcare Professional", "Pharmaceutical Researcher"],
    },
    "27": {
        "name": "Mathematics and Statistics",
        "majors": ["Mathematics", "Statistics", "Applied Mathematics", "Data Science", "Actuarial Science"],
        "careers": ["Data Scientist", "Actuary", "Quantitative Analyst", "Statistician"],
    },
    "51": {
        "name": "Health Professions",
        "majors": ["Nursing", "Pre-Medicine", "Public Health", "Health Administration", "Physical Therapy"],
        "careers": ["Nurse", "Physician", "Healthcare Administrator", "Physical Therapist"],
    },
    "13": {
        "name": "Education",
        "majors": ["Elementary Education", "Secondary Education", "Special Education", "Education Policy", "Curriculum Design"],
        "careers": ["Teacher", "School Administrator", "Curriculum Designer", "Education Consultant"],
    },
    "44": {
        "name": "Public Administration and Social Service",
        "majors": ["Social Work", "Public Policy", "Human Services", "Criminal Justice", "Nonprofit Management"],
        "careers": ["Social Worker", "Policy Analyst", "Nonprofit Manager", "Probation Officer"],
    },
    "50": {
        "name": "Visual and Performing Arts",
        "majors": ["Graphic Design", "Fine Arts", "Film/Video Production", "Music", "Theater"],
        "careers": ["Graphic Designer", "Art Director", "Filmmaker", "Musician"],
    },
    "52": {
        "name": "Business, Management, and Marketing",
        "majors": ["Business Administration", "Finance", "Marketing", "Accounting", "Entrepreneurship"],
        "careers": ["Business Analyst", "Financial Analyst", "Marketing Manager", "Accountant"],
    },
    "09": {
        "name": "Communication and Journalism",
        "majors": ["Communications", "Journalism", "Public Relations", "Media Studies", "Digital Media"],
        "careers": ["Communications Specialist", "Journalist", "PR Manager", "Content Strategist"],
    },
    "45": {
        "name": "Social Sciences",
        "majors": ["Psychology", "Sociology", "Political Science", "Anthropology", "Economics"],
        "careers": ["Researcher", "Policy Analyst", "Counselor", "Market Researcher"],
    },
    "42": {
        "name": "Psychology",
        "majors": ["Psychology", "Clinical Psychology", "Counseling Psychology", "Industrial-Organizational Psychology", "Developmental Psychology"],
        "careers": ["Psychologist", "Counselor", "HR Specialist", "Research Assistant"],
    },
    "03": {
        "name": "Natural Resources and Conservation",
        "majors": ["Environmental Science", "Conservation Biology", "Natural Resource Management", "Environmental Policy"],
        "careers": ["Environmental Scientist", "Conservation Officer", "Sustainability Consultant"],
    },
    "04": {
        "name": "Architecture and Related Services",
        "majors": ["Architecture", "Urban Planning", "Interior Design", "Landscape Architecture"],
        "careers": ["Architect", "Urban Planner", "Interior Designer"],
    },
}

# Function to recommended CIP codes
_FUNC_TO_CIP: dict[str, list[str]] = {
    "technology": ["11", "14", "27"],
    "healthcare": ["51", "26", "42"],
    "education": ["13", "45", "42"],
    "finance": ["52", "27"],
    "design": ["50", "04", "11"],
    "arts-media": ["50", "09"],
    "social-service": ["44", "42", "45"],
    "engineering": ["14", "11", "27"],
    "science": ["26", "27", "03"],
    "legal": ["44", "45"],
    "business": ["52", "45"],
    "sales": ["52", "09"],
    "marketing": ["52", "09", "50"],
}


def recommend_majors(
    function: str,
    skills: list[str],
    ideal_careers: list[str] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Recommend college majors based on career function and skills.

    Args:
        function: Primary career function (technology, healthcare, etc.)
        skills: List of extracted student skills
        ideal_careers: Optional list of career aspirations
        top_n: Max number of recommendations

    Returns:
        List of major recommendations with match scores
    """
    cip_codes = _FUNC_TO_CIP.get(function, ["52", "13", "45"])

    results = []
    seen_majors = set()

    for cip in cip_codes:
        info = _CIP_MAJORS.get(cip)
        if not info:
            continue

        # Score each major within the CIP category
        for major in info.get("majors", []):
            if major in seen_majors:
                continue
            seen_majors.add(major)

            # Calculate match score based on skill overlap
            score = _calculate_major_fit(major, skills, ideal_careers)

            results.append({
                "major": major,
                "category": info["name"],
                "cip_code": cip,
                "fit_score": round(score, 1),
                "sample_careers": info.get("careers", [])[:3],
            })

    # Sort by fit score
    results.sort(key=lambda x: -x["fit_score"])
    return results[:top_n]


def _calculate_major_fit(
    major: str,
    skills: list[str],
    ideal_careers: list[str] | None = None,
) -> float:
    """Calculate how well a major fits the student's profile."""
    major_lower = major.lower()
    skills_lower = [s.lower() for s in skills]
    score = 0.0

    # Skill-to-major keyword matching
    skill_keywords: dict[str, list[str]] = {
        "computer science": ["python", "java", "programming", "coding", "software", "computer"],
        "data science": ["data", "statistics", "python", "machine learning", "analysis"],
        "biology": ["biology", "science", "laboratory", "research"],
        "nursing": ["healthcare", "patient", "medical", "care"],
        "business administration": ["business", "management", "leadership", "finance"],
        "psychology": ["psychology", "counseling", "mental health"],
        "graphic design": ["design", "graphic", "photoshop", "illustrator", "creative"],
        "communications": ["communication", "writing", "social media", "public speaking"],
        "education": ["teaching", "tutor", "mentoring", "classroom", "education"],
        "engineering": ["engineering", "math", "physics", "design", "cad"],
    }

    for keyword, related_skills in skill_keywords.items():
        if keyword in major_lower:
            overlap = sum(1 for s in related_skills if any(s in skill for skill in skills_lower))
            score += overlap * 0.5

    # Career alignment bonus
    if ideal_careers:
        for career in ideal_careers:
            career_lower = career.lower()
            # Check if any sample career matches
            if any(c.lower() in career_lower or career_lower in c.lower()
                   for c in _CIP_MAJORS.get("", {}).get("careers", [])):
                score += 1.0

    return min(score, 10.0)


def get_major_info(major_name: str) -> dict | None:
    """Get detailed info about a specific major."""
    for cip, info in _CIP_MAJORS.items():
        for major in info.get("majors", []):
            if major.lower() == major_name.lower():
                return {
                    "major": major,
                    "category": info["name"],
                    "cip_code": cip,
                    "related_majors": info["majors"],
                    "sample_careers": info["careers"],
                }
    return None