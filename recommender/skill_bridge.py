"""Gap-to-action bridge - recommends courses, projects, and certifications for skill gaps.

Phase 3b: For each skill gap, suggest actionable learning resources.
Uses a knowledge base of Coursera, edX, and free resources mapped to skills.
"""

from __future__ import annotations

# Course knowledge base - maps skills to recommended learning resources
# In production, this would be fetched from Coursera/edX APIs
_COURSE_KB: dict[str, list[dict]] = {
    "python": [
        {"name": "Python for Everybody", "platform": "Coursera", "url": "https://www.coursera.org/specializations/python", "type": "course", "duration": "~4 months"},
        {"name": "CS50's Introduction to Programming with Python", "platform": "edX", "url": "https://www.edx.org/course/cs50s-introduction-to-programming-with-python", "type": "course", "duration": "~9 weeks"},
    ],
    "sql": [
        {"name": "SQL for Data Science", "platform": "Coursera", "url": "https://www.coursera.org/learn/sql-for-data-science", "type": "course", "duration": "~4 weeks"},
        {"name": "Learn SQL", "platform": "Codecademy", "url": "https://www.codecademy.com/learn/learn-sql", "type": "course", "duration": "~8 hours"},
    ],
    "javascript": [
        {"name": "The Complete JavaScript Course", "platform": "Udemy", "url": "https://www.udemy.com/course/the-complete-javascript-course/", "type": "course", "duration": "~69 hours"},
        {"name": "JavaScript Algorithms and Data Structures", "platform": "freeCodeCamp", "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/", "type": "free", "duration": "~300 hours"},
    ],
    "machine learning": [
        {"name": "Machine Learning Specialization", "platform": "Coursera", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "type": "course", "duration": "~2 months"},
        {"name": "Fast.ai Practical Deep Learning", "platform": "fast.ai", "url": "https://course.fast.ai/", "type": "free", "duration": "~8 weeks"},
    ],
    "data analysis": [
        {"name": "Google Data Analytics Certificate", "platform": "Coursera", "url": "https://www.coursera.org/professional-certificates/google-data-analytics", "type": "certificate", "duration": "~6 months"},
    ],
    "project management": [
        {"name": "Google Project Management Certificate", "platform": "Coursera", "url": "https://www.coursera.org/professional-certificates/google-project-management", "type": "certificate", "duration": "~6 months"},
        {"name": "Project Management Principles", "platform": "edX", "url": "https://www.edx.org/learn/project-management", "type": "course", "duration": "~8 weeks"},
    ],
    "communication": [
        {"name": "Improve Your English Communication Skills", "platform": "Coursera", "url": "https://www.coursera.org/specializations/improve-english", "type": "course", "duration": "~4 months"},
    ],
    "leadership": [
        {"name": "Leading People and Teams", "platform": "Coursera", "url": "https://www.coursera.org/specializations/leading-teams", "type": "course", "duration": "~5 months"},
    ],
    "aws": [
        {"name": "AWS Cloud Practitioner Essentials", "platform": "AWS Training", "url": "https://aws.amazon.com/training/digital/aws-cloud-practitioner-essentials/", "type": "free", "duration": "~6 hours"},
    ],
    "cloud": [
        {"name": "Introduction to Cloud Computing", "platform": "Coursera", "url": "https://www.coursera.org/learn/introduction-to-cloud", "type": "course", "duration": "~5 weeks"},
    ],
    "excel": [
        {"name": "Excel Skills for Business", "platform": "Coursera", "url": "https://www.coursera.org/specializations/excel", "type": "course", "duration": "~6 months"},
    ],
}

# Project ideas mapped to skill clusters
_PROJECT_IDEAS: dict[str, list[str]] = {
    "python": [
        "Build a personal portfolio website with Flask",
        "Create a data analysis dashboard with Streamlit",
        "Automate a repetitive task with Python scripts",
    ],
    "machine learning": [
        "Build a sentiment analysis tool for product reviews",
        "Create an image classifier for a hobby dataset",
        "Participate in a Kaggle competition",
    ],
    "javascript": [
        "Build a to-do app with React",
        "Create a browser extension",
        "Build a real-time chat application",
    ],
    "communication": [
        "Start a blog about your field of interest",
        "Present at a local meetup or school event",
        "Create tutorial videos on a skill you know",
    ],
    "leadership": [
        "Lead a student club project",
        "Organize a community volunteer event",
        "Mentor younger students in your field",
    ],
}


def recommend_actions(
    skill_gaps: list[str],
    function: str = "",
    top_n: int = 5,
) -> list[dict]:
    """Recommend courses, projects, and certifications for skill gaps.

    Args:
        skill_gaps: List of missing skills
        function: Career function for context
        top_n: Max recommendations per skill

    Returns:
        List of action items with name, platform, url, type, duration
    """
    actions = []
    seen = set()

    for gap in skill_gaps[:top_n]:
        gap_lower = gap.lower().strip()
        # Try exact match
        resources = _COURSE_KB.get(gap_lower, [])
        # Try partial match
        if not resources:
            for key, value in _COURSE_KB.items():
                if key in gap_lower or gap_lower in key:
                    resources = value
                    break

        for resource in resources[:2]:
            key = resource["name"]
            if key not in seen:
                seen.add(key)
                actions.append({
                    "skill": gap,
                    "name": resource["name"],
                    "platform": resource["platform"],
                    "url": resource.get("url", ""),
                    "type": resource.get("type", "course"),
                    "duration": resource.get("duration", ""),
                })

        # Add project ideas
        projects = _PROJECT_IDEAS.get(gap_lower, [])
        if not projects:
            for key, value in _PROJECT_IDEAS.items():
                if key in gap_lower or gap_lower in key:
                    projects = value
                    break

        for proj in projects[:1]:
            key = f"project:{proj}"
            if key not in seen:
                seen.add(key)
                actions.append({
                    "skill": gap,
                    "name": proj,
                    "platform": "Self-Directed",
                    "url": "",
                    "type": "project",
                    "duration": "Self-paced",
                })

    return actions[:top_n]


def generate_skill_path_summary(
    current_skills: list[str],
    core_gaps: list[str],
    ideal_careers: list[str] | None = None,
) -> str:
    """Generate a human-readable skill development path summary."""
    goals = ideal_careers or ["your target career"]

    if not core_gaps:
        return (
            f"Your skills align well with {goals[0]}. "
            f"Focus on gaining hands-on experience through internships or projects "
            f"to demonstrate your abilities."
        )

    top_gaps = core_gaps[:3]
    actions = recommend_actions(top_gaps, top_n=3)

    if not actions:
        return (
            f"To work toward {goals[0]}, focus on developing: {', '.join(top_gaps)}. "
            f"Look for relevant courses on Coursera or edX, and build projects "
            f"to demonstrate these skills."
        )

    lines = [f"To move toward {goals[0]}, focus on:"]
    for action in actions[:3]:
        platform = action["platform"]
        name = action["name"]
        a_type = action["type"]
        if a_type == "project":
            lines.append(f"  - Project: {name}")
        else:
            lines.append(f"  - {a_type.title()}: {name} ({platform})")

    return "\n".join(lines)