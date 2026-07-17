"""Student lifecycle management - determines what kinds of jobs are appropriate.

Phase 1b: distinguishes high school, college freshman, and college senior stages
to recommend internships vs entry-level vs aspirational roles.
"""

from __future__ import annotations

from recommender.config import get_student_lifecycle

_CFG = get_student_lifecycle()


def detect_student_stage(study_level: str) -> dict:
    """Detect student life stage from study_level.

    Args:
        study_level: One of high-school, certificate, associate, bachelor, master, doctoral

    Returns:
        dict with stage_id, label, recommended_job_levels, aspirational_job_levels
    """
    mapping = dict(_CFG.study_level_mapping)
    stage_id = mapping.get(study_level, "college_freshman")
    stages = dict(_CFG.stages)
    stage = stages.get(stage_id, stages["college_freshman"])

    recommended = stage.get("recommended", "internships")
    aspirational = stage.get("aspirational", "entry_level")

    # Normalize to lists
    if isinstance(recommended, str):
        recommended = [recommended]
    if isinstance(aspirational, str):
        aspirational = [aspirational]

    return {
        "stage_id": stage_id,
        "label": stage.get("label", ""),
        "recommended_job_levels": _expand_levels(recommended),
        "aspirational_job_levels": _expand_levels(aspirational),
    }


def _expand_levels(categories: list[str]) -> list[str]:
    """Expand category names to actual level strings."""
    level_filter = dict(_CFG.level_filter)
    levels = []
    for cat in categories:
        # Handle plural/singular mismatch (e.g., "internships" → "intern")
        cat_normalized = cat.rstrip("s") if cat.endswith("s") else cat
        cat_normalized = cat_normalized.replace("_", "-")
        # Try exact match first, then normalized
        if cat in level_filter:
            levels.extend(level_filter[cat])
        elif cat_normalized in level_filter:
            levels.extend(level_filter[cat_normalized])
        else:
            # Try partial match
            for key, vals in level_filter.items():
                if key in cat or cat in key:
                    levels.extend(vals)
                    break
    return levels


def get_internship_levels() -> list[str]:
    """Return level filter values for internship-only search."""
    level_filter = dict(_CFG.level_filter)
    return list(level_filter.get("intern", ["intern", "Intern", "internship", "Internship"]))


def get_entry_levels() -> list[str]:
    """Return level filter values for entry-level search."""
    level_filter = dict(_CFG.level_filter)
    return list(level_filter.get("entry", ["entry", "Entry", "junior", "Junior"]))


def get_all_student_levels() -> list[str]:
    """Return all levels appropriate for students (intern + entry)."""
    return get_internship_levels() + get_entry_levels()