"""
EC Scoring — LLM Prompt Definitions
All prompts used by the Effective Communicator scoring pipeline.

Each prompt covers specific features — documented inline.
Model: claude-haiku-4-5-20251001 (cheapest; sufficient for rubric scoring)
Output: always JSON only, no preamble.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 1: Written Communication Quality
#
# Features scored by this prompt:
#   - improvement_suggestions       (FY Exit Survey)
#   - match_quality_feedback        (Round Exit Survey)
#   - cross_career_learning         (Leadership Course Survey)
#   - rating_reason                 (Internship Session Feedback)
#   - additional_feedback           (Internship Session Feedback)
#
# Sub-dimension: Written Communication (25 pts bucket)
# ─────────────────────────────────────────────────────────────────────────────

WRITTEN_COMM_QUALITY_PROMPT = """You are scoring a student's written response for the Effective Communicator competency in SPEAKHIRE, a youth career-readiness program. Evaluate the response on clarity, specificity, and professional register.

RUBRIC:
5 - Clear, organized, professional tone; specific and actionable content; no grammar issues affecting comprehension.
4 - Generally clear and professional; minor issues in structure or precision.
3 - Understandable but informal or vague; some clarity gaps.
2 - Difficult to follow or very brief; significant informality or ambiguity.
1 - Unintelligible, placeholder, or no communicative content.

STUDENT RESPONSE:
{text}

Return ONLY valid JSON, no preamble or markdown:
{{"score": <integer 1-5>, "justification": "<one sentence, max 20 words>"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 2: Stayed In Touch — Binary Parse + Initiative Quality
#
# Features scored by this prompt:
#   - stayed_in_touch               (Round Exit Survey)
#
# Produces two values: stayed_binary (0/1) and followup_quality (1-5).
# Sub-dimension: Interpersonal Skills (25 pts bucket)
# ─────────────────────────────────────────────────────────────────────────────

STAYED_IN_TOUCH_PROMPT = """You are parsing a student's response about whether they stayed in contact with their SPEAKHIRE mentor (Champion) after their internship round ended.

TASK:
1. Determine if the student actually stayed in touch (yes=1, no=0, unclear=0).
2. If yes, rate the quality of their follow-up initiative (1-5). If no, score is 1.

FOLLOW-UP QUALITY RUBRIC:
5 - Describes a specific, ongoing relationship or concrete next step they initiated.
4 - Confirms contact with some detail about how or what they discussed.
3 - Confirms contact but vaguely ("yes we texted", "we follow each other").
2 - Ambiguous — may have stayed in touch but response is unclear.
1 - Did not stay in touch, or no response.

STUDENT RESPONSE:
{text}

Return ONLY valid JSON, no preamble or markdown:
{{"stayed_binary": <0 or 1>, "followup_quality": <integer 1-5>, "justification": "<one sentence, max 20 words>"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 3: Discussion Topics Depth
#
# Features scored by this prompt:
#   - discussion_topics             (Internship Session Feedback)
#
# Scores depth and relevance of what intern discussed with their Champion.
# Sub-dimension: Verbal Communication (30 pts bucket)
# ─────────────────────────────────────────────────────────────────────────────

DISCUSSION_TOPICS_PROMPT = """You are evaluating an intern's description of what they discussed during a SPEAKHIRE mentorship session with their Champion (career mentor). Score for depth and career-relevance of the conversation.

RUBRIC:
5 - Multiple specific topics mentioned; at least one directly career or professional-growth related; shows active engagement.
4 - Two or more topics mentioned; generally relevant; some specificity.
3 - One topic mentioned with some detail, or multiple vague topics.
2 - Very brief or generic ("talked about work", "my week"); minimal content.
1 - No meaningful content; placeholder or off-topic.

STUDENT RESPONSE:
{text}

Return ONLY valid JSON, no preamble or markdown:
{{"score": <integer 1-5>, "justification": "<one sentence, max 20 words>"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 4: Self-Leadership / Cross-Career Learning Reflection
#
# Features scored by this prompt:
#   - self_leadership_learning      (Leadership Course Survey — text field)
#
# This is an EC-adjacent field: articulation quality + insight depth.
# Sub-dimension: Written Communication + Interpersonal Skills
# ─────────────────────────────────────────────────────────────────────────────

SELF_LEADERSHIP_REFLECTION_PROMPT = """You are evaluating a student's written reflection on what they learned about themselves as a leader during a SPEAKHIRE Leadership Course. Score on the quality of self-insight and clarity of expression.

RUBRIC:
5 - Specific personal insight grounded in a course experience; clear and articulate writing; shows changed self-perception or concrete takeaway.
4 - Clear insight but partially grounded; or specific but without implications for future behavior.
3 - General self-awareness statement without clear evidence or specificity.
2 - Surface-level or prompted-sounding; limited original reflection; unclear writing.
1 - Non-responsive, vague, or minimal engagement.

STUDENT RESPONSE:
{text}

Return ONLY valid JSON, no preamble or markdown:
{{"score": <integer 1-5>, "justification": "<one sentence, max 20 words>"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 5: Existing Skills Inventory — Communication Signal
#
# Features scored by this prompt:
#   - existing_skills               (Interns CRM — "3 Skills Currently Have")
#
# Checks if student self-identifies any communication-related skills,
# and scores articulation quality of the response.
# Sub-dimension: Verbal Communication
# ─────────────────────────────────────────────────────────────────────────────

EXISTING_SKILLS_COMM_PROMPT = """You are evaluating an intern's self-reported skills list from a SPEAKHIRE program application. Score two things: (1) whether any communication-related skills are mentioned, and (2) how clearly and specifically the skills are described.

COMMUNICATION SKILLS include: public speaking, listening, writing, presenting, teamwork, interpersonal skills, languages, customer service, tutoring, leadership communication, conflict resolution, or similar.

CLARITY RUBRIC (clarity_score):
5 - Skills named specifically with brief context or examples; professional phrasing.
4 - Skills named clearly, some specificity, minor vagueness.
3 - Skills named but generic ("communication", "teamwork") with no elaboration.
2 - Very vague list or single word answers.
1 - No skills listed or completely off-topic.

STUDENT RESPONSE:
{text}

Return ONLY valid JSON, no preamble or markdown:
{{"has_comm_skill": <0 or 1>, "clarity_score": <integer 1-5>, "justification": "<one sentence, max 20 words>"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Registry: maps each prompt to the features it produces
# Used by the pipeline to route text fields to the right prompt.
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_REGISTRY = {
    "written_comm_quality": {
        "prompt": WRITTEN_COMM_QUALITY_PROMPT,
        "output_keys": ["score", "justification"],
        "primary_output": "score",
        "covers_features": [
            "improvement_suggestions",
            "match_quality_feedback",
            "cross_career_learning",
            "rating_reason",
            "additional_feedback",
        ],
        "subdimension": "Written Communication",
    },
    "stayed_in_touch": {
        "prompt": STAYED_IN_TOUCH_PROMPT,
        "output_keys": ["stayed_binary", "followup_quality", "justification"],
        "primary_output": "followup_quality",
        "covers_features": ["stayed_in_touch"],
        "subdimension": "Interpersonal Skills",
    },
    "discussion_topics": {
        "prompt": DISCUSSION_TOPICS_PROMPT,
        "output_keys": ["score", "justification"],
        "primary_output": "score",
        "covers_features": ["discussion_topics"],
        "subdimension": "Verbal Communication",
    },
    "self_leadership_reflection": {
        "prompt": SELF_LEADERSHIP_REFLECTION_PROMPT,
        "output_keys": ["score", "justification"],
        "primary_output": "score",
        "covers_features": ["self_leadership_learning"],
        "subdimension": "Written Communication + Interpersonal",
    },
    "existing_skills_comm": {
        "prompt": EXISTING_SKILLS_COMM_PROMPT,
        "output_keys": ["has_comm_skill", "clarity_score", "justification"],
        "primary_output": "clarity_score",
        "covers_features": ["existing_skills"],
        "subdimension": "Verbal Communication",
    },
}
