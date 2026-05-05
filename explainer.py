"""
Recommendation Explanation Module
AI-Based Career Recommendation System
Author: Harsh Sharma | A41105222189
"""

import json

SKILL_LABELS = {
    "prog_skill": "Programming", "web_dev_skill": "Web Development",
    "db_skill": "Database Management", "networking_skill": "Networking",
    "ml_skill": "Machine Learning", "mobile_skill": "Mobile Development",
    "cloud_skill": "Cloud Computing", "data_analysis_skill": "Data Analysis",
}
SOFT_LABELS = {
    "communication": "Communication", "teamwork": "Teamwork",
    "leadership": "Leadership", "adaptability": "Adaptability",
    "problem_solving": "Problem-Solving",
}
INTEREST_LABELS = {
    "interest_tech": "Technology", "interest_mgmt": "Management/Business",
    "interest_science": "Science/Research", "interest_design": "Arts & Design",
    "interest_health": "Healthcare", "interest_finance": "Finance",
    "interest_edu": "Education",
}
SKILL_LEVEL = {0: "No", 1: "Beginner", 2: "Intermediate", 3: "Advanced"}

class ExplanationGenerator:
    def __init__(self, kb_path="data/career_requirements.json"):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

    def explain(self, career, raw_input):
        if career not in self.kb:
            return f"{career} aligns with your overall profile."
        meta = self.kb[career]
        parts = [meta["description"]]
        tech_strengths = []
        for field in meta["key_tech_skills"]:
            val = int(raw_input.get(field, 0))
            if val >= 2:
                tech_strengths.append(f"{SKILL_LEVEL[val]} {SKILL_LABELS.get(field, field)}")
        if tech_strengths:
            parts.append(f"Your technical strengths include: {', '.join(tech_strengths)}.")
        soft_strengths = []
        for field in meta["key_soft_skills"]:
            val = int(raw_input.get(field, 0))
            if val >= 4:
                soft_strengths.append(SOFT_LABELS.get(field, field))
        if soft_strengths:
            parts.append(f"You demonstrate strong {', '.join(soft_strengths)} skills.")
        matched_interests = []
        for field in meta["primary_interests"]:
            if int(raw_input.get(field, 0)) == 1:
                matched_interests.append(INTEREST_LABELS.get(field, field))
        if matched_interests:
            parts.append(f"Your interest in {', '.join(matched_interests)} aligns well with this career.")
        cs = float(raw_input.get("cs_score", 0))
        math = float(raw_input.get("math_score", 0))
        if career in ("AI/ML Engineer", "Data Scientist", "Software Developer", "DevOps Engineer") and math >= 80:
            parts.append(f"Your strong Mathematics score ({math:.0f}/100) further supports this recommendation.")
        elif cs >= 80:
            parts.append(f"Your Computer Science score ({cs:.0f}/100) adds to your suitability.")
        return " ".join(parts)
