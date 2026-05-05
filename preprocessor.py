"""
Data Preprocessing Module
AI-Based Career Recommendation System
Author: Harsh Sharma | A41105222189
"""

import numpy as np
import pickle
import json


FEATURE_COLS = [
    "math_score", "science_score", "english_score", "cs_score",
    "prog_skill", "web_dev_skill", "db_skill", "networking_skill",
    "ml_skill", "mobile_skill", "cloud_skill", "data_analysis_skill",
    "communication", "teamwork", "leadership", "adaptability", "problem_solving",
    "interest_tech", "interest_mgmt", "interest_science", "interest_design",
    "interest_health", "interest_finance", "interest_edu",
    "personality_score",
]

FIELD_RANGES = {
    "math_score": (0, 100), "science_score": (0, 100),
    "english_score": (0, 100), "cs_score": (0, 100),
    "prog_skill": (0, 3), "web_dev_skill": (0, 3), "db_skill": (0, 3),
    "networking_skill": (0, 3), "ml_skill": (0, 3), "mobile_skill": (0, 3),
    "cloud_skill": (0, 3), "data_analysis_skill": (0, 3),
    "communication": (1, 5), "teamwork": (1, 5), "leadership": (1, 5),
    "adaptability": (1, 5), "problem_solving": (1, 5),
    "interest_tech": (0, 1), "interest_mgmt": (0, 1), "interest_science": (0, 1),
    "interest_design": (0, 1), "interest_health": (0, 1),
    "interest_finance": (0, 1), "interest_edu": (0, 1),
    "personality_score": (0.0, 1.0),
}


class InputPreprocessor:
    def __init__(self, scaler_path="model/scaler.pkl"):
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def preprocess(self, raw: dict) -> np.ndarray:
        """Validate + transform raw form dict → scaled numpy array."""
        errors = []
        coerced = {}

        for col in FEATURE_COLS:
            if col not in raw:
                errors.append(f"Missing field: {col}")
                continue
            try:
                val = float(raw[col])
            except (ValueError, TypeError):
                errors.append(f"Invalid value for {col}: {raw[col]}")
                continue
            lo, hi = FIELD_RANGES[col]
            if not (lo <= val <= hi):
                errors.append(f"{col} must be between {lo} and {hi}, got {val}")
                continue
            coerced[col] = val

        if errors:
            raise ValueError("; ".join(errors))

        vec = np.array([[coerced[c] for c in FEATURE_COLS]])
        return self.scaler.transform(vec)
