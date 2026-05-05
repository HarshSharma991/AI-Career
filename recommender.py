"""
Machine Learning Model Module (Recommender)
AI-Based Career Recommendation System
Author: Harsh Sharma | A41105222189
"""

import pickle
import numpy as np


class CareerRecommender:
    CONFIDENCE_THRESHOLD = 0.30

    def __init__(self, model_path="model/career_model.pkl", le_path="model/label_encoder.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(le_path, "rb") as f:
            self.le = pickle.load(f)

    def recommend(self, feature_vec: np.ndarray, top_n: int = 5):
        proba = self.model.predict_proba(feature_vec)[0]
        career_names = self.le.inverse_transform(np.arange(len(proba)))

        pairs = sorted(zip(career_names, proba), key=lambda x: x[1], reverse=True)
        top = pairs[:top_n]

        low_conf = bool(top[0][1] < self.CONFIDENCE_THRESHOLD)

        return [
            {
                "career": str(name),
                "confidence": round(float(conf) * 100, 1),
                "low_confidence": low_conf,
            }
            for name, conf in top
        ]
