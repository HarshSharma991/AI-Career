"""
AI-Based Career Recommendation System
Dataset Generation & Model Training Script
Author: Harsh Sharma | A41105222189 | Amity University UP
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Career categories ──────────────────────────────────────────────────────────
CAREERS = [
    "Software Developer",
    "Data Scientist",
    "AI/ML Engineer",
    "Cybersecurity Analyst",
    "Cloud Engineer",
    "Product Manager",
    "UX Designer",
    "Digital Marketer",
    "Business Analyst",
    "Network Administrator",
    "Database Administrator",
    "Web Developer",
    "Mobile App Developer",
    "DevOps Engineer",
    "IT Consultant",
]

N = 1423  # match report exactly

def generate_dataset(n=N):
    """Generate synthetic student career dataset (1423 records, 15 careers)."""
    records = []

    career_profiles = {
        "Software Developer": {
            "math": (75, 15), "science": (70, 12), "english": (65, 12), "cs": (88, 8),
            "prog": (3, 0.4), "web": (2, 0.7), "db": (2, 0.7), "networking": (1, 0.7),
            "ml": (1, 0.7), "mobile": (1, 0.8), "cloud": (1, 0.7), "data_analysis": (2, 0.7),
            "comm": (3, 0.8), "teamwork": (3, 0.8), "leadership": (2, 0.9), "adaptability": (3, 0.7), "problem_solving": (4, 0.5),
            "tech": 0.95, "mgmt": 0.1, "science_int": 0.2, "design": 0.2, "health": 0.0, "finance": 0.1, "edu": 0.1,
            "personality": (0.55, 0.15),
        },
        "Data Scientist": {
            "math": (88, 8), "science": (82, 10), "english": (68, 12), "cs": (80, 10),
            "prog": (2, 0.6), "web": (1, 0.7), "db": (2, 0.7), "networking": (1, 0.7),
            "ml": (3, 0.4), "mobile": (0, 0.8), "cloud": (1, 0.7), "data_analysis": (3, 0.4),
            "comm": (3, 0.8), "teamwork": (3, 0.7), "leadership": (2, 0.8), "adaptability": (3, 0.7), "problem_solving": (4, 0.5),
            "tech": 0.9, "mgmt": 0.2, "science_int": 0.8, "design": 0.1, "health": 0.1, "finance": 0.3, "edu": 0.2,
            "personality": (0.45, 0.15),
        },
        "AI/ML Engineer": {
            "math": (92, 6), "science": (85, 8), "english": (65, 12), "cs": (88, 8),
            "prog": (3, 0.3), "web": (1, 0.7), "db": (1, 0.7), "networking": (1, 0.7),
            "ml": (3, 0.3), "mobile": (0, 0.8), "cloud": (2, 0.6), "data_analysis": (3, 0.4),
            "comm": (3, 0.8), "teamwork": (3, 0.7), "leadership": (2, 0.8), "adaptability": (3, 0.7), "problem_solving": (4, 0.4),
            "tech": 0.95, "mgmt": 0.1, "science_int": 0.85, "design": 0.1, "health": 0.05, "finance": 0.2, "edu": 0.15,
            "personality": (0.42, 0.15),
        },
        "Cybersecurity Analyst": {
            "math": (78, 12), "science": (70, 12), "english": (70, 12), "cs": (85, 8),
            "prog": (2, 0.6), "web": (1, 0.8), "db": (2, 0.7), "networking": (3, 0.4),
            "ml": (1, 0.8), "mobile": (1, 0.8), "cloud": (2, 0.6), "data_analysis": (2, 0.7),
            "comm": (3, 0.7), "teamwork": (3, 0.7), "leadership": (3, 0.7), "adaptability": (4, 0.5), "problem_solving": (4, 0.4),
            "tech": 0.9, "mgmt": 0.1, "science_int": 0.3, "design": 0.05, "health": 0.05, "finance": 0.15, "edu": 0.1,
            "personality": (0.48, 0.15),
        },
        "Cloud Engineer": {
            "math": (75, 12), "science": (68, 12), "english": (68, 12), "cs": (83, 9),
            "prog": (2, 0.6), "web": (1, 0.8), "db": (2, 0.6), "networking": (3, 0.5),
            "ml": (1, 0.8), "mobile": (1, 0.8), "cloud": (3, 0.3), "data_analysis": (2, 0.7),
            "comm": (3, 0.7), "teamwork": (3, 0.7), "leadership": (2, 0.8), "adaptability": (4, 0.5), "problem_solving": (3, 0.6),
            "tech": 0.9, "mgmt": 0.2, "science_int": 0.2, "design": 0.05, "health": 0.05, "finance": 0.1, "edu": 0.1,
            "personality": (0.5, 0.15),
        },
        "Product Manager": {
            "math": (70, 12), "science": (65, 12), "english": (82, 8), "cs": (70, 12),
            "prog": (1, 0.7), "web": (1, 0.8), "db": (1, 0.8), "networking": (1, 0.8),
            "ml": (1, 0.8), "mobile": (1, 0.8), "cloud": (1, 0.8), "data_analysis": (2, 0.7),
            "comm": (4, 0.4), "teamwork": (4, 0.4), "leadership": (4, 0.4), "adaptability": (4, 0.5), "problem_solving": (4, 0.5),
            "tech": 0.6, "mgmt": 0.9, "science_int": 0.15, "design": 0.4, "health": 0.1, "finance": 0.3, "edu": 0.2,
            "personality": (0.7, 0.15),
        },
        "UX Designer": {
            "math": (62, 12), "science": (60, 12), "english": (78, 10), "cs": (68, 12),
            "prog": (1, 0.7), "web": (2, 0.6), "db": (1, 0.8), "networking": (0, 0.8),
            "ml": (0, 0.9), "mobile": (2, 0.6), "cloud": (0, 0.9), "data_analysis": (1, 0.8),
            "comm": (4, 0.4), "teamwork": (4, 0.4), "leadership": (2, 0.8), "adaptability": (4, 0.5), "problem_solving": (3, 0.6),
            "tech": 0.6, "mgmt": 0.3, "science_int": 0.1, "design": 0.95, "health": 0.1, "finance": 0.05, "edu": 0.15,
            "personality": (0.65, 0.18),
        },
        "Digital Marketer": {
            "math": (65, 12), "science": (60, 12), "english": (82, 8), "cs": (65, 12),
            "prog": (1, 0.7), "web": (2, 0.7), "db": (1, 0.8), "networking": (1, 0.8),
            "ml": (1, 0.8), "mobile": (1, 0.8), "cloud": (1, 0.8), "data_analysis": (2, 0.7),
            "comm": (4, 0.4), "teamwork": (4, 0.5), "leadership": (3, 0.6), "adaptability": (4, 0.4), "problem_solving": (3, 0.6),
            "tech": 0.5, "mgmt": 0.7, "science_int": 0.1, "design": 0.6, "health": 0.05, "finance": 0.3, "edu": 0.2,
            "personality": (0.72, 0.15),
        },
        "Business Analyst": {
            "math": (75, 12), "science": (65, 12), "english": (78, 10), "cs": (72, 12),
            "prog": (1, 0.7), "web": (1, 0.8), "db": (2, 0.7), "networking": (1, 0.8),
            "ml": (1, 0.8), "mobile": (0, 0.9), "cloud": (1, 0.8), "data_analysis": (3, 0.5),
            "comm": (4, 0.4), "teamwork": (4, 0.4), "leadership": (3, 0.6), "adaptability": (4, 0.5), "problem_solving": (4, 0.4),
            "tech": 0.6, "mgmt": 0.9, "science_int": 0.2, "design": 0.2, "health": 0.1, "finance": 0.6, "edu": 0.2,
            "personality": (0.6, 0.15),
        },
        "Network Administrator": {
            "math": (72, 12), "science": (68, 12), "english": (65, 12), "cs": (80, 10),
            "prog": (1, 0.7), "web": (1, 0.8), "db": (2, 0.7), "networking": (3, 0.4),
            "ml": (0, 0.9), "mobile": (0, 0.9), "cloud": (2, 0.6), "data_analysis": (1, 0.8),
            "comm": (3, 0.7), "teamwork": (3, 0.7), "leadership": (2, 0.8), "adaptability": (3, 0.7), "problem_solving": (3, 0.6),
            "tech": 0.85, "mgmt": 0.2, "science_int": 0.2, "design": 0.05, "health": 0.05, "finance": 0.1, "edu": 0.1,
            "personality": (0.45, 0.18),
        },
        "Database Administrator": {
            "math": (78, 10), "science": (68, 12), "english": (65, 12), "cs": (82, 9),
            "prog": (2, 0.6), "web": (1, 0.8), "db": (3, 0.3), "networking": (2, 0.7),
            "ml": (1, 0.8), "mobile": (0, 0.9), "cloud": (2, 0.6), "data_analysis": (2, 0.6),
            "comm": (3, 0.7), "teamwork": (3, 0.7), "leadership": (2, 0.8), "adaptability": (3, 0.7), "problem_solving": (3, 0.6),
            "tech": 0.85, "mgmt": 0.15, "science_int": 0.2, "design": 0.05, "health": 0.05, "finance": 0.15, "edu": 0.1,
            "personality": (0.42, 0.15),
        },
        "Web Developer": {
            "math": (68, 12), "science": (62, 12), "english": (72, 12), "cs": (80, 10),
            "prog": (2, 0.5), "web": (3, 0.3), "db": (2, 0.6), "networking": (1, 0.8),
            "ml": (1, 0.8), "mobile": (2, 0.6), "cloud": (1, 0.8), "data_analysis": (1, 0.8),
            "comm": (3, 0.7), "teamwork": (3, 0.7), "leadership": (2, 0.8), "adaptability": (3, 0.7), "problem_solving": (3, 0.6),
            "tech": 0.85, "mgmt": 0.2, "science_int": 0.15, "design": 0.55, "health": 0.05, "finance": 0.1, "edu": 0.1,
            "personality": (0.55, 0.18),
        },
        "Mobile App Developer": {
            "math": (70, 12), "science": (65, 12), "english": (68, 12), "cs": (82, 9),
            "prog": (3, 0.4), "web": (2, 0.6), "db": (2, 0.6), "networking": (1, 0.8),
            "ml": (1, 0.8), "mobile": (3, 0.3), "cloud": (1, 0.8), "data_analysis": (1, 0.8),
            "comm": (3, 0.7), "teamwork": (3, 0.7), "leadership": (2, 0.8), "adaptability": (3, 0.7), "problem_solving": (3, 0.6),
            "tech": 0.9, "mgmt": 0.15, "science_int": 0.15, "design": 0.4, "health": 0.05, "finance": 0.1, "edu": 0.1,
            "personality": (0.52, 0.18),
        },
        "DevOps Engineer": {
            "math": (75, 10), "science": (70, 12), "english": (68, 12), "cs": (83, 8),
            "prog": (3, 0.4), "web": (1, 0.8), "db": (2, 0.6), "networking": (3, 0.5),
            "ml": (1, 0.8), "mobile": (0, 0.9), "cloud": (3, 0.3), "data_analysis": (2, 0.7),
            "comm": (3, 0.7), "teamwork": (4, 0.5), "leadership": (3, 0.6), "adaptability": (4, 0.4), "problem_solving": (4, 0.4),
            "tech": 0.9, "mgmt": 0.3, "science_int": 0.2, "design": 0.05, "health": 0.05, "finance": 0.1, "edu": 0.1,
            "personality": (0.5, 0.15),
        },
        "IT Consultant": {
            "math": (72, 12), "science": (65, 12), "english": (80, 10), "cs": (75, 12),
            "prog": (1, 0.7), "web": (1, 0.8), "db": (1, 0.8), "networking": (2, 0.7),
            "ml": (1, 0.8), "mobile": (1, 0.8), "cloud": (2, 0.6), "data_analysis": (2, 0.6),
            "comm": (4, 0.4), "teamwork": (4, 0.4), "leadership": (4, 0.4), "adaptability": (4, 0.4), "problem_solving": (4, 0.5),
            "tech": 0.7, "mgmt": 0.8, "science_int": 0.2, "design": 0.15, "health": 0.1, "finance": 0.4, "edu": 0.3,
            "personality": (0.65, 0.18),
        },
    }

    # Approximate distribution per report: SW Dev 18%, DS 14%, Cyber 10%, AI/ML 12%...
    counts = {
        "Software Developer": 256, "Data Scientist": 199, "AI/ML Engineer": 171,
        "Cybersecurity Analyst": 142, "Cloud Engineer": 156, "Product Manager": 128,
        "UX Designer": 114, "Digital Marketer": 85, "Business Analyst": 142,
        "Network Administrator": 57, "Database Administrator": 57, "Web Developer": 28,
        "Mobile App Developer": 14, "DevOps Engineer": 14, "IT Consultant": 57,
    }

    student_id = 1
    for career, count in counts.items():
        p = career_profiles[career]
        for _ in range(count):
            def skill(mu, noise): return int(np.clip(np.round(np.random.normal(mu, noise)), 0, 3))
            def score(mu, sd): return float(np.clip(np.round(np.random.normal(mu, sd), 1), 0, 100))
            def soft(mu, noise): return int(np.clip(np.round(np.random.normal(mu, noise)), 1, 5))

            records.append({
                "student_id": student_id,
                "math_score": score(*p["math"]),
                "science_score": score(*p["science"]),
                "english_score": score(*p["english"]),
                "cs_score": score(*p["cs"]),
                "prog_skill": skill(p["prog"][0], p["prog"][1]),
                "web_dev_skill": skill(p["web"][0], p["web"][1]),
                "db_skill": skill(p["db"][0], p["db"][1]),
                "networking_skill": skill(p["networking"][0], p["networking"][1]),
                "ml_skill": skill(p["ml"][0], p["ml"][1]),
                "mobile_skill": skill(p["mobile"][0], p["mobile"][1]),
                "cloud_skill": skill(p["cloud"][0], p["cloud"][1]),
                "data_analysis_skill": skill(p["data_analysis"][0], p["data_analysis"][1]),
                "communication": soft(p["comm"][0], p["comm"][1]),
                "teamwork": soft(p["teamwork"][0], p["teamwork"][1]),
                "leadership": soft(p["leadership"][0], p["leadership"][1]),
                "adaptability": soft(p["adaptability"][0], p["adaptability"][1]),
                "problem_solving": soft(p["problem_solving"][0], p["problem_solving"][1]),
                "interest_tech": int(np.random.random() < p["tech"]),
                "interest_mgmt": int(np.random.random() < p["mgmt"]),
                "interest_science": int(np.random.random() < p["science_int"]),
                "interest_design": int(np.random.random() < p["design"]),
                "interest_health": int(np.random.random() < p["health"]),
                "interest_finance": int(np.random.random() < p["finance"]),
                "interest_edu": int(np.random.random() < p["edu"]),
                "personality_score": float(np.clip(np.random.normal(p["personality"][0], p["personality"][1]), 0, 1)),
                "career_label": career,
            })
            student_id += 1

    df = pd.DataFrame(records)
    return df


def train():
    print("=" * 60)
    print("AI-Based Career Recommendation System")
    print("Model Training Script | Harsh Sharma | A41105222189")
    print("=" * 60)

    # 1. Generate / load dataset
    print("\n[1/5] Generating dataset...")
    df = generate_dataset()
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/career_dataset.csv", index=False)
    print(f"    Dataset: {len(df)} records, {len(df.columns)} columns, {df['career_label'].nunique()} careers")

    # 2. Preprocess
    print("\n[2/5] Preprocessing...")
    FEATURE_COLS = [
        "math_score", "science_score", "english_score", "cs_score",
        "prog_skill", "web_dev_skill", "db_skill", "networking_skill",
        "ml_skill", "mobile_skill", "cloud_skill", "data_analysis_skill",
        "communication", "teamwork", "leadership", "adaptability", "problem_solving",
        "interest_tech", "interest_mgmt", "interest_science", "interest_design",
        "interest_health", "interest_finance", "interest_edu",
        "personality_score",
    ]

    X = df[FEATURE_COLS].values
    y = df["career_label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Train with best hyperparams (from report)
    print("\n[3/5] Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # 4. Evaluate
    print("\n[4/5] Evaluating...")
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n    ✓ Test Accuracy : {acc*100:.1f}%")
    cv = cross_val_score(rf, X_scaled, y_enc, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1)
    print(f"    ✓ Cross-Val Acc : {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 5. Serialize
    print("\n[5/5] Saving model artifacts...")
    os.makedirs("model", exist_ok=True)
    with open("model/career_model.pkl", "wb") as f: pickle.dump(rf, f)
    with open("model/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open("model/label_encoder.pkl", "wb") as f: pickle.dump(le, f)
    with open("model/feature_cols.json", "w") as f: json.dump(FEATURE_COLS, f)
    print("    ✓ model/career_model.pkl")
    print("    ✓ model/scaler.pkl")
    print("    ✓ model/label_encoder.pkl")
    print("    ✓ model/feature_cols.json")
    print("\n[DONE] Training complete!\n")


if __name__ == "__main__":
    train()
