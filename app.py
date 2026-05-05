"""
Backend API Module - Flask Application
AI-Based Career Recommendation System
Author: Harsh Sharma | A41105222189 | Amity University UP
"""

import json
import os
from flask import Flask, request, jsonify, render_template

from preprocessor import InputPreprocessor
from recommender import CareerRecommender
from explainer import ExplanationGenerator

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
preprocessor = InputPreprocessor(os.path.join(BASE, "model/scaler.pkl"))
recommender  = CareerRecommender(
    os.path.join(BASE, "model/career_model.pkl"),
    os.path.join(BASE, "model/label_encoder.pkl"),
)
explainer = ExplanationGenerator(os.path.join(BASE, "data/career_requirements.json"))

with open(os.path.join(BASE, "data/career_requirements.json"), encoding="utf-8") as f:
    CAREER_KB = json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "Random Forest", "accuracy": "87.5%"})

@app.route("/careers")
def careers():
    return jsonify({"careers": list(CAREER_KB.keys())})

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        raw = request.get_json(force=True)
        if not raw:
            return jsonify({"error": "No JSON body received."}), 400
        feature_vec = preprocessor.preprocess(raw)
        results = recommender.recommend(feature_vec, top_n=5)
        for r in results:
            career = r["career"]
            r["explanation"] = explainer.explain(career, raw)
            meta = CAREER_KB.get(career, {})
            r["description"] = meta.get("description", "")
            r["emoji"] = meta.get("emoji", "🎯")
            r["avg_salary"] = meta.get("avg_salary", "N/A")
            r["top_skills"] = meta.get("top_skills_label", [])
        return jsonify({"recommendations": results})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Unexpected error")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI-Based Career Recommendation System")
    print("  Author : Harsh Sharma | A41105222189")
    print("  Guide  : Mr. Bhanu Prakash Lohani")
    print("  Dept   : CSE, Amity University UP")
    print("="*60)
    print("\n  Open http://127.0.0.1:5000 in your browser\n")
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=False)
