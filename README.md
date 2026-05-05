# AI-Based Career Recommendation System
**Major Project (ETMJ100) | Harsh Sharma | A41105222189**  
B.Tech CSE 2022–26 | Amity University UP | Guide: Mr. Bhanu Prakash Lohani

---

## How to Run on Your Local Computer

### Prerequisites
- Python 3.9 or above  
- pip (comes with Python)
- Any modern browser (Chrome, Firefox, Edge)

---

### Step 1 — Install Python
Download Python from: https://www.python.org/downloads/  
During installation, **check "Add Python to PATH"**.

---

### Step 2 — Extract the Project
Extract the zip to a folder, e.g. `C:\CareerAI\` (Windows) or `~/CareerAI/` (Mac/Linux).

---

### Step 3 — Open Terminal / Command Prompt
- **Windows**: Press `Win + R` → type `cmd` → Enter  
- **Mac/Linux**: Open Terminal

Navigate to the project folder:
```
cd C:\CareerAI\career_recommendation_system
```

---

### Step 4 — Install Dependencies
```
pip install -r requirements.txt
```
This installs: Flask, scikit-learn, pandas, numpy.

---

### Step 5 — Train the Model (one-time setup)
```
python train_model.py
```
This generates the dataset and trains the Random Forest model.  
You will see accuracy results in the terminal (~87%).  
It saves model files to the `model/` folder.

---

### Step 6 — Run the Application
```
python app.py
```
You will see:
```
  AI-Based Career Recommendation System
  → Open http://127.0.0.1:5000 in your browser
```

---

### Step 7 — Open in Browser
Open your browser and go to:
```
http://127.0.0.1:5000
```

Fill in the 4-section form and get your personalized career recommendations!

---

## Project Structure
```
career_recommendation_system/
├── app.py                  ← Flask backend (main file to run)
├── train_model.py          ← Dataset generation + model training
├── requirements.txt        ← Python dependencies
├── model/
│   ├── career_model.pkl    ← Trained Random Forest model
│   ├── scaler.pkl          ← StandardScaler parameters
│   └── label_encoder.pkl   ← Career label encoder
├── data/
│   ├── career_dataset.csv      ← Generated training dataset
│   └── career_requirements.json ← Career knowledge base
├── utils/
│   ├── preprocessor.py     ← Input preprocessing module
│   ├── recommender.py      ← ML inference module
│   └── explainer.py        ← Explanation generation module
├── templates/
│   └── index.html          ← Frontend HTML
└── static/
    ├── css/style.css       ← Stylesheet
    └── js/app.js           ← JavaScript logic
```

## Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | Random Forest Classifier (scikit-learn) |
| Backend | Python + Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Data | pandas, NumPy |
| Dataset | 1,423 synthetic student profiles |
| Careers | 15 categories |

---

*Submitted to: Dept. of CSE, ASET, Amity University Uttar Pradesh*
