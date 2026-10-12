---
title: Loan Risk Assessment System
emoji: "🏦"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🏦 Loan Risk Assessment System

An end-to-end machine learning system for loan risk prediction. Given an applicant's financial profile, the system predicts whether a loan should be approved or rejected, estimates the interest rate, segments the borrower, and explains the decision in plain English.

## One Hugging Face Space

| Component | URL |
|---|---|
| 🖥️ Live Demo | [https://abdulmoiz123-loan-risk-app.hf.space](https://abdulmoiz123-loan-risk-app.hf.space/) |
| 🤗 Models | [Abdulmoiz123/loan-risk-models](https://huggingface.co/Abdulmoiz123/loan-risk-models) |

Deploy this repository root as a Docker Space. Streamlit runs publicly on port
7860 and calls FastAPI internally at `http://127.0.0.1:8000`.
The frontend waits for healthy models and database before displaying its forms.

See [SINGLE_SPACE_GUIDE.md](SINGLE_SPACE_GUIDE.md) for the project walkthrough,
complete changed-file list, VS Code commands, and deployment instructions.

Models remain in [Abdulmoiz123/loan-risk-models](https://huggingface.co/Abdulmoiz123/loan-risk-models).
The existing Neon PostgreSQL database remains external to the Space.

---

## What It Does

| Task | Model | Output |
|------|-------|--------|
| Loan Decision | LightGBM (Classification) | APPROVED / REJECTED + default probability |
| Interest Rate | RandomForest (Regression) | Predicted rate % |
| Borrower Segment | GMM (Clustering) | High Value Borrower / Standard Borrower |
| Explainability | SHAP + Plain English | Why the decision was made |

---

## System Architecture
User (Browser)
↓
Streamlit Frontend (port 7860)
↓ internal HTTP, same Docker Space
FastAPI Backend (127.0.0.1:8000)
↓          ↓
ONNX Models    PostgreSQL DB (Neon Cloud)

---

# Models

> ⚠️ **Models and data are NOT included in this repository.**
> They are stored on Hugging Face and downloaded automatically at startup.

---

## Model Files

All trained models, pipelines, and data files are stored on Hugging Face:

🤗 **[Abdulmoiz123/loan-risk-models](https://huggingface.co/Abdulmoiz123/loan-risk-models)**

| File | Purpose |
|------|---------|
| `models_onnx/cls_LightGBM.onnx` | Classification ONNX model |
| `models_onnx/reg_RandomForest.onnx` | Regression ONNX model |
| `models_onnx/cluster_GMM.onnx` | Clustering ONNX model |
| `models/Production_pipelines/*.joblib` | Preprocessing + FE pipelines |
| `models/Final_Models/*.joblib` | Original models for SHAP |
| `models/Project_Parameter_Files/feature_lists.json` | Feature column lists |
| `data/engineered_data/X_train_cls_engineered.csv` | SHAP background data |
| `data/engineered_data/X_train_reg_engineered.csv` | SHAP background data |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | API + model health status |
| GET | `/stats` | Prediction statistics |
| POST | `/predict/new` | New applicant — predicts rate + decision |
| POST | `/predict/existing` | Existing loan — decision only |
| GET | `/applicant/{cnic}` | Get applicant profile by CNIC |
| DELETE | `/applicant/{cnic}` | Delete applicant and all records |

---

## Two Workflows

**New Applicant** — no interest rate yet:
Regression → Classification → Clustering → SHAP → Plain Reasons

**Existing Loan** — rate already assigned:
Classification → Clustering → SHAP → Plain Reasons

---

## Quickstart in VS Code (PowerShell)

Use Python 3.11. Open the project root, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.space.txt
# Only if .env does not already exist:
Copy-Item .env.example .env
# Edit .env with your existing DATABASE_URL before starting.
.\.venv\Scripts\python.exe start_space.py
```

Open http://localhost:7860. Local API documentation is at http://127.0.0.1:8000/docs.
Stop with Ctrl+C. Never replace an existing .env that contains your credentials.

For Docker Desktop:

```powershell
docker build -t loan-risk-space .
docker run --rm --name loan-risk-space --env-file .env -p 7860:7860 loan-risk-space
```

The Docker API is internal; only the frontend is published. Set DATABASE_URL and,
if needed, HF_TOKEN as Secrets in the single Hugging Face Space's settings.

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `applicants` | Tracks unique applicants by CNIC |
| `predictions` | One row per API call — model outputs |
| `inputs` | Raw user inputs per prediction |
| `explanations` | SHAP values per prediction |
| `model_metadata` | Model version tracking |

Cascade deletes — removing an applicant removes all their predictions, inputs, and explanations.

---

## Models

| Task | Model | Key Metric |
|------|-------|-----------|
| Classification | LightGBM | F1 = 0.8390, AUC = 0.9471 |
| Regression | RandomForest | R² = 0.9077, RMSE = 0.9811 |
| Clustering | GMM | Silhouette = 0.2196 |

All models exported to ONNX format for faster inference in production.
SHAP explanations use original joblib models with TreeExplainer.

---

## Dataset

- **Source:** Credit Risk Dataset (Kaggle)
- **Size:** 32,581 rows × 12 features
- **Target (Classification):** `loan_status` — default or not
- **Target (Regression):** `loan_int_rate` — interest rate

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, LightGBM, SHAP, ONNX Runtime |
| API | FastAPI, Uvicorn, Pydantic |
| Database | PostgreSQL, SQLAlchemy, Neon Cloud |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | HuggingFace Spaces |
| Model Storage | HuggingFace Hub |
| Tuning | Optuna |

---

## Development Phases

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Problem Framing | ✅ |
| 2 | Environment Setup | ✅ |
| 3 | Data Collection | ✅ |
| 4 | Data Validation | ✅ |
| 5 | EDA | ✅ |
| 6 | Preprocessing | ✅ |
| 7 | Feature Engineering | ✅ |
| 8 | Baseline Models | ✅ |
| 9 | Model Training | ✅ |
| 10 | Hyperparameter Tuning | ✅ |
| 11 | Model Evaluation | ✅ |
| 12 | Explainability (SHAP) | ✅ |
| 13 | Model Selection | ✅ |
| 14 | ONNX Conversion | ✅ |
| 15 | Database Setup | ✅ |
| 16 | API Development | ✅ |
| 17 | Frontend | ✅ |
| 18 | Testing | ✅ |
| 19 | Docker | ✅ |
| 20 | Deployment | ✅ |

---

## Author

**Abdul Moiz**
Final Year SE Student — End-to-End ML Systems
- GitHub: [@moizishere-droid](https://github.com/moizishere-droid)
- HuggingFace: [Abdulmoiz123](https://huggingface.co/Abdulmoiz123)

---

## License

MIT License
