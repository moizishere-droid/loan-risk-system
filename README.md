# 🏦 Loan Risk Assessment System

An end-to-end machine learning system for loan risk prediction. Given an applicant's financial profile, the system predicts whether a loan should be approved or rejected, estimates the interest rate, segments the borrower, and explains the decision in plain English.

## 🚀 Live Demo

| Component | URL |
|---|---|
| 🖥️ Frontend | [https://abdulmoiz123-loan-risk-app.hf.space](https://abdulmoiz123-loan-risk-app.hf.space) |
| ⚡ Backend API | [https://abdulmoiz123-loan-risk-api.hf.space](https://abdulmoiz123-loan-risk-api.hf.space) |
| 📖 API Docs | [https://abdulmoiz123-loan-risk-api.hf.space/docs](https://abdulmoiz123-loan-risk-api.hf.space/docs) |
| 🤗 Models | [Abdulmoiz123/loan-risk-models](https://huggingface.co/Abdulmoiz123/loan-risk-models) |

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
Streamlit Frontend
↓
FastAPI Backend
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

## Quickstart — Docker

### Requirements
- Docker Desktop installed and running
- A Hugging Face read token from https://huggingface.co/settings/tokens

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/moizishere-droid/loan-risk-system.git
cd loan-risk-system
```

**2. Create `.env` file in project root**
```env
HF_TOKEN=hf_your_token_here
DB_HOST=db
DB_PORT=5432
DB_NAME=loan_risk_db
DB_USER=postgres
DB_PASSWORD=yourpassword
```

**3. Build and run**
```bash
docker-compose up --build
```

**4. Access**

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

**5. Stop**
```bash
docker-compose down
```

---

## Quickstart — Local

**1. Clone and setup**
```bash
git clone https://github.com/moizishere-droid/loan-risk-system.git
cd loan-risk-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.api.txt
pip install -r requirements.frontend.txt
```

**2. Create `.env` file**
```env
HF_TOKEN=hf_your_token_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=loan_risk_db
DB_USER=postgres
DB_PASSWORD=yourpassword
```

**3. Create PostgreSQL database**
```sql
CREATE DATABASE loan_risk_db;
```

**4. Download models**
```bash
python download_models.py
```

**5. Run**
```bash
uvicorn api.main:app --reload
streamlit run frontend/app.py
```

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

## Author

**Abdul Moiz**
Final Year SE Student — End-to-End ML Systems
- GitHub: [@moizishere-droid](https://github.com/moizishere-droid)
- HuggingFace: [Abdulmoiz123](https://huggingface.co/Abdulmoiz123)

---

## License

MIT License