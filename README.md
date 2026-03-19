# 🏦 Loan Risk Assessment System

An end-to-end machine learning system for loan risk prediction. Given an applicant's financial profile, the system predicts whether a loan should be approved or rejected, estimates the interest rate, segments the borrower, and explains the decision in plain English.

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

```
User (Browser)
    ↓
Streamlit Frontend  (port 8501)
    ↓
FastAPI Backend     (port 8000)
    ↓          ↓
ONNX Models    PostgreSQL DB
```

---

## Project Structure

```
loan-risk-system/
├── api/                        # FastAPI backend
│   ├── main.py                 # App entry point
│   ├── routes.py               # API endpoints
│   ├── pipeline.py             # ML inference pipeline
│   ├── model_loader.py         # Model loading at startup
│   └── schemas.py              # Request/response schemas
├── database/
│   └── database.py             # SQLAlchemy ORM + CRUD
├── frontend/
│   └── app.py                  # Streamlit UI
├── src/                        # ML training code
├── models/                     # Downloaded at runtime — see below
├── models_onnx/                # Downloaded at runtime — see below
├── data/                       # Downloaded at runtime — see below
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── download_models.py          # Downloads models from Hugging Face
├── requirements.api.txt
└── requirements.frontend.txt
```

> ⚠️ **Models and data are NOT included in this repository.**
> They are stored on Hugging Face and downloaded automatically at startup.
> See [Model Files](#model-files) section below.

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

These are downloaded automatically by `download_models.py` at startup.

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
```
Regression → Classification → Clustering → SHAP → Plain Reasons
```

**Existing Loan** — rate already assigned:
```
Classification → Clustering → SHAP → Plain Reasons
```

---

## Quickstart — Docker (Recommended)

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
# Hugging Face
HF_TOKEN=hf_your_token_here

# PostgreSQL
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

> First run will take a few minutes — models are downloaded from Hugging Face automatically.

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

## Quickstart — Local (Without Docker)

### Requirements
- Python 3.11
- PostgreSQL installed and running
- Hugging Face read token

### Steps

**1. Clone and setup**
```bash
git clone https://github.com/moizishere-droid/loan-risk-system.git
cd loan-risk-system

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.api.txt
pip install -r requirements.frontend.txt
pip install huggingface_hub
```

**2. Create `.env` file in project root**
```env
# Hugging Face
HF_TOKEN=hf_your_token_here

# PostgreSQL
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

**4. Download models from Hugging Face**
```bash
python download_models.py
```

**5. Run the API**
```bash
# Terminal 1
uvicorn api.main:app --reload
```

**6. Run the Frontend**
```bash
# Terminal 2
cd frontend
streamlit run app.py
```

**7. Access**

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

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
| Database | PostgreSQL, SQLAlchemy |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Model Storage | Hugging Face Hub |
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
| 19 | Deployment | ✅ |

---

## Author

**Abdul Moiz**
Final Year SE Student — End-to-End ML Systems

---

## License

MIT License