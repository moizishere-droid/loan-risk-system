# Loan Risk Assessment System

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
├── models/                     # Trained models + pipelines
├── models_onnx/                # ONNX production models
├── data/                       # Dataset + engineered features
├── tests/
│   └── test_api.py             # API test suite
├── docs/                       # Phase reports
├── notebooks/                  # Development notebooks
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── requirements.api.txt
└── requirements.frontend.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
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

## Quickstart — Docker

### Requirements
- Docker Desktop installed and running

### Run
```bash
git clone https://github.com/Abdulmoiz123/loan-risk-system.git
cd loan-risk-system
docker-compose up --build
```

### Access
| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

### Stop
```bash
docker-compose down
```

---

## Quickstart — Local

### Requirements
- Python 3.11
- PostgreSQL running locally

### Setup
```bash
# Clone
git clone https://github.com/Abdulmoiz123/loan-risk-system.git
cd loan-risk-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your DB credentials
```

### Run
```bash
# Terminal 1 — API
uvicorn api.main:app --reload

# Terminal 2 — Frontend
cd frontend
streamlit run app.py
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

All models are exported to ONNX format for faster inference in production.

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
| ML | scikit-learn, LightGBM, SHAP, ONNX |
| API | FastAPI, Uvicorn, Pydantic |
| Database | PostgreSQL, SQLAlchemy |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
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

---

## Author

**Abdul Moiz**
Final Year SE Student — End-to-End ML Systems

---

## License

MIT License