# Docker Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 18 March 2026
# Phase: 19 — Docker

## 1. Overview
Phase 19 containerizes the full Loan Risk Assessment System using Docker. All three components — PostgreSQL database, FastAPI backend, and Streamlit frontend — run as separate containers managed by Docker Compose. This means the entire system can be started with a single command on any machine without any manual setup.

## 2. Why Docker
| Reason           | Detail                                                           |
|------------------|------------------------------------------------------------------|
| Portability      | Runs the same on any machine — no "works on my machine" issues   |
| Isolation        | Each service runs in its own container with its own dependencies |
| Single command   | `docker-compose up --build` starts everything                    |
| Production ready | Standard industry practice for deploying ML systems              |

## 3. Files Created
| File                        | Purpose                                           | 
|-----------------------------|---------------------------------------------------|
| `Dockerfile.api`            | Builds the FastAPI + Uvicorn container            |
| `Dockerfile.frontend`       | Builds the Streamlit container                    |
| `docker-compose.yml`        | Defines and connects all 3 services               |
| `.dockerignore`             | Excludes unnecessary files from the build context |
| `requirements.api.txt`      | API-only dependencies — smaller image             |
| `requirements.frontend.txt` | Frontend-only dependencies — minimal image        |

## 4. Services

### 4.1 Service Summary
| Service    | Image                     | Port | Purpose                   |
|------------|---------------------------|------|---------------------------|
| `db`       | postgres:15               | 5432 | PostgreSQL database       |
| `api`      | loan-risk-system-api      | 8000 | FastAPI + Uvicorn backend |
| `frontend` | loan-risk-system-frontend | 8501 | Streamlit UI              |

### 4.2 Service Dependencies
db (PostgreSQL)
  └── api (waits for db to be healthy)
        └── frontend (waits for api to start)

The `db` service has a healthcheck — API only starts after PostgreSQL is ready. This prevents startup errors from DB connection failures.

## 5. Dockerfile Details

### 5.1 Dockerfile.api
Base image    : python:3.11-slim-bookworm
System deps   : gcc, g++, libpq-dev (for psycopg2)
Requirements  : requirements.api.txt
Copied files  : api/, database/, src/, models/Final_Models/,
                models/Production_pipelines/,
                models/Project_Parameter_Files/,
                models_onnx/, data/engineered_data/,
                resave_pipelines.py
Port          : 8000
Command       : uvicorn api.main:app --host 0.0.0.0 --port 8000

### 5.2 Dockerfile.frontend
Base image    : python:3.11-slim-bookworm
System deps   : gcc
Requirements  : requirements.frontend.txt
Copied files  : frontend/
Port          : 8501
Command       : streamlit run frontend/app.py --server.port=8501

## 6. Requirements Split
Splitting requirements into two files significantly reduces image sizes.

| Package              | API | Frontend | Reason                   |
|----------------------|-----|----------|--------------------------|
| fastapi, uvicorn     | YES | NO       | API only                 |
| sqlalchemy, psycopg2 | YES | NO       | DB connection API only   |
| onnxruntime          | YES | NO       | Model inference API only |
| shap, lightgbm       | YES | NO       | Explainability API only  |
| streamlit            | NO  | YES      | Frontend only            |
| requests             | YES | YES      | Both need HTTP calls     |
| numpy, pandas, scipy | YES | NO       | Data processing API only |

## 7. Environment Variables
Database credentials are passed via `docker-compose.yml` from `.env` file — never copied into the image.

| Variable    | Value in Docker                                  |
|-------------|--------------------------------------------------|
| DB_HOST     | `db` (container name — not localhost)            |
| DB_PORT     | 5432                                             |
| DB_NAME     | from .env                                        |
| DB_USER     | from .env                                        |
| DB_PASSWORD | from .env                                        |
| API_URL     | `http://api:8000` (frontend uses container name) |

## 8. How to Run

### Start everything
```bash
docker-compose up --build
```

### Start in background
```bash
docker-compose up --build -d
```

### Stop everything
```bash
docker-compose down
```

### Stop and remove volumes (clears DB)
```bash
docker-compose down -v
```

### Access
| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost:8501      |
| API docs | http://localhost:8000/docs |

## 9. Issues Faced and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `.env` not found | `.env` was in `.dockerignore` | Removed `COPY .env` — credentials passed via `docker-compose.yml` instead |
| `trixie` repos expired | `python:3.11-slim` uses unstable Debian trixie | Switched to `python:3.11-slim-bookworm` (stable) |
| `bookworm-updates` expired | Docker cached old repo metadata | Added `-o Acquire::Check-Valid-Until=false` to apt-get + ran `--no-cache` |
| `ModuleNotFoundError: yaml` | `pyyaml` missing from `requirements.api.txt` | Added `pyyaml==6.0.1` to `requirements.api.txt` |
| Frontend used `localhost` for API | In Docker, containers talk by service name not localhost | Updated `app.py` to read `API_URL` from environment variable |

## 10. Final Result
| Container          | Status                  |
|--------------------|-------------------------|
| loan_risk_db       |  Healthy                |
| loan_risk_api      |  Running on port 8000   |
| loan_risk_frontend |  Running on port 8501   |