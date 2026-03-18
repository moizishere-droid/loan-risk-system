from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from api.model_loader import load_all_models
from api.routes import router
from database.database import create_tables


# LIFESPAN — load models + create tables at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when uvicorn starts.
    - Creates DB tables if they don't exist
    - Loads all models and pipelines
    """
    print("Starting up...")
    create_tables()        # creates tables on first run, skips if already exist
    load_all_models()
    yield
    print("Shutting down...")


# APP
app = FastAPI(
    title       = "Loan Risk Assessment System",
    description = "End-to-end ML API — Classification, Regression, Clustering, SHAP",
    version     = "1.0.0",
    lifespan    = lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Routes
app.include_router(router)