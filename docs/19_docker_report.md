# Docker setup

The project has one Dockerfile shared by local Docker Compose and Hugging Face.

## Local laptop

Docker Desktop must be running. From the repository root:

```powershell
docker compose up --build
```

Open http://localhost:7860. Compose starts:

- `app`: Streamlit and its private FastAPI service using the root Dockerfile.
- `db`: PostgreSQL 15 with the persistent `postgres_data` volume.

Local database settings default to postgres/localdev/loan_risk_db. Optional
DB_USER, DB_PASSWORD, and DB_NAME values in .env override them. Compose ignores
DATABASE_URL so an existing Neon configuration does not replace its local DB.
Only port 7860 is published; APP_PORT can change that host port.

Stop while preserving records:

```powershell
docker compose down
```

Existing populated database volumes need their original database credentials.

## Hugging Face

The Space builds the same root Dockerfile and exposes Streamlit on port 7860.
FastAPI runs privately on 127.0.0.1:8000. DATABASE_URL is supplied as a Secret
for the external Neon database. PostgreSQL is not run inside the Space.

## Dependencies and startup

The Dockerfile installs requirements.txt. requirements-dev.txt is optional for
training, notebooks, and development; it includes requirements.txt.

start_space.py starts Streamlit and the API worker together. The worker downloads
model assets, initializes tables, and loads models. The frontend polls API
health and enables operations only when models and the database are ready.
The image runs as a non-root user and excludes .env, local datasets, virtual
environments, and model files from its build context.

See README.md and SINGLE_SPACE_GUIDE.md for the full commands and file structure.
