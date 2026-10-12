# Single-Space migration guide

All paths below are relative to `C:\Users\DELL\Desktop\loan-risk-system`.
The complete updated source files are already saved in this project.

## 1. Project walkthrough, from development to deployment

1. `notebook/01_data_collection.ipynb` through `04_eda.ipynb` cover collection,
   validation, splitting, and exploration. `src/data_loader.py` reads datasets
   and project configuration. These are development steps, not startup tasks.
2. `05_preprocessing.ipynb` and `src/preprocessing_pipeline.py` define cleaning,
   capping, encoding, scaling, and custom transformers. The fitted pipelines
   are saved as joblib assets.
3. `06_feature_engineering.ipynb` and `src/feature_engineering_pipeline.py`
   create ratios, interactions, bins, domain features, and the final feature
   selection. Keep these modules: saved joblib objects import their classes.
4. `07_baseline_model_training.ipynb`, `08_model_training.ipynb`, and
   `src/model.py` cover model training. Evaluation and comparison are in
   `09_model_evaluation.ipynb`, `11_model_selection.ipynb`,
   `src/evaluate.py`, and `src/model_selection.py`.
5. `10_model_explainability.ipynb` and `src/explainability.py` cover SHAP.
   `12_model_onnx_conversion.ipynb` exports inference models; the deployed API
   also retains original joblib models and background data for SHAP.
6. The deployed model assets live in `Abdulmoiz123/loan-risk-models` on the Hub.
   `download_models.py` obtains three ONNX models, four fitted pipelines,
   two original models, feature lists, and two SHAP background CSV files.
   Downloading is skipped when each file already exists.
7. `api/main.py` creates FastAPI. Its lifespan creates database tables and loads
   all assets through `api/model_loader.py` before serving requests.
8. `api/schemas.py` validates inputs and structures responses. `api/routes.py`
   exposes health, statistics, prediction, profile lookup, and deletion.
9. `api/pipeline.py` performs the actual deployed inference. A new applicant
   follows regression -> classification -> clustering -> SHAP -> plain reasons.
   An existing loan uses its submitted interest rate and skips regression.
   The classification default-risk threshold is 0.6. The API pipeline uses
   ONNX sessions directly; `src/onnx_inference.py` is a separate helper module.
10. `database/database.py` stores applicants, predictions, inputs, explanations,
    and model metadata in PostgreSQL. The existing Neon database remains
    external to the Space so records survive container restarts.
11. `frontend/app.py` is the Streamlit interface with four tabs: new applicant,
    existing loan, applicant profile, and deletion. Streamlit's Python process
    makes HTTP requests to FastAPI; the browser does not call the API directly.
12. Previously, `hf-backend` and `hf-frontend` were separate Git checkouts,
    represented by Git links in the parent repository. Their Dockerfiles each
    exposed port 7860. The deployed frontend hardcoded the other Space's URL
    as its fallback. The root frontend had no wake-up gate.
13. The new deployment builds from the parent project root, using its `api`,
    `database`, `src`, and `frontend` folders. The two legacy HF folders are
    excluded from the image and are no longer deployment sources. The old
    split Dockerfiles and Compose setup remain legacy alternatives; use the
    new root Dockerfile and launcher for this migration.

This is a source and runtime-flow analysis. It does not re-run the training
notebooks or independently validate the model quality figures in the README.
`config/config.yaml` contains older MySQL/MongoDB and API-path settings that the
current API/database startup does not use; runtime configuration comes from
environment variables and the current Python modules.

## 2. New runtime and wake-up behavior

```text
Visitor opens the one Hugging Face Space
                  |
       HF starts/resumes its container
                  |
             start_space.py
             /             \
Streamlit :7860          API worker
(public UI)             download_models.py
     |                  FastAPI startup
     |                  create tables + load models/SHAP
     |                         |
     +--- GET http://127.0.0.1:8000/health
     |         retry until models AND database are healthy
     |
     +--- display forms and enable normal work
     +--- POST /predict/new or /predict/existing
     +--- GET/DELETE /applicant/{cnic}
```

Hugging Face wakes a sleeping Space when it receives a visitor. Both services
share that lifecycle. A Python function inside the container cannot execute
while the entire container is asleep. Here `wake_up_api()` is a readiness gate
once Streamlit starts: it retries connections and unhealthy responses for up
to five minutes, then displays a retry button if necessary. It also checks
readiness immediately before each operation.

The old function only checked HTTP 200, caught connection failures outside the
retry loop, and callers ignored its False result. The new gate checks health
content and every model/database flag; it stops the frontend run on failure.
Predictions and deletions are sent once after readiness, not automatically
retried, to avoid duplicate writes.

Only Streamlit is published on 7860. FastAPI binds to 127.0.0.1:8000 inside the
same container. There is no public API URL or public `/docs` route in this
layout. When running directly on your computer, `/docs` is reachable at
http://127.0.0.1:8000/docs. A public API alongside the UI would require a
reverse proxy; it is not needed for the requested frontend workflow.

## 3. Complete file list

| File at project root or relative location | Change and purpose |
|---|---|
| `Dockerfile` | New combined Docker Space image; installs both dependency sets and exposes 7860. |
| `start_space.py` | New launcher; starts API and UI concurrently, forces the internal API address, and stops both if either exits. |
| `requirements.space.txt` | New combined dependency entry point; includes existing requirement files, Hugging Face Hub, and a tested PyArrow pin. |
| `frontend/backend_client.py` | New bounded health polling with retries for startup failures. |
| `frontend/app.py` | Updated internal API fallback; adds `wake_up_api()` and readiness gates. |
| `api/routes.py` | Health probes use `SELECT 1` rather than aggregate prediction-statistics queries. |
| `database/database.py` | Enables connection-pool pre-ping for stale cloud database connections. |
| `download_models.py` | Loads local environment settings and uses paths relative to the project instead of the caller's working directory. |
| `.dockerignore` | Excludes secrets, legacy deployment copies, local model/data assets, and development media. |
| `.env.example` | New credential-free configuration example. Keep the existing real `.env`. |
| `README.md` | Adds Docker Space YAML metadata and replaces the two-Space deployment instructions. |
| `SINGLE_SPACE_GUIDE.md` | This analysis and migration/run guide. |
| `checks/test_single_space.py` | Offline readiness and launcher regression tests. |
| `checks/smoke_single_space.py` | Integration check using existing models and a disposable local SQLite database. |
| `checks/deploy_space.py` | Uploads the combined app and configures Secrets from your local environment. |
| `checks/verify_space.py` | Confirms live UI and API startup; optionally pauses the old API Space. |

`requirements.api.txt` and `requirements.frontend.txt` are reused unchanged.
No retraining, model conversion, schema redesign, or edits inside the two
legacy HF Git checkouts are required.

## 4. Run in the VS Code PowerShell terminal

Open `C:\Users\DELL\Desktop\loan-risk-system` in VS Code. Use Python 3.11.

```powershell
Set-Location 'C:\Users\DELL\Desktop\loan-risk-system'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.space.txt
```

Keep the existing `.env` if it already contains the working Neon connection.
If it is missing, create it:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
code .env
```

Set `DATABASE_URL` to your existing PostgreSQL connection string, including
Neon's SSL options. `HF_TOKEN` is optional for public model files and required
if access is restricted. A read token suffices for runtime model downloads.
Do not use `DB_HOST=db` with the new local launcher: that hostname belongs to
the old Compose network. Use your Neon URL, or a reachable local PostgreSQL URL.

Start both services with one command:

```powershell
.\.venv\Scripts\python.exe start_space.py
```

Open http://localhost:7860. The interface waits while the API starts. The first
startup downloads missing model files. Local API docs:
http://127.0.0.1:8000/docs. Stop both services with Ctrl+C.
No virtual-environment activation or PowerShell execution-policy change is
needed when calling the interpreter by its full relative path.

## 5. Optional Docker run

Start Docker Desktop and wait for its engine to be ready, then:

```powershell
docker build -t loan-risk-space .
docker run --rm --name loan-risk-space --env-file .env -p 7860:7860 loan-risk-space
```

Open http://localhost:7860. To stop from another terminal:

```powershell
docker stop loan-risk-space
```

Use a database URL reachable from the container. For PostgreSQL running on
Windows itself, Docker Desktop commonly uses `host.docker.internal` as the
host. A Neon URL does not need this substitution. Only the UI port is published.

## 6. Deploy the existing frontend Space as the combined Space

1. Keep `Abdulmoiz123/loan-risk-app` as your single application Space. Its existing
   Docker SDK setting is suitable. The new root README declares port 7860.
2. In that Space's Settings, add `DATABASE_URL` as a Secret using the existing
   Neon connection. Add `HF_TOKEN` as a Secret only if model access needs it.
   An old `API_URL` variable can be removed; the launcher overrides it internally.
3. Upload the root deployment files listed below, preserving their directories.
   Upload from this parent project, not from `hf-frontend` or `hf-backend`.
4. Watch Build logs, then Container logs. Look for model downloads, table creation,
   and successful model loading. Open the Space and try both prediction tabs,
   profile lookup, and deletion using your own test applicant.
5. After the combined Space works, retire the old `loan-risk-api` Space in its
   Settings. Pausing it is reversible; delete it only when you no longer need it.
   The deployment helper updates the frontend Space. The verification helper
   pauses the old API only when called with `--pause-old` after startup succeeds.

You can upload through the HF web interface, or use these commands after local
setup. Login requires a write-capable token for the Space, separate from any
runtime read token. The explicit file list excludes your real `.env`, local
models/data, videos, and nested Git checkouts.

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe upload Abdulmoiz123/loan-risk-app . . --repo-type space --include 'Dockerfile' --include 'README.md' --include 'SINGLE_SPACE_GUIDE.md' --include '.dockerignore' --include 'start_space.py' --include 'download_models.py' --include 'requirements.api.txt' --include 'requirements.frontend.txt' --include 'requirements.space.txt' --include 'api/*.py' --include 'database/*.py' --include 'src/*.py' --include 'frontend/*.py'
```

Uploading triggers a rebuild. It adds/replaces the deployment files and does
not broadly delete other remote repository contents.

## 7. Verification

Offline regression checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s checks -p 'test_*.py' -v
```

Integration check, with the normal app stopped and model assets already present:

```powershell
.\.venv\Scripts\python.exe checks/smoke_single_space.py
```

This check launches both services, checks API readiness and Streamlit rendering,
executes both prediction workflows with ONNX and SHAP, checks profile storage,
and deletes its synthetic applicant. It uses a disposable SQLite database and
does not write to your configured Neon database.

## 8. Operational notes

- Models and SHAP initialization can take time. If the five-minute UI wait expires,
  inspect logs and retry; the launcher continues starting the backend.
- A database connection failure during FastAPI startup prevents service readiness.
  Fix the connection or credentials in the Space settings and restart the Space.
- Hub model assets download at container startup and are not baked into the image.
  The public model repository remains a separate repository, not a second Space.
- The image does not package training datasets or local `.env` files.
- The existing model/library versions are retained for joblib compatibility.
  Dependency modernization is a separate change from this deployment migration.
- A sleeping Space and a manually paused Space are different: manually resume a
  paused Space through Hugging Face settings before expecting it to serve visitors.

Official references:
[Docker Spaces and port configuration](https://huggingface.co/docs/hub/spaces-sdks-docker),
[Space sleep and visitor wake-up](https://huggingface.co/docs/hub/spaces-gpus),
[HF CLI uploads](https://huggingface.co/docs/huggingface_hub/guides/cli).


## 9. Push these changes to GitHub

Your configured remote is the requested `moizishere-droid/loan-risk-system`
repository, on branch `main`. Its URL currently embeds a GitHub token. The first
command below replaces that with the ordinary repository URL. Rotate the old
token in GitHub Settings -> Developer settings -> Personal access tokens.
Authenticate future pushes through Git Credential Manager or a replacement token.

Run from the project root in VS Code PowerShell:

```powershell
git remote set-url origin https://github.com/moizishere-droid/loan-risk-system.git
git add .dockerignore .env.example Dockerfile README.md SINGLE_SPACE_GUIDE.md requirements.space.txt start_space.py download_models.py frontend/app.py frontend/backend_client.py api/routes.py database/database.py checks
git diff --cached --stat
git commit -m "Combine frontend and API in one Hugging Face Space"
git push origin main
```

The explicit file list leaves your unrelated image, text, and video files out
of this commit. It also excludes your real `.env` and downloaded model/data
files. If GitHub rejects the push because `main` changed remotely, run
`git pull --rebase origin main`, resolve any conflicts, then push again.

A GitHub push alone does not sync Hugging Face automatically. To publish later
updates using the saved local configuration, run:

```powershell
.\.venv\Scripts\python.exe checks/deploy_space.py
```

This uploads only the deployment sources and sets Space Secrets from `.env`.
It does not commit credentials to either repository. The supplied HF token must
have write permission on the target Space. The runtime should use a more limited
read token when one is available for model downloads.


## 10. Validation completed during this update

- Five offline readiness/launcher tests passed.
- Both actual processes started together locally.
- Streamlit rendered all four tabs after the health gate passed.
- Both prediction workflows passed using the existing ONNX models and SHAP.
- Applicant storage, lookup, and deletion passed against a disposable database.
- The local integration check shut down its own services and cleaned its test data.
- Changed Python source parsed successfully; Git whitespace checks passed.

The Docker Desktop engine was not running locally. Cloud build/startup verification
is performed against the deployed HF Space separately.


Cloud startup exposed an unpinned PyArrow dependency that had moved to requiring
NumPy 2, while the existing models use NumPy 1.26.4. The combined requirements
now pin PyArrow 14.0.2, matching the successfully tested local environment.
The Docker build also imports Streamlit, NumPy, SHAP, and ONNX Runtime to catch
this class of dependency failure before deployment starts.


## 11. Final hosted result (12 October 2026, Asia/Karachi)

The combined application was deployed to:
https://huggingface.co/spaces/Abdulmoiz123/loan-risk-app

Public application: https://abdulmoiz123-loan-risk-app.hf.space

Cloud verification confirmed the Space was RUNNING, the public Streamlit health
endpoint returned `ok`, and runtime logs confirmed successful model loading after
database initialization. The previous `Abdulmoiz123/loan-risk-api` Space was then
paused, then permanently deleted at your request after confirming the combined
app was healthy. Only the combined app Space remains. The GitHub changes are prepared locally; no GitHub commit or push was
made on your behalf. Use the commands in section 9 to push them.
