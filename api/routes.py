from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.model_loader import is_loaded, get_model_status

from api.pipeline import run_new_applicant, run_existing_loan

from api.schemas import (
    NewApplicantInput, ExistingLoanInput,
    PredictionResponse, HealthResponse, StatsResponse,
    ApplicantResponse, PredictionRecord,
    InputRecord, ExplanationRecord
)
from database.database import (
    get_session, close_session,
    save_full_prediction, get_prediction_stats,
    get_applicant_history,
    delete_applicant_by_cnic
)


router = APIRouter()


# DEPENDENCY — DB SESSION
def get_db():
    """FastAPI dependency — provides a DB session per request."""
    session = get_session()
    try:
        yield session
    finally:
        close_session(session)


# HEALTH & STATS
@router.get("/", tags=["Root"])
def root():
    """Root endpoint — API info."""
    return {
        "name"   : "Loan Risk Assessment System",
        "version": "1.0.0",
        "status" : "running"
    }

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    Returns model load status and DB connection status.
    """
    try:
        stats  = get_prediction_stats(db)
        db_ok  = True
    except Exception:
        db_ok  = False

    models = get_model_status()
    models['database'] = db_ok

    return HealthResponse(
        status  = "healthy" if is_loaded() and db_ok else "unhealthy",
        version = "1.0.0",
        models  = models
    )

@router.get("/stats", response_model=StatsResponse, tags=["Stats"])
def stats(db: Session = Depends(get_db)):
    """
    Prediction statistics endpoint.
    Returns total, approved, rejected counts and averages.
    """
    return get_prediction_stats(db)


# APPLICANT — GET BY CNIC
@router.get("/applicant/{cnic}", response_model=ApplicantResponse,tags=["Applicant"])
def get_applicant(cnic: str, db: Session = Depends(get_db)):
    """
    Get full applicant profile by CNIC.
    Returns:
        - Applicant summary (visits, approved, rejected)
        - Last visit full detail (decision, inputs, SHAP)
        - Full prediction history
    """
    # Get applicant
    applicant, history = get_applicant_history(db, cnic)

    if not applicant:
        raise HTTPException(
            status_code=404,
            detail=f"No applicant found with CNIC: {cnic}"
        )

    # Build prediction records
    prediction_records = []
    for pred in history:
        # Get inputs for this prediction
        inp = pred.inputs[0] if pred.inputs else None
        exps = pred.explanations

        # Build input record
        input_record = InputRecord(
            loan_amnt                 = inp.loan_amnt                 if inp else None,
            loan_int_rate             = inp.loan_int_rate             if inp else None,
            loan_grade                = inp.loan_grade                if inp else None,
            loan_percent_income       = inp.loan_percent_income       if inp else None,
            loan_intent               = inp.loan_intent               if inp else None,
            person_income             = inp.person_income             if inp else None,
            person_age                = inp.person_age                if inp else None,
            person_emp_length         = inp.person_emp_length         if inp else None,
            person_home_ownership     = inp.person_home_ownership     if inp else None,
            cb_person_default_on_file = inp.cb_person_default_on_file if inp else None
        ) if inp else None

        # Build explanation records
        exp_records = [
            ExplanationRecord(
                task          = e.task,
                feature_name  = e.feature_name,
                readable_name = e.readable_name,
                shap_impact   = e.shap_impact,
                direction     = e.direction
            ) for e in exps
        ]

        prediction_records.append(PredictionRecord(
            prediction_id       = pred.id,
            decision            = pred.decision,
            default_probability = pred.default_probability,
            interest_rate       = pred.interest_rate,
            cluster_label       = pred.cluster_label,
            inputs              = input_record,
            explanations        = exp_records
        ))

    # Last visit = most recent prediction
    last_visit = prediction_records[0] if prediction_records else None

    return ApplicantResponse(
        cnic               = applicant.cnic,
        first_seen         = str(applicant.first_seen),
        last_seen          = str(applicant.last_seen),
        total_visits       = applicant.total_visits,
        total_approved     = applicant.total_approved,
        total_rejected     = applicant.total_rejected,
        last_decision      = applicant.last_decision,
        last_loan_amnt     = applicant.last_loan_amnt,
        last_interest_rate = applicant.last_interest_rate,
        last_visit         = last_visit,
        #history            = prediction_records   -> we can uncomment it but for now no
    )


# DELETE - APPLICANT
@router.delete("/applicant/{cnic}", tags=["Applicant"])
def delete_applicant(cnic: str, db: Session = Depends(get_db)):
    """Delete applicant and all their records by CNIC."""
    deleted = delete_applicant_by_cnic(db, cnic)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No applicant found with CNIC: {cnic}"
        )
    return {"message": f"Applicant {cnic} and all records deleted successfully"}


# PREDICT — NEW APPLICANT
@router.post("/predict/new", response_model=PredictionResponse, tags=["Predict"])
def predict_new(input_data : NewApplicantInput, db : Session = Depends(get_db)):
    """
    New Applicant workflow.
    - No loan_int_rate provided
    - Runs: Regression → Classification → Clustering → SHAP
    - Saves full prediction to database
    """
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet. Please retry in a moment."
        )
    try:
        # Run pipeline
        result = run_new_applicant(input_data.model_dump())
        # Save to DB
        saved = save_full_prediction(
            session         = db,
            cnic            = input_data.cnic,
            prediction_data = result['prediction_data'],
            input_data      = result['input_data'],
            explanations = (
                [{'task': 'classification', **r}
                for r in result['prediction_data']['classification']['reasons_for_default']] +
                [{'task': 'classification', **r}
                for r in result['prediction_data']['classification']['reasons_against_default']] +
                [{'task': 'regression', **r}
                for r in result['prediction_data']['regression']['reasons_high_rate']] +
                [{'task': 'regression', **r}
                for r in result['prediction_data']['regression']['reasons_low_rate']]
            )
        )
        # Build response
        return PredictionResponse(
            prediction_id  = saved['prediction'].id,
            workflow       = result['workflow'],
            classification = result['prediction_data']['classification'],
            regression     = result['prediction_data']['regression'],
            clustering     = result['prediction_data']['clustering'],
            plain_reasons  = result['prediction_data']['plain_reasons']
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# PREDICT — EXISTING LOAN
@router.post("/predict/existing", response_model=PredictionResponse, tags=["Predict"])
def predict_existing(input_data : ExistingLoanInput, db : Session = Depends(get_db)):
    """
    Existing Loan workflow.
    - loan_int_rate is provided
    - Runs: Classification → Clustering → SHAP
    - Regression is skipped
    - Saves full prediction to database
    """
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet. Please retry in a moment."
        )
    try:
        # Run pipeline
        result = run_existing_loan(input_data.model_dump())
        # Save to DB
        saved = save_full_prediction(
            session         = db,
            cnic            = input_data.cnic,
            prediction_data = result['prediction_data'],
            input_data      = result['input_data'],
            explanations = (
                [{'task': 'classification', **r}
                for r in result['prediction_data']['classification']['reasons_for_default']] +
                [{'task': 'classification', **r}
                for r in result['prediction_data']['classification']['reasons_against_default']]
            )
        )
        # Build response
        return PredictionResponse(
            prediction_id  = saved['prediction'].id,
            workflow       = result['workflow'],
            classification = result['prediction_data']['classification'],
            regression     = None,
            clustering     = result['prediction_data']['clustering'],
            plain_reasons  = result['prediction_data']['plain_reasons']
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )