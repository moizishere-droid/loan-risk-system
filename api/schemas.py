from pydantic import BaseModel, Field
from typing import Optional


# INPUT SCHEMAS
class NewApplicantInput(BaseModel):
    """
    Input schema for New Applicant workflow.
    loan_int_rate is NOT provided — predicted by regression model.
    """
    # In NewApplicantInput — add after loan_percent_income
    cnic                       : str = Field(..., min_length=13, max_length=15,description="Applicant CNIC — format: 42101-1234567-1")
    loan_amnt                  : float = Field(..., gt=0,             description="Loan amount requested")
    loan_grade                 : str   = Field(..., pattern="^[A-G]$",description="Loan grade A-G")
    loan_intent                : str   = Field(...,                   description="Purpose of loan")
    loan_percent_income        : float = Field(..., gt=0, lt=1,       description="Loan as fraction of income")
    person_income              : float = Field(..., gt=0,             description="Annual income")
    person_age                 : int   = Field(..., gt=17, lt=100,    description="Age of applicant")
    person_emp_length          : float = Field(..., ge=0,             description="Employment length in years")
    person_home_ownership      : str   = Field(...,                   description="RENT / OWN / MORTGAGE / OTHER")
    cb_person_default_on_file  : str   = Field(..., pattern="^[YN]$", description="Previous default Y or N")
    cb_person_cred_hist_length : float = Field(..., ge=0,             description="Credit history length in years")
    # 11 col is loan_int_rate which we will predict in reg model

    # Example input
    model_config = {
        "json_schema_extra": {
            "example": {
                "cnic"                      : "42101-1234567-1",
                "loan_amnt"                 : 15000,
                "loan_grade"                : "F",
                "loan_intent"               : "DEBTCONSOLIDATION",
                "loan_percent_income"       : 0.43,
                "person_income"             : 35000,
                "person_age"                : 28,
                "person_emp_length"         : 2.0,
                "person_home_ownership"     : "RENT",
                "cb_person_default_on_file" : "N",
                "cb_person_cred_hist_length": 3.0
            }
        }
    }


class ExistingLoanInput(BaseModel):
    """
    Input schema for Existing Loan workflow.
    loan_int_rate IS provided — regression model is skipped.
    """
    cnic : str = Field(..., min_length=13, max_length=15,description="Applicant CNIC — format: 42101-1234567-1")
    loan_amnt                  : float = Field(..., gt=0,             description="Loan amount requested")
    loan_int_rate              : float = Field(..., gt=0, lt=100,     description="Interest rate already assigned")
    loan_grade                 : str   = Field(..., pattern="^[A-G]$",description="Loan grade A-G")
    loan_intent                : str   = Field(...,                   description="Purpose of loan")
    loan_percent_income        : float = Field(..., gt=0, lt=1,       description="Loan as fraction of income")
    person_income              : float = Field(..., gt=0,             description="Annual income")
    person_age                 : int   = Field(..., gt=17, lt=100,    description="Age of applicant")
    person_emp_length          : float = Field(..., ge=0,             description="Employment length in years")
    person_home_ownership      : str   = Field(...,                   description="RENT / OWN / MORTGAGE / OTHER")
    cb_person_default_on_file  : str   = Field(..., pattern="^[YN]$", description="Previous default Y or N")
    cb_person_cred_hist_length : float = Field(..., ge=0,             description="Credit history length in years")

    # Example input
    model_config = {
        "json_schema_extra": {
            "example": {
                "cnic"                      : "42101-1234567-1",
                "loan_amnt"                 : 15000,
                "loan_int_rate"             : 19.4,
                "loan_grade"                : "F",
                "loan_intent"               : "DEBTCONSOLIDATION",
                "loan_percent_income"       : 0.43,
                "person_income"             : 35000,
                "person_age"                : 28,
                "person_emp_length"         : 2.0,
                "person_home_ownership"     : "RENT",
                "cb_person_default_on_file" : "N",
                "cb_person_cred_hist_length": 3.0
            }
        }
    }


# OUTPUT SCHEMAS
class ExplanationItem(BaseModel):
    """Single SHAP explanation item."""
    feature       : str
    readable_name : str
    feature_value : float
    shap_impact   : float
    direction     : str


class ClassificationResult(BaseModel):
    """Classification model output."""
    decision                : str
    default_probability     : float
    threshold               : float
    reasons_for_default     : list[ExplanationItem]
    reasons_against_default : list[ExplanationItem]


class RegressionResult(BaseModel):
    """Regression model output — only for new applicant."""
    predicted_interest_rate : float
    reasons_high_rate       : list[ExplanationItem]
    reasons_low_rate        : list[ExplanationItem]


class ClusteringResult(BaseModel):
    """Clustering model output."""
    cluster_id    : int
    cluster_label : str
    probabilities : dict[str, float]


class PlainReasons(BaseModel):
    """Plain English reasons for decision and rate."""
    decision_reasons : list[str]
    rate_reasons     : list[str]

class PredictionResponse(BaseModel):
    """
    Full API response for both workflows.
    regression is None for existing loan workflow.
    """
    prediction_id  : int
    workflow       : str
    classification : ClassificationResult
    regression     : Optional[RegressionResult] = None
    clustering     : ClusteringResult
    plain_reasons  : Optional[PlainReasons] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status  : str
    version : str
    models  : dict[str, bool]

class StatsResponse(BaseModel):
    """Prediction statistics response."""
    total                   : int
    approved                : int
    rejected                : int
    avg_default_probability : float
    avg_interest_rate       : float


# Get all the Detail about the Applicant by the help of his/her cnic
class InputRecord(BaseModel):
    """Raw inputs stored in DB for a prediction."""
    loan_amnt                 : Optional[float] = None
    loan_int_rate             : Optional[float] = None
    loan_grade                : Optional[str]   = None
    loan_percent_income       : Optional[float] = None
    loan_intent               : Optional[str]   = None
    person_income             : Optional[float] = None
    person_age                : Optional[int]   = None
    person_emp_length         : Optional[float] = None
    person_home_ownership     : Optional[str]   = None
    cb_person_default_on_file : Optional[str]   = None


class ExplanationRecord(BaseModel):
    """Single SHAP explanation stored in DB."""
    task          : Optional[str]   = None
    feature_name  : Optional[str]   = None
    readable_name : Optional[str]   = None
    shap_impact   : Optional[float] = None
    direction     : Optional[str]   = None


class PredictionRecord(BaseModel):
    """Single prediction record with inputs and explanations."""
    prediction_id       : int
    decision            : Optional[str]   = None
    default_probability : Optional[float] = None
    interest_rate       : Optional[float] = None
    cluster_label       : Optional[str]   = None
    inputs              : Optional[InputRecord]         = None
    explanations        : Optional[list[ExplanationRecord]] = None


class ApplicantResponse(BaseModel):
    """Full applicant profile with prediction history."""
    cnic               : str
    first_seen         : Optional[str] = None
    last_seen          : Optional[str] = None
    total_visits       : int
    total_approved     : int
    total_rejected     : int
    last_decision      : Optional[str]   = None
    last_loan_amnt     : Optional[float] = None
    last_interest_rate : Optional[float] = None
    last_visit         : Optional[PredictionRecord] = None
#    history            : list[PredictionRecord]     = []    --> not needed now