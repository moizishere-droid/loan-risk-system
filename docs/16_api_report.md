# API Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 12 March 2026
# Phase: 16 — API

## 1. Overview
Phase 16 builds the API layer of the Loan Risk Assessment System. The notebook covers loading all models and pipelines, building the full preprocessing and inference pipeline, SHAP explanations, plain English reasons, and testing both workflows end to end. The final API src files are written after everything is verified in the notebook.

## 2. What Was Loaded
| Item                       | Detail                                            |
|----------------------------|---------------------------------------------------|
| cls_LightGBM.onnx          | 1.05 MB — classification inference                |
| reg_RandomForest.onnx      | 20.85 MB — regression inference                   |
| cluster_GMM.onnx           | 138 KB  — clustering inference                    |
| cls_pipeline.joblib        | Preprocessing pipeline for classification         |
| reg_pipeline.joblib        | Preprocessing pipeline for regression             |
| cls_fe_pipeline.joblib     | Feature engineering pipeline for classification   |
| reg_fe_pipeline.joblib     | Feature engineering pipeline for regression       |
| X_train_cls_engineered.csv | (27577, 23) — SHAP background for classification  |
| X_train_reg_engineered.csv | (24943, 16) — SHAP background for regression      |
| cls_LightGBM.joblib        | Final model for SHAP TreeExplainer                |
| reg_RandomForest.joblib    | Final model for SHAP TreeExplainer                |

## 3. Constants
| Constant              | Value                                                 |
|-----------------------|-------------------------------------------------------|
| BEST_THRESHOLD_CLS    | 0.6                                                   |
| CLUSTER_LABELS        | {0: High Value Borrower, 1: Standard Borrower}        |
| CLS_PREPROCESSED_COLS | 17 cols — ColumnTransformer output for classification |
| REG_PREPROCESSED_COLS | 17 cols — ColumnTransformer output for regression     |
| CLS_FE_INPUT_COLS     | 17 cols — FE pipeline fit order for classification    |
| REG_FE_INPUT_COLS     | 17 cols — FE pipeline fit order for regression        |

## 4. Preprocessing Pipeline

### Why 17 columns?
After ColumnTransformer processes the raw input it outputs 17 columns:

| Group     | Transformer    | Columns                                                                                     | Count |
|-----------|----------------|---------------------------------------------------------------------------------------------|-------|
| Numerical | StandardScaler | person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_percent_income | 6     |
| Ordinal   | OrdinalEncoder | loan_grade                                                                                  | 1     |
| OneHot    | OneHotEncoder  | person_home_ownership (3 cols)                                                              | 3     |
| OneHot    | OneHotEncoder  | loan_intent (6 cols)                                                                        | 6     |
| Remainder | Passthrough    | cb_person_default_on_file                                                                   | 1     |
| Total     |                |                                                                                             | 17    |

### Why prefixes?
ColumnTransformer automatically adds prefixes to output columns:

| Prefix      | Transformer    | Example                              |
|-------------|----------------|--------------------------------------|
| num__       | StandardScaler | num__person_age                      |
| ohe__       | OneHotEncoder  | ohe__loan_intent_EDUCATION           |
| ord__       | OrdinalEncoder | ord__loan_grade                      |
| remainder__ | Passthrough    | remainder__cb_person_default_on_file |

These must be stripped before passing to the FE pipeline.

### Full transformation flow

Raw input (10-12 fields)
      ↓ ColumnTransformer
17 preprocessed cols (with num__, ohe__, ord__ prefixes)
      ↓ strip prefixes
17 clean cols
      ↓ reorder to match FE pipeline fit order
17 reordered cols
      ↓ FE pipeline
23 engineered cols (cls) or 16 engineered cols (reg)
      ↓ ONNX model
prediction

### Transformation output (verified):
Step 1 — Raw input shape          : (1, 12)
Step 2 — After preprocessing      : (1, 17)
Step 3 — After stripping prefixes : (1, 17)
Step 4 — After reordering         : (1, 17)
Step 5 — After FE pipeline        : (1, 23)  ← cls
                                    (1, 16)  ← reg

## 5. Note on cb_person_cred_hist_length
cb_person_cred_hist_length must be passed to the pipeline because CrossColumnFixer uses it to cap employment length against age. After that fix it is dropped and never reaches the model. It is therefore not saved to the DB inputs table.

| Where     | Include? | Why                          |
|-----------|----------|------------------------------|
| API input | YES      | CrossColumnFixer requires it |
| Pipeline  | YES      | Used once then dropped       |
| DB inputs | NO       | Never reached the model      |

## 6. Note on loan_status
loan_status is the classification target column. At prediction time we always pass loan_status = 0 as a placeholder because the applicant has not defaulted yet. The pipeline passes it through but the FE pipeline and model never use it.

## 7. ONNX Inference Results

### Classification — Grade F sample:
Default probability : 1.0
Threshold           : 0.6
Decision            : REJECTED

### Regression — Grade F sample:
Predicted interest rate : 18.7842%

### Clustering — Grade F sample:
Cluster ID    : 0
Cluster label : High Value Borrower
Probabilities :
   High Value Borrower : 1.0
   Standard Borrower   : 0.0

## 8. SHAP Explanations

### Classification — Top 5 increases default risk:
| Feature                       | SHAP Impact |
|-------------------------------|-------------|
| loan_grade_x_loan_int_rate    | +5.873      |
| loan_percent_income           | +3.910      |
| person_home_ownership_RENT    | +1.347      |
| loan_int_rate                 | +1.162      |
| loan_intent_DEBTCONSOLIDATION | +0.997      |

### Classification — Top 5 decreases default risk:
| Feature                        | SHAP Impact |
|--------------------------------|-------------|
| loan_intent_HOMEIMPROVEMENT    | -0.175      |
| loan_intent_MEDICAL            | -0.054      |
| person_age_x_person_emp_length | -0.045      |
| person_income                  | -0.041      |
| person_age                     | -0.012      |

### Regression — Top 5 increases interest rate:
| Feature                          | SHAP Impact |
|----------------------------------|-------------|
| loan_grade                       | +3.938      |
| loan_grade_x_loan_amnt           | +1.604      |
| loan_grade_x_loan_percent_income | +1.249      |
| debt_burden_score                | +0.361      |
| loan_amnt                        | +0.244      |

### Regression — Top 5 decreases interest rate:
| Feature                        | SHAP Impact |
|--------------------------------|-------------|
| loan_percent_income            | -0.050      |
| person_income                  | -0.047      |
| income_per_emp_year            | -0.046      |
| person_home_ownership_OWN      | -0.000      |
| person_age_x_person_emp_length | +0.000      |

## 9. Full Pipeline Test Results

### REJECTED Case — Grade F
| Field               | Value               |
|---------------------|---------------------|
| Workflow            | new_applicant       |
| Decision            | REJECTED            |
| Default Probability | 100.0%              |
| Predicted Rate      | 18.78%              |
| Cluster             | High Value Borrower |

Plain reasons:
1. Your loan application was REJECTED with a 100.0% estimated default risk (threshold: 60%).
2. Your loan grade F (Bad) is a high-risk grade, which strongly increases your chance of default.
3. Your loan amount ($15,000) is 43.0% of your annual income ($35,000), which is very high and increases default risk.
4. Your interest rate of 18.78% is high, which significantly increases default risk.
5. The combination of grade F and a high interest rate is the strongest signal for default risk.
6. Renting your home is associated with slightly higher default risk compared to homeowners.
7. Borrowing for debt consolidation is associated with slightly higher default rates.
8. Your annual income of $35,000 helps reduce default risk.

### APPROVED Case — Grade A
| Field               | Value             |
|---------------------|-------------------|
| Workflow            | new_applicant     |
| Decision            | APPROVED          |
| Default Probability | 9.53%             |
| Predicted Rate      | 10.99%            |
| Cluster             | Standard Borrower |

Plain reasons:
1. Your loan application was APPROVED with only a 9.5% estimated default risk (threshold: 60%).
2. Your loan grade A (Excellent) is a low-risk grade, which works in your favor.
3. Your loan-to-income ratio of 8.0% is manageable relative to your income of $65,000.
4. Your interest rate of 11.0% has a minor upward effect on default risk.
5. Owning your home or having a mortgage is a positive financial stability signal.
6. Borrowing for education is associated with slightly higher default rates.
7. Your annual income of $65,000 helps reduce default risk.

### Existing Loan — Grade F
| Field               | Value               |
|---------------------|---------------------|
| Workflow            | existing_loan       |
| Decision            | REJECTED            |
| Default Probability | 100.0%              |
| Regression          | None — skipped      |
| Cluster             | High Value Borrower |

## 10. Functions Built
| Function                 | Purpose                                                           |
|--------------------------|-------------------------------------------------------------------|
| run_new_applicant()      | Full 8-step pipeline — reg → cls → cluster → SHAP → plain reasons |
| run_existing_loan()      | Full 5-step pipeline — cls → cluster → SHAP → plain reasons       |
| generate_plain_reasons() | Converts SHAP + original values to human readable text            |

## 11. API Files Built
| File                | Purpose                                                                                     |
|---------------------|---------------------------------------------------------------------------------------------|
| api/schemas.py      | Pydantic input + output schemas + ApplicantResponse                                         |
| api/model_loader.py | Load all models at startup                                                                  |
| api/pipeline.py     | run_new_applicant() + run_existing_loan()                                                   |
| api/routes.py       | 6 endpoints — health, stats, predict new, predict existing, get applicant, delete applicant |
| api/main.py         | FastAPI app entry point with lifespan                                                       |

## 12. Endpoints
| Method | Endpoint          | Purpose                             |
|--------|-------------------|-------------------------------------|
| GET    | /                 | API info                            |
| GET    | /health           | Model + DB status                   |
| GET    | /stats            | Total, approved, rejected, averages |
| POST   | /predict/new      | New applicant workflow              |
| POST   | /predict/existing | Existing loan workflow              |
| GET    | /applicant/{cnic} | Get applicant profile + last visit  |
| DELETE | /applicant/{cnic} | Delete applicant + all records      |

## 13. CNIC Tracking
- Every prediction requires a CNIC
- First visit → new applicant row created
- Return visit → stats updated (total_visits, approved, rejected)
- GET /applicant/{cnic} returns full profile + last visit inputs + SHAP
- DELETE /applicant/{cnic} cascades to all predictions, inputs, explanations