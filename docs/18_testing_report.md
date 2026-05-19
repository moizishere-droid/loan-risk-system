# Testing Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 17 March 2026
# Phase: 18 — Testing

## 1. Overview
Phase 18 tests all API endpoints of the Loan Risk Assessment System end to end. Tests cover both prediction workflows, applicant tracking, delete functionality, and edge cases with invalid inputs. All tests were run against the live FastAPI server with PostgreSQL database connected.

## 2. Test Environment
| Item        | Detail                              |
|-------------|-------------------------------------|
| API URL     | http://localhost:8000               |
| Database    | PostgreSQL — loan_risk_db           |
| Server      | Uvicorn                             |
| Test Method | Python requests in Jupyter notebook |
| Notebook    | notebooks/15_testing.ipynb          |

## 3. Health & Stats

### GET /health
| Field      | Value   |
|------------|---------|
| Status     | 200     |
| API Status | healthy |
| Version    | 1.0.0   |

All models loaded:
| Model           | Status |
|-----------------|--------|
| cls_onnx        | DONE   |
| reg_onnx        | DONE   |
| cluster_onnx    | DONE   |
| cls_pipeline    | DONE   |
| reg_pipeline    | DONE   |
| cls_fe_pipeline | DONE   |
| reg_fe_pipeline | DONE   |
| cls_explainer   | DONE   |
| reg_explainer   | DONE   |
| database        | DONE   |

### GET /stats
| Field                   | Value    |
|-------------------------|----------|
| Status                  | 200      |
| Total Predictions       | 14       |
| Approved                | 7        |
| Rejected                | 7        |
| Avg Default Probability | 0.5315   |
| Avg Interest Rate       | 15.5316% |

## 4. Prediction Tests

### POST /predict/new — REJECTED (Grade F)
| Field               | Value                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------|
| Status              | 200                                                                                       |
| Workflow            | new_applicant                                                                             |
| Decision            | REJECTED                                                                                  |
| Default Probability | 100.0%                                                                                    |
| Predicted Rate      | 18.7842%                                                                                  |
| Cluster             | High Value Borrower                                                                       |
| Top Reason          | Your loan application was REJECTED with a 100.0% estimated default risk (threshold: 60%). |

### POST /predict/new — APPROVED (Grade A)
| Field           | Value                                                                                            |
|-----------------|--------------------------------------------------------------------------------------------------|     
| Status              | 200                                                                                          |
| Workflow            | new_applicant                                                                                |
| Decision            | APPROVED                                                                                     |
| Default Probability | 9.53%                                                                                        |
| Predicted Rate      | 10.9955%                                                                                     |
| Cluster             | Standard Borrower                                                                            |
| Top Reason          | Your loan application was APPROVED with only a 9.5% estimated default risk (threshold: 60%). |

### POST /predict/existing — REJECTED (Grade F, rate provided)
| Field               | Value                                 |
|---------------------|---------------------------------------|
| Status              | 200                                   |
| Workflow            | existing_loan                         |
| Decision            | REJECTED                              |
| Default Probability | 100.0%                                |
| Regression          | Not run — user provided rate of 19.4% |
| Cluster             | High Value Borrower                   |

### POST /predict/existing — APPROVED (Grade A, rate provided)
| Field               | Value                                |
|---------------------|--------------------------------------|
| Status              | 200                                  |
| Workflow            | existing_loan                        |
| Decision            | APPROVED                             |
| Default Probability | 0.99%                                |
| Regression          | Not run — user provided rate of 7.5% |
| Cluster             | Standard Borrower                    |

## 5. Applicant Tracking Test

### GET /applicant/{cnic}
| Field                    | Value                            |
|--------------------------|----------------------------------|
| Status                   | 200                              |
| CNIC                     | 42101-1234567-1                  |
| First Seen               | 2026-03-15 16:57:38              |
| Last Seen                | 2026-03-17 16:23:39              |
| Total Visits             | 13                               |
| Total Approved           | 2                                | 
| Total Rejected           | 11                               |
| Last Decision            | REJECTED                         |
| Last Loan Amount         | $15,000                          |
| Last Rate                | 19.4%                            |
| Last Visit Prediction ID | 30                               |
| Last Visit Inputs        | All 10 fields returned correctly |

### DELETE /applicant/{cnic}
| Step                  | Status | Result                                                         |
|-----------------------|--------|----------------------------------------------------------------|
| Create temp applicant | 200    | Prediction ID 32 created                                       |
| Verify exists         | 200    | Total visits = 1                                               |
| Delete                | 200    | Applicant 99999-0000000-0 and all records deleted successfully |
| Verify gone           | 404    | No applicant found with CNIC: 99999-0000000-0                  |
Cascade delete confirmed — applicant + all predictions + inputs + explanations deleted in one call.

## 6. Edge Cases
| Test                                    | Expected | Got | Result |
|-----------------------------------------|----------|-----|--------|
| Invalid loan grade (Z)                  | 422      | 422 | DONE   |
| Invalid default flag (X)                | 422      | 422 | DONE   |
| Negative income                         | 422      | 422 | DONE   |
| Missing CNIC                            | 422      | 422 | DONE   |
| CNIC not found                          | 404      | 404 | DONE   |
| Missing loan_int_rate for existing loan | 422      | 422 | DONE   |

## 7. Full Results Summary
| Test                            | Expected | Got | Pass |
|---------------------------------|----------|-----|------|
| GET /health                     | 200      | 200 | DONE |
| GET /stats                      | 200      | 200 | DONE |
| POST /predict/new REJECTED      | 200      | 200 | DONE |
| POST /predict/new APPROVED      | 200      | 200 | DONE |
| POST /predict/existing REJECTED | 200      | 200 | DONE |
| POST /predict/existing APPROVED | 200      | 200 | DONE |
| GET /applicant/{cnic}           | 200      | 200 | DONE |
| DELETE /applicant/{cnic}        | 200      | 200 | DONE |
| Invalid loan grade              | 422      | 422 | DONE |
| Invalid default flag            | 422      | 422 | DONE |
| Negative income                 | 422      | 422 | DONE |
| Missing CNIC                    | 422      | 422 | DONE |
| CNIC not found                  | 404      | 404 | DONE |
| Missing loan_int_rate           | 422      | 422 | DONE |

Total : 14
Passed: 14
Failed: 0

## 8. Key Observations
| Observation                   | Detail                                                        |
|-------------------------------|---------------------------------------------------------------|
| Both workflows work correctly | New applicant predicts rate, existing loan uses provided rate |
| Grade F always rejected       | 100% default probability — high risk signals dominate         |
| Grade A always approved       | 0.99% — 9.53% default probability depending on other inputs   |
| Cascade delete works          | One DELETE call removes all linked records                    |
| Pydantic validation works     | All invalid inputs correctly rejected with 422                |
| Applicant tracking works      | Visit counts, last decision, last rate all update correctly   |