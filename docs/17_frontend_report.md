q`# Frontend Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 17 March 2026
# Phase: 17 — Streamlit Frontend


## 1. Overview
Phase 17 builds the frontend for the Loan Risk Assessment System using Streamlit. The UI is intentionally simple and functional — the goal is to demonstrate the full system end to end, not to build a polished product. The frontend communicates with the FastAPI backend built in Phase 16.


## 2. File Structure
| File             | Purpose                        |
|------------------|--------------------------------|
| frontend/app.py  | Single file — full application |


## 3. How to Run
Terminal 1 — Start API:
uvicorn api.main:app --reload

Terminal 2 — Start Frontend:
cd frontend
streamlit run app.py

Open http://localhost:8501 in browser.


## 4. Tabs

### 4.1 Tab 1 — New Applicant
- Input form — CNIC, personal info, loan details, credit info
- No interest rate field — system predicts it
- Calls POST /predict/new
- Shows full results on submit

### 4.2 Tab 2 — Existing Loan
- Same input form + interest rate field
- Calls POST /predict/existing
- Regression section skipped in results

### 4.3 Tab 3 — Applicant Profile
- Enter CNIC → calls GET /applicant/{cnic}
- Shows applicant summary — total visits, approved, rejected, last decision
- Shows last visit — decision, probability, inputs submitted, SHAP explanations

### 4.4 Tab 4 — Delete Applicant
- Enter CNIC → calls DELETE /applicant/{cnic}
- Warning shown before delete
- Cascade deletes all predictions, inputs, explanations

## 5. Results Display
After a prediction the following is shown:

| Section          | What                                             |
|------------------|--------------------------------------------------|
| Decision banner  | APPROVED (green) or REJECTED (red) with risk %   |
| Metrics          | Default probability, threshold, predicted rate   |
| Borrower segment | High Value Borrower or Standard Borrower         |
| Decision reasons | Plain English numbered list                      |
| Rate reasons     | Plain English numbered list (new applicant only) |
| SHAP table       | Feature, SHAP impact, direction — classification |
| SHAP table       | Feature, SHAP impact, direction — regression     |

## 6. Input Form Fields
| Field                     | Type    | Tab 1 | Tab 2 |
|---------------------------|---------|-------|-------|
| CNIC                      | Text    | YES   | YES   |
| Age                       | Integer | YES   | YES   |
| Annual Income ($)         | Float   | YES   | YES   |
| Employment Length (years) | Float   | YES   | YES   |
| Home Ownership            | Select  | YES   | YES   |
| Loan Amount ($)           | Float   | YES   | YES   |
| Loan Grade                | Select  | YES   | YES   |
| Loan Purpose              | Select  | YES   | YES   |
| Loan % of Income          | Float   | YES   | YES   |
| Previous Default          | Select  | YES   | YES   |
| Credit History (years)    | Float   | YES   | YES   |
| Interest Rate (%)         | Float   | NO    | YES   |

## 7. API Endpoints Used
| Method | Endpoint          | Used In |
|--------|-------------------|---------|
| POST   | /predict/new      | Tab 1   |
| POST   | /predict/existing | Tab 2   |
| GET    | /applicant/{cnic} | Tab 3   |
| DELETE | /applicant/{cnic} | Tab 4   |

## 8. Design Decisions
| Decision                       | Reason                                                    |
|--------------------------------|-----------------------------------------------------------|
| Single file app.py             | Simple project — no need for multi-file frontend          |
| No custom CSS                  | Default Streamlit is clean enough for a portfolio project |
| No Plotly charts               | Tables are simpler and easier to explain                  |
| Reusable input_form() function | Avoids duplicate code for Tab 1 and Tab 2                 |
| key_prefix on all widgets      | Prevents Streamlit duplicate widget key errors            |
| Warning before delete          | Prevents accidental data loss                             |