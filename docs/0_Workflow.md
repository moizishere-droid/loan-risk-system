## Use Case No : 01

User opens frontend → selects Tab 1 "New Applicant"

Fills form:
  loan_amnt              = 15000
  loan_grade             = F
  loan_intent            = DEBTCONSOLIDATION
  person_income          = 35000
  person_age             = 28
  person_emp_length      = 2
  person_home_ownership  = RENT
  cb_person_default_on_file = N
  loan_percent_income    = 0.43
  (loan_int_rate = NOT PROVIDED)

        ↓
Frontend sends POST /predict/new to API

        ↓
API Step 1 — Preprocessing
  Clean + encode all 10 raw fields

        ↓
API Step 2 — Feature Engineering (Regression path)
  Create 16 reg features from cleaned data

        ↓
API Step 3 — Regression (reg_RandomForest ONNX)
  Input  : 16 features
  Output : predicted_interest_rate = 19.4%

        ↓
API Step 4 — Feature Engineering (Classification path)
  Inject predicted rate into features
  Create 23 cls features

        ↓
API Step 5 — Classification (cls_LightGBM ONNX)
  Input  : 23 features (including predicted rate)
  Output : probability = 0.85, decision = REJECTED

        ↓
API Step 6 — Clustering (cluster_GMM ONNX)
  Input  : 23 cls features
  Output : cluster_label = High Value Borrower
           probabilities = {High Value: 0.78, Standard: 0.22}

        ↓
API Step 7 — SHAP Explanations
  cls explanations : top 5 features increasing/decreasing default risk
  reg explanations : top 5 features increasing/decreasing rate

        ↓
API Step 8 — Save to DB
  inputs table      : 10 raw fields (loan_int_rate = NULL)
  predictions table : decision=REJECTED, prob=0.85,
                      interest_rate=19.4, cluster=High Value Borrower
  explanations table: cls SHAP rows + reg SHAP rows

        ↓
API returns response to Frontend

        ↓
Frontend shows:
  Predicted Interest Rate : 19.4%
  Decision                : REJECTED
  Default Probability     : 85%
  Borrower Segment        : High Value Borrower
  Why rejected:
    - Very high interest rate       +2.33
    - Debt consolidation intent     +2.08
    - Bad grade × high rate         +1.66
  Why rate is high:
    - Loan grade F                  +3.89
    - Large loan × bad grade        +2.23

## Use Case No: 02

Same person comes back → selects Tab 2 "Existing Loan"

Fills form:
  loan_amnt              = 15000
  loan_grade             = F
  loan_intent            = DEBTCONSOLIDATION
  person_income          = 35000
  person_age             = 28
  person_emp_length      = 2
  person_home_ownership  = RENT
  cb_person_default_on_file = N
  loan_percent_income    = 0.43
  loan_int_rate          = 19.4  ← PROVIDED by user this time

        ↓
Frontend sends POST /predict/existing to API

        ↓
API Step 1 — Preprocessing
  Clean + encode all 10 raw fields including loan_int_rate

        ↓
API Step 2 — Feature Engineering (Classification path)
  Use provided rate directly
  Create 23 cls features

        ↓
API Step 3 — Classification (cls_LightGBM ONNX)
  Input  : 23 features (including user-provided rate)
  Output : probability = 0.85, decision = REJECTED
  (Regression SKIPPED)

        ↓
API Step 4 — Clustering (cluster_GMM ONNX)
  Input  : 23 cls features
  Output : cluster_label = High Value Borrower
           probabilities = {High Value: 0.78, Standard: 0.22}

        ↓
API Step 5 — SHAP Explanations
  cls explanations only (no reg explanations)

        ↓
API Step 6 — Save to DB
  inputs table      : 10 raw fields (loan_int_rate = 19.4)
  predictions table : decision=REJECTED, prob=0.85,
                      interest_rate=NULL, cluster=High Value Borrower
  explanations table: cls SHAP rows only

        ↓
API returns response to Frontend

        ↓
Frontend shows:
  Decision                : REJECTED
  Default Probability     : 85%
  Borrower Segment        : High Value Borrower
  (No rate section — rate was already known)
  Why rejected:
    - Very high interest rate       +2.33
    - Debt consolidation intent     +2.08
    - Bad grade × high rate         +1.66