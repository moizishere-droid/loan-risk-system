# Feature Engineering Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 3 to 4 March 2026

## 1. Steps Applied

Step	What
1		Load preprocessed splits
2		Create ratio features
3		Create interaction features
4		Create binning features
5		Create domain knowledge features
6		Encode new binned features
7		Scale new numerical features
8		Handle missing values from new features
9		Drop low variance features
10		Drop highly correlated features
11		Feature selection — Filter method
12		Feature selection — Wrapper method (RFE)
13		Feature selection — Embedded method
14		Feature importance check
15		Compare before vs after feature engineering
16		Verify final features
17		Save engineered splits
18		Save feature engineering pipeline
19		Save selected feature list
20		Document report

## 2. Features Created

income_to_loan_ratio
loan_to_age_ratio
income_per_emp_year
loan_grade_x_loan_int_rate # Classification only — loan_int_rate is a feature in classification
loan_grade_x_loan_percent_income
person_age_x_person_emp_length
loan_grade_x_loan_amnt # Regression only — loan_int_rate is the target in regression so cannot be used here
age_group
income_group
loan_size_group
is_high_risk
is_experienced_borrower
debt_burden_score

## 3. Features Dropped

### Low Variance (Step 9)
- is_high_risk — almost all same value
- is_experienced_borrower — almost all same value
- loan_status (reg only) — low variance
### Highly Correlated (Step 10)
- loan_grade (cls) — corr=0.89 with loan_int_rate
- age_group — corr=0.90 with person_age
- income_group — corr=0.90 with person_income
- loan_size_group — corr=0.91 with loan_amnt
### Feature Selection (Steps 11-13)
- Regression only — loan_intent_* columns
  — all 3 methods agreed weak for interest rate prediction


## 4. Feature Selection Summary

Regression — drop loan_intent columns
All 3 methods agree they are weak for predicting interest rate
EDA confirmed no clear pattern

## 5. Final Feature Lists
 
 Clssification = ['person_age',
 'person_income',
 'person_emp_length',
 'loan_amnt',
 'loan_int_rate',
 'loan_percent_income',
 'cb_person_default_on_file',
 'person_home_ownership_MORTGAGE',
 'person_home_ownership_OWN',
 'person_home_ownership_RENT',
 'loan_intent_DEBTCONSOLIDATION',
 'loan_intent_EDUCATION',
 'loan_intent_HOMEIMPROVEMENT',
 'loan_intent_MEDICAL',
 'loan_intent_PERSONAL',
 'loan_intent_VENTURE',
 'income_to_loan_ratio',
 'loan_to_age_ratio',
 'income_per_emp_year',
 'loan_grade_x_loan_int_rate',
 'loan_grade_x_loan_percent_income',
 'person_age_x_person_emp_length',
 'debt_burden_score']

 Regression = ['person_age',
 'person_income',
 'person_emp_length',
 'loan_grade',
 'loan_amnt',
 'loan_percent_income',
 'person_home_ownership_MORTGAGE',
 'person_home_ownership_OWN',
 'person_home_ownership_RENT',
 'income_to_loan_ratio',
 'loan_to_age_ratio',
 'income_per_emp_year',
 'loan_grade_x_loan_amnt',
 'loan_grade_x_loan_percent_income',
 'person_age_x_person_emp_length',
 'debt_burden_score']

## 6. Before vs After Comparison

=== Classification ===
Before FE — Accuracy: 0.9288
After FE  — Accuracy: 0.9296
Before FE — F1: 0.8100
After FE  — F1: 0.8114
=== Regression ===
Before FE — r2_score: 0.9127
After FE  — r2_score: 0.9154
Before FE — mean_squared_error: 0.9797
After FE  — mean_squared_error: 0.9488

## 7. Saved Files

- data/engineered_data/ — all 12 splits
- models/fe_pipeline.joblib
- models/feature_lists.json


## My Notes:
For mine loan risk project specifically:

Classification:
92.88% → 92.96% accuracy
Already very high baseline — hard to improve much
F1 improved 0.14% — more meaningful than accuracy
Verdict: Small but meaningful

Regression:
R2: 0.9127 → 0.9154 — good improvement
MSE: 0.9797 → 0.9488 — 3.15% reduction in error
MSE improvement is more significant than it looks
Verdict: Good improvement

Why improvements are small:
Mine preprocessing was already very clean
loan_grade alone explains 82% of regression variance
Hard to improve on already strong signal

Key takeaway:
FE never hurt the model
FE improved both models
Small improvements at high baseline = good feature engineering
Real value comes in Phase 10 with proper tuning