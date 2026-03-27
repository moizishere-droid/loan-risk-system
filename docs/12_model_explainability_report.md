# Model Explainability Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: March 2026
# Phase: 12 — Model Explainability

## 1. Objective
Explain the predictions of the best trained models for classification, regression, and clustering tasks using SHAP values, feature importance plots, calibration curves, and cluster radar charts. The goal is to ensure model decisions are interpretable and trustworthy for production deployment.

## 2. Models Explained
| Task           | Model Explained | Reason                        |
|----------------|-----------------|-------------------------------|
| Classification | LightGBM        | Best F1=0.84, AUC=0.9414      |
| Regression     | RandomForest    | Best R2=0.9077, RMSE=0.9811   |
| Clustering     | GMM + KMeans    | Both explained for comparison |

## 3. Classification — Feature Importance
Models: LightGBM, RandomForest, XGBoost
| Rank | LightGBM                   | RandomForest                     | XGBoost                       |
|------|----------------------------|----------------------------------|-------------------------------|
| 1    | person_income              | loan_percent_income              | person_home_ownership_RENT    |
| 2    | loan_int_rate              | loan_int_rate                    | person_home_ownership_OWN     |
| 3    | loan_grade_x_loan_int_rate | person_income                    | cb_person_default_on_file     |
| 4    | loan_percent_income        | loan_grade_x_loan_int_rate       | loan_intent_DEBTCONSOLIDATION |
| 5    | loan_to_age_ratio          | loan_grade_x_loan_percent_income | loan_intent_HOMEIMPROVEMENT   |

Key Insights:
- loan_percent_income is most consistent feature across all models
- loan_int_rate appears in top 3 across LightGBM and RandomForest — this is expected because loan_int_rate is an input feature in classification (the bank has already assigned the rate). A higher assigned rate signals the bank already assessed this borrower as risky, making it a strong predictor of default.
- Engineered features (loan_grade_x_loan_int_rate, debt_burden_score) appear in all models confirming feature engineering was effective
- LightGBM uses count-based importance — values appear as 500-1500 vs normalized 0.0-0.15 for others

## 4. Classification — SHAP Summary
Model:* LightGBM | Sample: 1000 training samples

### SHAP Feature Importance (Bar):
| Rank | Feature                    | Mean SHAP |
|------|----------------------------|-----------|
| 1    | loan_percent_income        | 1.10      |
| 2    | loan_int_rate              | 0.98      |
| 3    | person_home_ownership_OWN  | 0.95      |
| 4    | person_income              | 0.88      |
| 5    | loan_intent_VENTURE        | 0.52      |
| 6    | loan_grade_x_loan_int_rate | 0.50      |

### SHAP Beeswarm — Direction Analysis:
| Feature                    | High Value Effect      | Insight                                   |
|----------------------------|------------------------|-------------------------------------------|
| loan_percent_income        | Increases default risk | High loan burden = more likely to default |
| loan_int_rate              | Increases default risk | High assigned rate = more likely to default |
| person_home_ownership_OWN  | Decreases default risk | Owning home = safer borrower              |
| person_home_ownership_RENT | Increases default risk | Renting = higher default risk             |
| loan_intent_VENTURE        | Decreases default risk | Venture loans slightly safer              |

Key Insights:
- loan_percent_income is the single most impactful feature — engineered feature proving its value
- Home ownership strongly protective — OWN reduces risk, RENT increases risk
- Engineered feature loan_grade_x_loan_int_rate confirms interaction between grade and rate

## 5. Classification — SHAP Waterfall
Base rate (E[f(X)]): -2.111

### Non-Default Sample (prob=0.0567)
| Feature                   | Value  | SHAP  | Effect                      |
|---------------------------|--------|-------|-----------------------------|
| person_income             | 0.875  | -0.89 | Reduces default risk        |
| loan_percent_income       | -0.552 | -0.57 | Low loan burden — safe      |
| loan_int_rate             | 0.31   | +0.60 | Moderate rate — slight risk |
| person_home_ownership_OWN | 0      | +0.51 | Not owning — slight risk    |
Conclusion: High income + low loan burden drives safely non-default prediction

### Default Sample (prob=0.9802)
| Feature                       | Value | SHAP  | Effect                            | 
|-------------------------------|-------|-------|-----------------------------------|
| loan_int_rate                 | 1.403 | +2.33 | Very high assigned rate — huge risk        |
| loan_intent_DEBTCONSOLIDATION | 1     | +2.08 | Debt consolidation = risky intent |
| loan_grade_x_loan_int_rate    | 1.012 | +1.66 | Engineered feature firing         |
| person_home_ownership_OWN     | 0     | +0.61 | Not owning home — risky           |
Conclusion: Very high assigned interest rate + debt consolidation intent = strong default signal

### Borderline Sample (prob=0.6051)
| Feature                       | Value  | SHAP  | Effect                            |
|-------------------------------|--------|-------|-----------------------------------|
| person_income                 | -0.263 | +1.41 | Low income pushing toward default |
| loan_int_rate                 | 0.154  | -0.30 | Low rate reducing risk            |
| loan_percent_income           | -0.784 | -0.28 | Low burden reducing risk          |
| loan_intent_DEBTCONSOLIDATION | 1      | +0.31 | Risky intent pushing up           |
Conclusion: Mixed signals — low income vs low burden = borderline prediction at 0.6051

## 6. Classification — Calibration Curve
| Model        | Calibration Quality | Pattern                                   |
|--------------|---------------------|-------------------------------------------| 
| XGBoost      | Best                | Closest to diagonal                       |
| LightGBM     | Good                | Slightly underconfident at 0.5-0.8 range  |
| RandomForest | Worst               | Highly erratic — unreliable probabilities |

Probability Trustworthiness:
| Model        | Trust Probabilities?            |
|--------------|---------------------------------|
| XGBoost      | Yes — well calibrated           |
| LightGBM     | Mostly — slight underconfidence |
| RandomForest | No — erratic                    |

Key Insight: Despite LightGBM having best F1 and AUC, XGBoost has more trustworthy probabilities. This is important for loan risk decisions where probability magnitude matters.

## 7. Regression — Feature Importance
Models: LightGBM, RandomForest, XGBoost
| Rank | LightGBM                       | RandomForest                     | XGBoost                          |
|------|--------------------------------|----------------------------------|----------------------------------|
| 1    | income_per_emp_year            | loan_grade                       | loan_grade                       |
| 2    | loan_to_age_ratio              | loan_grade_x_loan_amnt           | loan_grade_x_loan_amnt           |
| 3    | loan_percent_income            | loan_grade_x_loan_percent_income | loan_amnt                        |
| 4    | loan_amnt                      | loan_amnt                        | loan_grade_x_loan_percent_income |
| 5    | person_age_x_person_emp_length | loan_percent_income              | loan_percent_income              |

Key Insights:
- loan_grade dominates RandomForest (0.60) and XGBoost (0.85) — overwhelmingly dominant
- LightGBM spreads importance more evenly across features
- Engineered features loan_grade_x_loan_amnt and loan_grade_x_loan_percent_income appear in all 3 models
- loan_grade directly determines interest rate in banking — model correctly learned this real-world relationship

## 8. Regression — SHAP Summary
Model: RandomForest | Sample: 1000 training samples

### SHAP Feature Importance (Bar):
| Rank | Feature                          | Mean SHAP |
|------|----------------------------------|-----------|
| 1    | loan_grade                       | 1.80      |
| 2    | loan_grade_x_loan_amnt           | 0.30      |
| 3    | loan_grade_x_loan_percent_income | 0.28      |
| 4    | loan_amnt                        | 0.15      |
| 5    | loan_percent_income              | 0.12      |

### SHAP Beeswarm — Direction Analysis:
| Feature | High Value Effect | Insight |
|---------|------------------|---------|
| loan_grade | Increases interest rate | Higher grade = higher rate |
| loan_grade_x_loan_amnt | Increases interest rate | Bad grade + large loan = very high rate |
| loan_grade_x_loan_percent_income | Increases interest rate | Engineered interaction confirmed |

Key Insights:
- loan_grade SHAP = 1.80 vs next feature = 0.30 — 6x more important than any other feature
- Explains vertical clusters seen in residual plots in Phase 11
- Engineered features rank 2nd and 3rd confirming feature engineering value

## 9. Regression — SHAP Waterfall
Base rate (E[f(X)]): 11.008
### Low Rate Sample (pred=6.6467)
| Feature                        | Value  | SHAP  | Effect                   |
|--------------------------------|--------|-------|--------------------------|
| loan_grade                     | -1.043 | -2.75 | Best grade = lowest rate |
| loan_amnt                      | -1.254 | -0.20 | Small loan               |
| loan_percent_income            | -1.066 | -0.16 | Low burden               |
| person_home_ownership_MORTGAGE | 1      | -0.12 | Mortgage = safer         |

Conclusion: Best grade borrower — all features pushing rate down

### High Rate Sample (pred=19.4053)
| Feature | Value | SHAP | Effect |
|---------|-------|------|--------|
| loan_grade | 4.098 | +3.89 | Worst grade = highest rate |
| loan_grade_x_loan_amnt | 8.564 | +2.23 | Bad grade + large loan |
| loan_grade_x_loan_percent_income | 1.517 | +0.75 | Engineered feature firing |
| debt_burden_score | 0.023 | +0.66 | High debt burden |

Conclusion: Worst grade + large loan + high debt = maximum interest rate

### Mid Rate Sample (pred=11.1010)
| Feature                | Value  | SHAP  | Effect           |
|------------------------|--------|-------|------------------|
| loan_grade             | -0.186 | +0.40 | Slight push up   |
| loan_grade_x_loan_amnt | -0.121 | -0.21 | Slight push down |
| debt_burden_score      | 0.007  | -0.10 | Low debt burden  |

Conclusion: Near average grade — forces nearly cancel out = average rate

## 10. Cluster Radar Chart
Features: loan_percent_income, person_income, loan_amnt, loan_int_rate, person_age, person_emp_length

### GMM Cluster Profiles:
| Feature             | Cluster 0 — High Value (660) | Cluster 1 — Standard (1784)  |
|---------------------|------------------------------|------------------------------|
| person_income       | High                         | Low                          |
| loan_percent_income | High                         | Low                          |
| loan_amnt           | Very High                    | Low                          |
| loan_int_rate       | High                         | Low                          |
| person_age          | Very High                    | Low                          |
| person_emp_length   | High                         | Low                          |

### KMeans Cluster Profiles:
| Feature             | Cluster 0 — Standard (1617) | Cluster 1 — High Value (827) |
|---------------------|------------------------------|-------------------------------|
| person_income       | Low                          | Very High                     |
| loan_percent_income | Low                          | High                          |
| loan_amnt           | Low                          | Very High                     |
| loan_int_rate       | Low                          | High                          |
| person_age          | Low                          | High                          |
| person_emp_length   | Low                          | High                          |

Key Insights:
- Both models consistently identify the same 2 borrower groups
- Perfect separation visible in radar chart — no overlap
- person_age is the strongest visual separator between clusters
- High Value Borrowers = older, higher income, larger loans, higher rates
- Standard Borrowers = younger, lower income, smaller loans, lower rates

## 11. Key Insights Summary
### Classification
| Insight                  | Detail                                              |
|--------------------------|-----------------------------------------------------|
| Most important feature   | loan_percent_income (SHAP=1.10) — engineered feature|
| Strongest default signal | High loan_int_rate + debt consolidation intent      |
| Strongest safety signal  | person_home_ownership_OWN                           |
| Best calibrated model    | XGBoost — most trustworthy probabilities            |
| Best overall model       | LightGBM — F1=0.84, AUC=0.9414                      |

### Regression
| Insight                  | Detail                                             |
|--------------------------|----------------------------------------------------|
| Most important feature   | loan_grade (SHAP=1.80) — 6x more than next feature |
| Strongest rate predictor | loan_grade directly maps to interest rate          |
| Engineered features      | loan_grade_x_loan_amnt ranked 2nd across models    |
| Best model               | RandomForest — R2=0.9077, RMSE=0.9811              |

### Clustering
| Insight           | Detail                                                    |
|-------------------|-----------------------------------------------------------|
| Both models agree | Same 2 groups found consistently                          |
| Group 1           | High Value Borrowers — older, higher income, larger loans |
| Group 2           | Standard Borrowers — younger, lower income, smaller loans |
| Best model        | GMM — Silhouette=0.2287                                   |

## 12. Saved Files
| File                                                                         | Description                  |
|------------------------------------------------------------------------------|------------------------------|
| models/Project_Parameter_Files/shap_values/shap_values_cls.npy               | SHAP values — classification |
| models/Project_Parameter_Files/shap_values/shap_values_reg.npy               | SHAP values — regression     |
| models/Project_Parameter_Files/shap_values/cluster_labels_cluster_GMM.npy    | GMM cluster labels           |
| models/Project_Parameter_Files/shap_values/cluster_labels_cluster_KMeans.npy | KMeans cluster labels        |