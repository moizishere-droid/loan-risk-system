# Data Validation Report
# Project: Loan Risk Assessment System
# Date: 26 February 2026
# Dataset: credit_risk_dataset.csv


## 1. Schema Validation
| Check         | Result                        |
|---------------|-------------------------------|
| Total Columns |  12 columns found             |
| Column Names  |  All expected columns present |
| Data Types    |  All dtypes correct           |


## 2. Missing Values
| Column            | Missing Count | Percentage | Status   |
|-------------------|---------------|------------|----------|
| person_emp_length | 895           | 2.75%      |  Missing |
| loan_int_rate     | 3116          | 9.56%      |  Missing |
| All Other Columns | 0             | 0.00%      |  Complete|
Total Missing Values: 4011


## 3. Duplicate Rows
| Check                | Result                  |
|----------------------|-------------------------|
| Total Duplicate Rows | 165 duplicates found    |
| Percentage           | 0.51% of total data     |


## 4. Outlier Detection (IQR Method)
| Column                     | Lower Bound | Upper Bound | Outliers | Percentage |
|----------------------------|-------------|-------------|----------|------------|
| person_age                 | 12.50       | 40.50       | 1494     | 4.59%      |
| person_income              | -22550.00   | 140250.00   | 1484     | 4.55%      |
| person_emp_length          | -5.50       | 14.50       | 853      | 2.62%      |
| loan_amnt                  | -5800.00    | 23000.00    | 1689     | 5.18%      |
| loan_int_rate              | -0.46       | 21.83       | 6        | 0.02%      |
| loan_percent_income        | -0.12       | 0.44        | 651      | 2.00%      |
| cb_person_cred_hist_length | -4.50       | 15.50       | 1142     | 3.51%      |
Total Outliers: 7319


## 5. Value Range Check
| Column                     | Expected Range | Actual Range   | Status       |
|----------------------------|----------------|----------------|--------------|
| person_age                 | 18 - 100       | 20 - 144       | Max too high |
| person_income              | 0 - 1000000    | 4000 - 6000000 | Max too high |
| person_emp_length          | 0 - 60         | 0 - 123        | Max too high |
| loan_amnt                  | 500 - 35000    | 500 - 35000    | Within range |
| loan_int_rate              | 0 - 30         | 5.42 - 23.22   | Within range |
| loan_percent_income        | 0 - 1          | 0 - 0.83       | Within range |
| cb_person_cred_hist_length | 0 - 30         | 2 - 30         | Within range |


## 6. Cross Columns Logic
| Cross Column Logic | 7836 invalid emp_length vs age records | 


## 7. Class Imbalance
| Class           | Count | Percentage |
|-----------------|-------|------------|
| 0 - Non Default | 25473 | 78.18%     |
| 1 - Default     | 7108  | 21.82%     |
Imbalance Ratio: 3.58:1
Status: Imbalanced


## 8. Cardinality Check
| Column                    | Unique Values | Values                                                                    | Status |
|---------------------------|---------------|---------------------------------------------------------------------------|--------|
| person_home_ownership     | 4             | RENT, MORTGAGE, OWN, OTHER                                                | ✅ Low |
| loan_intent               | 6             | EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT | ✅ Low |
| loan_grade                | 7             | A, B, C, D, E, F, G                                                       | ✅ Low |
| cb_person_default_on_file | 2             | Y, N                                                                      | ✅ Low |


## 9. Constant Columns Check
| Check                  | Result                                              |
|------------------------|-----------------------------------------------------|
| Constant Columns Found | None — all columns have more than 1 unique value    |


## 10. Correlation Check
| Feature 1  | Feature 2                  | Correlation | Status           |
|------------|----------------------------|-------------|------------------|
| person_age | cb_person_cred_hist_length | 0.86        | High Correlation |

Note: Older applicants naturally have longer credit history. Consider dropping one in feature engineering phase.


## 11. Data Leakage Check
| Column                     | Correlation with Target | Status |
|----------------------------|-------------------------|--------|
| loan_percent_income        | 0.38                    | Safe   |
| loan_int_rate              | 0.34                    | Safe   |
| person_income              | 0.14                    | Safe   |
| loan_amnt                  | 0.11                    | Safe   |
| person_emp_length          | 0.08                    | Safe   |
| person_age                 | 0.02                    | Safe   |
| cb_person_cred_hist_length | 0.02                    | Safe   |

No data leakage detected.


## 12. Summary & Action Plan

| Issue            | Finding                                                             | Action                           | Phase |
|------------------|------------------------------------------------------------------- -|----------------------------------|-------|
| Missing Values   | person_emp_length: 895, loan_int_rate: 3116                         | Median imputation                | Preprocessing |
| Duplicates       | 165 duplicate rows                                                  | Remove duplicates                | Preprocessing |
| Outliers         | 7319 outliers across 7 columns                                      | IQR Capping (Winsorization)      | Preprocessing |
| Value Range      | person_age max=144, person_emp_length max=123, person_income max=6M | Cap at realistic values          | Preprocessing |
| Class Imbalance  | 78% vs 22%                                                          | class_weight='balanced' in model | Modeling      |
| High Correlation | person_age & cb_person_cred_hist_length (0.86)                      | Consider dropping one            | Feature engr  |
| Data Leakage     | None detected                                                       | No action needed                 | —             |
| Cardinality      | All low cardinality                                                 | One-Hot & Ordinal Encoding       | Preprocessing |


| Cross Col Logic |
Rule           | Action                                                          |   Phase             
Rule 1 — 8731  | rowsCap emp_length to (age - 18)                                |  Preprocessing
Rule 2 — 781   | rowsCap cred_hist_length to (age - 18)                          |   Preprocessing
Rule 4 — 388   | rowsRecalculate loan_percent_income as loan_amnt/person_income  |   Preprocessing