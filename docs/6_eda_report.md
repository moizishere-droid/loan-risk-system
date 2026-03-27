# EDA Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 28 February 2026
# Dataset: credit_risk_dataset.csv


## 1. Dataset Overview
| Property                | Value                 |
|-------------------------|-----------------------|
| Total Rows              | 27,693 (training set) |
| Total Columns           | 12                    |
| Numerical Features      | 7                     |
| Categorical Features    | 4                     |
| Target (Classification) | loan_status           |
| Target (Regression)     | loan_int_rate         |


## 2. Missing Values Summary
| Column            | Missing Count | Percentage | Status   |
|-------------------|---------------|------------|----------|
| person_emp_length | 778           | 2.81%      | Missing  |
| loan_int_rate     | 2646          | 9.56%      | Missing  |
| All Other Columns | 0             | 0.00%      | Complete |
Total Missing Values: 3424


## 3. Duplicate Rows
| Check            | Result                   |
|------------------|--------------------------|
| Total Duplicates | 165 duplicate rows found |
| Percentage       | 0.51% of training data   |


## 4. Target Distribution

### Classification Target (loan_status)
| Class           | Count  | Percentage |
|-----------------|--------|------------|
| 0 - Non Default | 21,591 | 78.18%     |
| 1 - Default     | 6,021  | 21.82%     |
Status: Imbalanced — ratio 3.58:1

### Regression Target (loan_int_rate)
| Metric       | Value                       |
|--------------|-----------------------------|
| Mean         | 11.02%                      |
| Median       | 10.99%                      |
| Min          | 5.42%                       |
| Max          | 22.48%                      |
| Distribution | Bimodal — 2 borrower groups |
Status: Nearly normal distribution (skew=0.20)


## 5. Numerical Features Summary
| Column                     | Mean   | Std    | Min   | Max       | Skewness | Status        |
|----------------------------|--------|--------|-------|-----------|----------|---------------|
| person_age                 | 27.71  | 6.33   | 20    | 144       | 2.71     | Outliers      |
| person_income              | 65,973 | 62,710 | 4,000 | 6,000,000 | 35.69    | Extreme skew  |
| person_emp_length          | 4.79   | 4.17   | 0     | 123       | 2.84     | Outliers      |
| loan_amnt                  | 9,579  | 6,303  | 500   | 35,000    | 1.19     | Moderate skew |
| loan_int_rate              | 11.02  | 3.24   | 5.42  | 22.48     | 0.20     | Normal        |
| loan_percent_income        | 0.17   | 0.11   | 0.00  | 0.83      | 1.07     | Moderate skew |
| cb_person_cred_hist_length | 5.79   | 4.04   | 2     | 30        | 1.66     | Moderate skew |


## 6. Categorical Features Summary
| Column                    | Unique Values | Most Common       | Default Rate          |
|---------------------------|---------------|-------------------|-----------------------|
| person_home_ownership     | 4             | RENT (50.6%)      | RENT=31%, OWN=7%      |
| loan_intent               | 6             | EDUCATION (19.8%) | DEBTCONSOLIDATION=27% |
| loan_grade                | 7             | A (33.1%)         | A=10%, G=98%          |
| cb_person_default_on_file | 2             | N (82.3%)         | Y=37%, N=18%          |


## 7. Outlier Summary
| Column                     | Outliers | Percentage | Action      |
|----------------------------|----------|------------|-------------|
| person_age                 | 1221     | 4.41%      | IQR Capping |
| person_income              | 1353     | 4.89%      | IQR Capping |
| person_emp_length          | 736      | 2.66%      | IQR Capping |
| loan_amnt                  | 1454     | 5.25%      | IQR Capping |
| loan_int_rate              | 4        | 0.01%      | Keep        |
| loan_percent_income        | 560      | 2.02%      | IQR Capping |
| cb_person_cred_hist_length | 953      | 3.44%      | IQR Capping |
Total Outliers: 6281


## 8. Correlation Summary

### High Correlation Between Features
| Feature 1 | Feature 2                  | Correlation | Action                          |
|-----------|----------------------------|-------------|---------------------------------|
| person_age| cb_person_cred_hist_length | 0.85        | Drop cb_person_cred_hist_length |
| loan_amnt | loan_percent_income        | 0.57        | Keep — not too high             |

### Correlation With Classification Target (loan_status)
| Feature             | Correlation |
|---------------------|-------------|
| loan_percent_income | 0.38        |
| loan_int_rate       | 0.33        |
| person_income       | 0.14        |
| loan_amnt           | 0.11        |
| Others              | < 0.10      |
``
### Correlation With Regression Target (loan_int_rate)
| Feature             | Correlation |
|---------------------|-------------|
| loan_status         | 0.33        |
| loan_amnt           | 0.14        |
| loan_percent_income | 0.11        |
| Others              | < 0.10      |


## 9. Feature Importance Summary

### Classification (Random Forest)
| Rank | Feature               | Importance |
|------|-----------------------|------------|
| 1    | loan_percent_income   | 22.9%      |
| 2    | person_income         | 14.8%      |
| 3    | loan_int_rate         | 11.5%      |
| 4    | loan_grade            | 11.1%      |
| 5    | person_home_ownership | 10.1%      |

### Regression (Random Forest)
| Rank | Feature             | Importance |
|------|---------------------|------------|
| 1    | loan_grade          | 81.9%      |
| 2    | person_income       | 4.1%       |
| 3    | loan_amnt           | 3.0%       |
| 4    | loan_percent_income | 2.4%       |
| 5    | person_age          | 2.3%       |


## 10. Business Insights

### High Risk Applicant Profile
- Home ownership: RENT
- Loan intent: DEBTCONSOLIDATION
- Loan grade: D, E, F, G
- Loan percent of income: > 40%
- Past default on file: Y
- Default rate can reach 98% (grade G)

### Low Risk Applicant Profile
- Home ownership: MORTGAGE or OWN
- Loan grade: A or B
- Loan percent of income: < 15%
- Past default on file: N
- Default rate as low as 7-10%

### Key Business Rules
- Reject all grade G applicants — 98% default rate
- Flag RENT applicants with loan > 40% of income
- Past default history doubles default probability
- Higher interest rate = higher default risk
- loan_grade is the single most powerful predictor for interest rate (82% importance)


## 11. Action Plan for Preprocessing

| Issue              | Finding                                      | Action                          | Phase         |
|--------------------|----------------------------------------------|---------------------------------|---------------|
| Missing values     | person_emp_length: 778, loan_int_rate: 2646  | Median imputation               | Preprocessing |
| Duplicates         | 165 duplicate rows                           | Remove duplicates               | Preprocessing |
| Outliers           | 6281 outliers across 6 columns               | IQR Capping (Winsorization)     | Preprocessing |
| Extreme skewness   | person_income skew=35.69                     | Log transformation              | Preprocessing |
| Invalid age        | person_age max=144                           | Cap at 100                      | Preprocessing |
| Invalid emp_length | person_emp_length max=123                    | Cap at (age-18)                 | Preprocessing |
| High correlation   | person_age & cb_person_cred_hist_length=0.85 | Drop cb_person_cred_hist_length | Feature Engr  |
| Class imbalance    | 78% vs 22%                                   | class_weight='balanced'         | Modeling      |
| Encoding needed    | 4 categorical columns                        | OrdinalEncoder + OneHotEncoder  | Preprocessing |
| Scaling needed     | All numerical columns                        | StandardScaler                  | Preprocessing |