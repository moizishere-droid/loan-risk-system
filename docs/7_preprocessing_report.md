# Preprocessing Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 1 March 2026 to 2 March 2026

## 1. Steps Applied
1. Load splits
2. Remove duplicates
3. Fix cross column logic
4. Handle missing values
5. IQR Capping
6. Log transformation
7. Ordinal encoding
8. One-Hot encoding
9. Map encoding
10. Feature scaling
11. Drop Rare column
12. Drop Redundant feature
13. Data Type Optimization
14. Verify preprocessed data
15. Save preprocessed splits
16. Save preprocessing pipeline
17. Document report


## 2. Duplicates Removed
### CLassification:
Shape of X_train_cls_dup before removing duplicates: (27693, 11)
Shape of y_train_cls_dup before removing duplicates: (27693, 1)
Shape of X_train_cls_dup after removing duplicates: (27577, 11)
Shape of y_train_cls_dup after removing duplicates: (27577, 1)
Number of duplicate rows removed: 116
### Regression:
Shape of X_train_reg_dup before removing duplicates: (25045, 11)
Shape of y_train_reg_dup before removing duplicates: (25045, 1)
Shape of X_train_reg_dup after removing duplicates: (24943, 11)
Shape of y_train_reg_dup after removing duplicates: (24943, 1)
Number of duplicate rows removed: 102


## 3. Cross Column Fixes
Applied Rule 1: Capped emp_length to (age - 18)
Applied Rule 2: Capped cred_hist_length to (age - 18)
Applied Rule 3: Recalculated loan_percent_income
Finished fixing X_train_cls_fixed 

Applied Rule 1: Capped emp_length to (age - 18)
Applied Rule 2: Capped cred_hist_length to (age - 18)
Applied Rule 3: Recalculated loan_percent_income
Finished fixing X_test_cls_fixed 

Applied Rule 1: Capped emp_length to (age - 18)
Applied Rule 2: Capped cred_hist_length to (age - 18)
Applied Rule 3: Recalculated loan_percent_income
Finished fixing X_val_cls_fixed 

Applied Rule 1: Capped emp_length to (age - 18)
Applied Rule 2: Capped cred_hist_length to (age - 18)
Applied Rule 3: Recalculated loan_percent_income
Finished fixing X_train_reg_fixed 

Applied Rule 1: Capped emp_length to (age - 18)
Applied Rule 2: Capped cred_hist_length to (age - 18)
Applied Rule 3: Recalculated loan_percent_income
Finished fixing X_test_reg_fixed 

Applied Rule 1: Capped emp_length to (age - 18)
Applied Rule 2: Capped cred_hist_length to (age - 18)
Applied Rule 3: Recalculated loan_percent_income
Finished fixing X_val_reg_fixed


## 4. Missing Values Handled
Cls after imputation:  0
Reg after imputation:  0


## 5. Outliers Capped
Before capping:
person_age max: 144
person_income max: 6000000

After capping:
person_age max: 40.5
person_income max: 139750

person_age range: 20.0 - 40.5
person_income range: 4000.0 - 139750.0
person_emp_length range: 0.0 - 12.0
cb_person_cred_hist_length range: 2.0 - 15.5
loan_percent_income range: 0.0008 - 0.4368

## 6. Log Transformation
skipped this step cause capping handle the skewness

Classification:
 person_age                    1.015975
person_income                 0.867167
person_emp_length             0.790442
loan_amnt                     0.785433
loan_int_rate                 0.199678
loan_percent_income           0.801530
cb_person_cred_hist_length    1.127058
dtype: float64

Regression:
 person_age                    1.019966
person_income                 0.863614
person_emp_length             0.779735
loan_amnt                     0.811065
loan_status                   0.000000
loan_percent_income           0.812467
cb_person_cred_hist_length    1.121246
dtype: float64


## 7. Encoding Summary
| Column                    | Method           | Result    |
|---------------------------|------------------|-----------|
| loan_grade                | Ordinal Encoding | 0-6       |
| person_home_ownership     | One-Hot Encoding | 4 columns |
| loan_intent               | One-Hot Encoding | 6 columns |
| cb_person_default_on_file | Map (Y/N)        | 0 or 1    |

## 8. Scaling Summary
Before scaling:
person_income mean: 62344.0862
person_income std:  31723.6131

After scaling:
person_income mean: -0.0000
person_income std:  1.0000

## 9. Drop Rare and Redundant feature and Dtype Optimization
Drop person_home_ownership_OTHER column
 Rare category column dropped
New shape: (27577, 18)

Drop cb_person_cred_hist_length from all splits
It is highly correlated with person_age (0.85)
 Redundant feature dropped
New shape: (27577, 17)

Dataset 1
Before: 3.79 MB
After:  2.00 MB
Memory saved: 47.22%

Dataset 2
Before: 0.32 MB
After:  0.16 MB
Memory saved: 49.98%

Dataset 3
Before: 0.32 MB
After:  0.16 MB
Memory saved: 49.98%

Dataset 4
Before: 3.43 MB
After:  1.81 MB
Memory saved: 47.22%

Dataset 5
Before: 0.29 MB
After:  0.14 MB
Memory saved: 49.98%

Dataset 6
Before: 0.29 MB
After:  0.14 MB
Memory saved: 49.98%

## 9. Final Dataset Shape
cls = Final Shape : (27577, 17)
reg = Final Shape : (24943, 17)

**Note:** Classification and regression have different row counts because regression rows with missing loan_int_rate were dropped during splitting — loan_int_rate is the regression target. Both tasks share the same 17 features after preprocessing — loan_int_rate is present as a feature in classification and absent as a column in regression since it is the target.

## 10. Saved Files
data/processed_data/X_train_cls_processed.csv
data/processed_data/X_test_cls_processed.csv
data/processed_data/X_val_cls_processed.csv
data/processed_data/y_train_cls_processed.csv
data/processed_data/y_test_cls_processed.csv
data/processed_data/y_val_cls_processed.csv
data/processed_data/X_train_reg_processed.csv
data/processed_data/X_test_reg_processed.csv
data/processed_data/X_val_reg_processed.csv
data/processed_data/y_train_reg_processed.csv
data/processed_data/y_test_reg_processed.csv
data/processed_data/y_val_reg_processed.csv

## 11. Pipeline Saved
models\preprocessing_pipeline.joblib