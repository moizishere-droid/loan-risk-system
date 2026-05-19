# Data Splitting Report

## Split Ratios
Train: 70.0%
Val:   15.0%
Test:  15.0%

## Random State
Random State: 42

## Classification Splits
Classification Splits:
X_train_cls: (27693, 11)
X_val_cls:   (2444, 11)
X_test_cls:  (2444, 11)

## Regression Splits
Regression Splits:
X_train_reg: (25045, 11)
X_val_reg:   (2210, 11)
X_test_reg:  (2210, 11)

## Note on Split Differences
Classification and regression splits differ in two ways:
- **Row count:** Regression has fewer rows because rows with missing loan_int_rate values are dropped — loan_int_rate is the regression target and cannot be missing.
- **Feature set:** Classification includes loan_int_rate as an input feature. Regression excludes it because it is the target column.

## Clustering Note
Clustering will use X_train_cls for training
Clustering dataset shape: (27693, 11)

## Class Distribution Verification
| Dataset  | Class 0 | Class 1 |
|----------|---------|---------|
| Original | 78.18%  | 21.82%  |
| Train    | 78.18%  | 21.82%  |
| Val      | 78.19%  | 21.81%  |
| Test     | 78.19%  | 21.81%  |

## Overlap Check Results
Overlap between Train and Val: 0 samples
Overlap between Train and Test: 0 samples
Overlap between Val and Test: 0 samples

## Saved Files
| File            | Description                   |
|-----------------|-------------------------------|
| X_train_cls.csv | Classification train features |
| y_train_cls.csv | Classification train target   |
| X_val_cls.csv   | Classification val features   |
| y_val_cls.csv   | Classification val target     |
| X_test_cls.csv  | Classification test features  |
| y_test_cls.csv  | Classification test target    |
| X_train_reg.csv | Regression train features     |
| y_train_reg.csv | Regression train target       |
| X_val_reg.csv   | Regression val features       |
| y_val_reg.csv   | Regression val target         |
| X_test_reg.csv  | Regression test features      |
| y_test_reg.csv  | Regression test target        |