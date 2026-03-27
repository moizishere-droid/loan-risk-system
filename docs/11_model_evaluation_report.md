# Model Evaluation Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 6 to 7 March 2026
# Phase: 11 — Model Evaluation

## 1. Objective
Perform deep evaluation of all trained models for classification, regression, and clustering tasks using classification reports, confusion matrices, ROC curves, residual plots, and cluster visualizations. All models evaluated on both validation and test sets to ensure generalization before Phase 13 — Model Selection.

## 2. Data Used
| Task           | Val Dataset              | Test Dataset              | Shape      |
|----------------|--------------------------|---------------------------|------------|
| Classification | X_val_cls_engineered.csv | X_test_cls_engineered.csv | (2444, 23) |
| Regression     | X_val_reg_engineered.csv | X_test_reg_engineered.csv | (2210, 16) |
| Clustering     | X_val_cls_engineered.csv | X_test_cls_engineered.csv | (2444, 23) |

**Note:** Classification and clustering use 23 features including loan_int_rate as an input feature. Regression uses 16 features — loan_int_rate is the target being predicted and is excluded from inputs. Each task is evaluated independently.

## 3. Threshold Tuning
Model: LightGBM (Best Classification Model)
Default threshold: 0.5
Method: Test thresholds from 0.2 to 0.8, select best F1

| Threshold | F1     | Precision | Recall |
|-----------|--------|-----------|--------|
| 0.8       | 0.7942 | 0.9916    | 0.6623 |
| 0.7       | 0.8229 | 0.9695    | 0.7148 |
| 0.6       | 0.8245 | 0.9233    | 0.7448 |
| 0.5       | 0.8173 | 0.9083    | 0.7430 |
| 0.4       | 0.7962 | 0.8008    | 0.7917 |
| 0.3       | 0.7741 | 0.7332    | 0.8199 |
| 0.2       | 0.7201 | 0.6236    | 0.8518 |
Decision: Best threshold = 0.6 
- Highest F1 = 0.8245
- High precision = 0.923 — few false alarms
- Good recall = 0.745 — catches most defaulters
- Applied to all classification models for remaining evaluation

## 4. Classification Report — Validation Set
Threshold = 0.6

| Model              | Class       | Precision | Recall | F1   | Support |
|--------------|-------------|-----------|--------|------|---------|
| LightGBM           | Non-Default | 0.93      | 0.98   | 0.96 | 1911    |
|                    | Default     | 0.92      | 0.74   | 0.82 | 533     |
|                    | Accuracy    |           |        | 0.93 | 2444    |
| XGBoost            | Non-Default | 0.93      | 0.99   | 0.96 | 1911    |
|                    | Default     | 0.96      | 0.72   | 0.82 | 533     |
|                    | Accuracy*   |           |        | 0.93 | 2444    |
| RandomForest       | Non-Default | 0.91      | 1.00   | 0.95 | 1911    |
|                    | Default     | 0.98      | 0.65   | 0.78 | 533     |
|                    | Accuracy    |           |        | 0.92 | 2444    |
| LogisticRegression | Non-Default | 0.91      | 0.88   | 0.90 | 1911    |
|                    | Default     | 0.62      | 0.70   | 0.66 | 533     |
|                    | Accuracy    |           |        | 0.84 | 2444    |

## 5. Confusion Matrix Analysis
| Model              | True Neg | False Pos | False Neg | True Pos |
|--------------------|----------|-----------|-----------|----------|
| LightGBM           | 1878     | 33        | 136       | 397      |
| XGBoost            | 1894     | 17        | 151       | 382      |
| RandomForest       | 1905     | 6         | 185       | 348      |
| LogisticRegression | 1685     | 226       | 162       | 371      |

Key Insight:
- False Negatives = missed defaulters = most costly for loan risk
- LightGBM fewest missed defaulters (136)
- RandomForest fewest false alarms (6) but misses most defaulters (185)
- LogisticRegression worst — 226 false alarms

## 6. ROC Curve Analysis
| Model              | AUC    |
|--------------------|--------|
| LightGBM           | 0.9414 |
| XGBoost            | 0.9399 |
| RandomForest       | 0.9259 |
| LogisticRegression | 0.8630 |

Key Insight:
- LightGBM highest AUC = 0.9414
- LightGBM and XGBoost curves nearly identical — only 0.0015 difference
- RandomForest drops at low FPR — less reliable at strict thresholds
- LogisticRegression clearly separated from ensemble models

## 7. Regression Residual Analysis
| Model            | Pattern                    | Bias          | Error Range |
|------------------|----------------------------|---------------|-------------|
| RandomForest     | Vertical clusters          | Minimal       | -2 to +4    |
| XGBoost          | Vertical clusters          | Slight        | -3 to +3    |
| LightGBM         | Vertical clusters          | Slight        | -3 to +3    |
| LinearRegression | Vertical clusters + spread | Clear bias    | -6 to +3    |

Key Insight:
- Vertical clusters expected — loan_grade is discrete (6 values)
- RandomForest residuals tightest around 0
- LinearRegression largest error spread — up to -6
- No funnel shape — no heteroscedasticity

## 8. Actual vs Predicted Analysis
| Model            | Fit to Line | Spread |
|------------------|-------------|--------|
| RandomForest     | Tightest    | Low    |
| XGBoost          | Good        | Low    |
| LightGBM         | Good        | Low    | 
| LinearRegression | Moderate    | High   |

Key Insight:
- All ensemble models close to perfect prediction line
- All models slightly underpredict at high interest rates — edge case behavior
- LinearRegression most scattered at extremes
- Discrete clusters visible due to loan_grade dominance — expected

## 9. Cluster Visualization — PCA 2D
| Info                   | Value |
|------------------------|-------|
| PC1 Explained Variance | 14.2% |
| PC2 Explained Variance | 12.9% |
| Total Explained        | 27.1% |

Key Insight:
- Only 27.1% variance captured in 2D — expected for 23 feature dataset
- KMeans shows cleaner separation — hard boundaries
- GMM shows overlapping clusters — soft boundaries expected
- Both models consistently identify same 2 groups

## 10. Cluster Profiles

### GMM Cluster Profiles
| Feature             | Cluster 0 (660) | Cluster 1 (1784) | Interpretation            |
|---------------------|-----------------|------------------|---------------------------|
| loan_percent_income | +0.385          | -0.176           | Cluster 0 = higher burden |
| person_income       | +0.357          | -0.109           | Cluster 0 = higher income |
| loan_amnt           | +0.592          | -0.232           | Cluster 0 = larger loans  |
| loan_int_rate       | +0.413          | -0.184           | Cluster 0 = higher rate   |
| person_age          | +0.618          | -0.209           | Cluster 0 = older         |
- Cluster 0 = High Value Borrowers — older, higher income, larger loans
- Cluster 1 = Standard Borrowers — younger, lower income, smaller loans

### KMeans Cluster Profiles
| Feature             | Cluster 0 (1617) | Cluster 1 (827) | Interpretation            |
|---------------------|------------------|-----------------|---------------------------|
| loan_percent_income | -0.192           | +0.303          | Cluster 1 = higher burden |
| person_income       | -0.362           | +0.756          | Cluster 1 = higher income |
| loan_amnt           | -0.496           | +0.941          | Cluster 1 = larger loans  |
| loan_int_rate       | -0.139           | +0.203          | Cluster 1 = higher rate   |
| person_age          | -0.179           | +0.393          | Cluster 1 = older         |

- Cluster 0 = Standard Borrowers — younger, lower income, smaller loans
- Cluster 1 = High Value Borrowers — older, higher income, larger loans

Both models found same 2 groups — consistent clustering

## 11. Test Set Results

### Classification
| Model              | Accuracy | F1   | Precision | Recall  | Macro F1 |
|--------------------|----------|------|-----------|---------|----------| 
| LightGBM           | 0.94     | 0.84 | 0.93      | 0.77    | 0.90     |
| XGBoost            | 0.93     | 0.83 | 0.94      | 0.75    | 0.90     |
| RandomForest       | 0.93     | 0.80 | 0.97      | 0.68    | 0.88     |
| LogisticRegression | 0.84     | 0.67 | 0.61      | 0.74    | 0.78     |

### Regression
| Model            | R2     | RMSE   | MAE    |
|------------------|--------|--------|--------|
| RandomForest     | 0.9077 | 0.9811 | 0.7697 |
| XGBoost          | 0.9071 | 0.9844 | 0.7707 |
| LightGBM         | 0.9066 | 0.9872 | 0.7722 |
| LinearRegression | 0.8772 | 1.1318 | 0.8869 |

### Clustering

| Model  | Silhouette | Davies-Bouldin |
|--------|------------|----------------|
| GMM    | 0.2287     | 4.2214         |
| KMeans | 0.1584     | 2.8406         |

## 12. Val vs Test Comparison

### Classification
| Model              | Val F1 | Test F1 | Change | Stable?  |
|--------------------|--------|---------|--------|----------|
| LightGBM           | 0.82   | 0.84    | +0.02  | Improved |
| XGBoost            | 0.82   | 0.83    | +0.01  | Stable   |
| RandomForest       | 0.78   | 0.80    | +0.02  | Stable   |
| LogisticRegression | 0.66   | 0.67    | +0.01  | Stable   |

### Regression
| Model            | Val R2 | Test R2 | Change | Stable? |
|------------------|--------|---------|--------|---------|
| RandomForest     | 0.9137 | 0.9077  | -0.006 | Stable  |
| XGBoost          | 0.9123 | 0.9071  | -0.005 | Stable  |
| LightGBM         | 0.9115 | 0.9066  | -0.005 | Stable  |
| LinearRegression | 0.8826 | 0.8772  | -0.005 | Stable  |

### Clustering
| Model  | Val Silhouette | Test Silhouette | Stable? |
|--------|----------------|-----------------|---------|
| GMM    | 0.2287         | 0.2287          | Perfect |
| KMeans | 0.1651         | 0.1584          | Stable  |

No overfitting detected — all models generalize well

## 13. Final Model Rankings
| Task           | Best Model   | Key Metric             | Reason                                 |
|----------------|--------------|------------------------|----------------------------------------|
| Classification | LightGBM     | F1=0.84, AUC=0.9414    | Best F1, AUC, fewest missed defaulters |
| Regression     | RandomForest | R2=0.9077, RMSE=0.9811 | Best R2, RMSE, MAE                     |
| Clustering     | GMM          | Silhouette=0.2287      | Best silhouette, soft probabilities    |

## 14. Saved Files
| File                                                    | Description                 |
|---------------------------------------------------------|-----------------------------|
| models/Project_Parameter_Files/test_results.json        | All test results            |
| models/Project_Parameter_Files/cls_test_results.csv     | Classification test results |
| models/Project_Parameter_Files/reg_test_results.csv     | Regression test results     |
| models/Project_Parameter_Files/cluster_test_results.csv | Clustering test results     |