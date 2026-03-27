# Model Selection Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 8 March 2026
# Phase: 13 — Model Selection

## 1. Objective
Select the best model for each task — classification, regression, and clustering — based on comprehensive evaluation results from Phase 11 and explainability insights from Phase 12. Final selected models will be used in Phase 14 (ONNX Conversion) and Phase 16 (API Development).

**Note:** Classification and regression are independent tools serving different business purposes. Classification predicts loan default risk using loan_int_rate as an input feature. Regression predicts loan_int_rate itself from borrower profile. Each model is selected independently on its own merits.

## 2. Classification — Final Comparison
| Model     | Val F1 | Test F1| Val AUC| TestAUC| Val Accuracy| Test Accuracy| Val Precision| Test Precision| Val Recall| Test Recall|
|-----------|--------|--------|--------|--------|-------------|--------------|--------------|---------------|-----------|------------|
| LightGBM  | 0.8245 | 0.8390 | 0.9414 | 0.9471 | 0.9309      | 0.9358       | 0.9233       | 0.9253        | 0.7448    | 0.7674     |
| XGBoost   | 0.8197 | 0.8321 | 0.9399 | 0.9458 | 0.9313      | 0.9341       | 0.9574       | 0.9366        | 0.7167    | 0.7486     |
| RF        | 0.7847 | 0.8040 | 0.9259 | 0.9307 | 0.9218      | 0.9272       | 0.9831       | 0.9733        | 0.6529    | 0.6848     |

Best Model Per Metric:

| Metric         | Best Model   | Value  |
|----------------|--------------|--------|
| Val F1         | LightGBM     | 0.8245 |
| Test F1        | LightGBM     | 0.8390 |
| Val AUC        | LightGBM     | 0.9414 |
| Test AUC       | LightGBM     | 0.9471 |
| Val Accuracy   | XGBoost      | 0.9313 |
| Test Accuracy  | LightGBM     | 0.9358 |
| Val Precision  | RandomForest | 0.9831 |
| Test Precision | RandomForest | 0.9733 |
| Val Recall     | LightGBM     | 0.7448 |
| Test Recall    | LightGBM     | 0.7674 |

LightGBM wins 7 out of 10 metrics
Decision: LightGBM

| Factor      | Detail                                                                                      |
|-------------|---------------------------------------------------------------------------------------------|
| F1          | Best — 0.8390                                                                               |
| AUC         | Best — 0.9471                                                                               |
| Recall      | Best — 0.7674 — catches most defaulters                                                     |
| Calibration | Good — probabilities trustworthy                                                            |
| Reason      | Higher recall is critical for loan risk — missing a defaulter costs more than a false alarm |

## 3. Regression — Final Comparison
| Model        | Val R2 | Test R2 | Val RMSE | Test RMSE | Val MAE | Test MAE |
|--------------|--------|---------|----------|-----------|---------|----------|
| RandomForest | 0.9130 | 0.9077  | 0.9878   | 0.9811    | 0.7849  | 0.7697   |
| XGBoost      | 0.9127 | 0.9071  | 0.9899   | 0.9844    | 0.7897  | 0.7707   |
| LightGBM     | 0.9115 | 0.9066  | 0.9967   | 0.9872    | 0.7942  | 0.7722   |

Best Model Per Metric:

| Metric    | Best Model   | Value  | Direction       |
|-----------|--------------|--------|-----------------|
| Val R2    | RandomForest | 0.9130 | ↑ higher better |
| Test R2   | RandomForest | 0.9077 | ↑ higher better |
| Val RMSE  | RandomForest | 0.9878 | ↓ lower better  |
| Test RMSE | RandomForest | 0.9811 | ↓ lower better  |
| Val MAE   | RandomForest | 0.7849 | ↓ lower better  |
| Test MAE  | RandomForest | 0.7697 | ↓ lower better  |

RandomForest wins all 6 out of 6 metrics

Decision: RandomForest

| Factor | Detail                               |
|--------|--------------------------------------|
| R2     | Best — 0.9077                        |
| RMSE   | Best — 0.9811                        |
| MAE    | Best — 0.7697                        |
| Reason | Wins every single metric — no debate |

## 4. Clustering — Final Comparison
| Model  | Val Silhouette | Test Silhouette | Val Davies Bouldin | Test Davies Bouldin | Val Calinski | Test Calinski |
|--------|----------------|-----------------|--------------------|---------------------|--------------|---------------| 
| GMM    | 0.2287         | 0.2196          | 4.2214             | 4.3057              | 115.77       | 118.08        |
| KMeans | 0.1584         | 0.1508          | 2.8406             | 2.8335              | 236.50       | 258.20        |

Best Model Per Metric:

| Metric                 | Best Model | Value | Direction       |
|------------------------|------------|-------|-----------------|
| Val Silhouette         | GMM        | 0.2287| ↑ higher better |
| Test Silhouette        | GMM        | 0.2196| ↑ higher better |
| Val Davies Bouldin     | KMeans     | 2.8406| ↓ lower better  |
| Test Davies Bouldin    | KMeans     | 2.8335| ↓ lower better  |
| Val Calinski Harabasz  | KMeans     | 236.50| ↑ higher better |
| Test Calinski Harabasz | KMeans     | 258.20| ↑ higher better |

Split — GMM wins Silhouette, KMeans wins Davies Bouldin and Calinski
Decision: GMM

| Factor             | GMM    | KMeans | Winner |
|--------------------|--------|--------|--------|
| Silhouette         | 0.2196 | 0.1508 | GMM    |
| Davies Bouldin     | 4.3057 | 2.8335 | KMeans |
| Calinski Harabasz  | 118.08 | 258.20 | KMeans |
| Soft probabilities | Yes    | No     | GMM    |
| Production value   | High   | Medium | GMM    |

Reason: Silhouette is the primary metric for cluster separation quality. GMM soft probabilities add significant production value — API can return cluster membership probability (e.g. "70% High Value Borrower") instead of a hard label. This makes the system more informative and trustworthy for end users.

## 5. Final Selected Models
| Task           | Model            | Key Metrics                          | Threshold |
|----------------|------------------|--------------------------------------|-----------|
| Classification | cls_LightGBM     | F1=0.8390, AUC=0.9471, Recall=0.7674 | 0.6       |
| Regression     | reg_RandomForest | R2=0.9077, RMSE=0.9811, MAE=0.7697   | —         |
| Clustering     | cluster_GMM      | Silhouette=0.2196                    | —         |

Cluster Labels — GMM:

| Cluster ID | Label               |
|------------|---------------------|
| 0          | High Value Borrower |
| 1          | Standard Borrower   |

## 6. Saved Files
| File                                                                     | Description                                 |
|--------------------------------------------------------------------------|---------------------------------------------|
| models/Project_Parameter_Files/Evalution_result/final_model_config.json  | Final model config with metrics and reasons |
| models/Project_Parameter_Files/Evalution_result/cls_val_results.csv      | Classification val results                  |
| models/Project_Parameter_Files/Evalution_result/cls_test_results.csv     | Classification test results                 |
| models/Project_Parameter_Files/Evalution_result/reg_val_results.csv      | Regression val results                      |
| models/Project_Parameter_Files/Evalution_result/reg_test_results.csv     | Regression test results                     |
| models/Project_Parameter_Files/Evalution_result/cluster_val_results.csv  | Clustering val results                      |
| models/Project_Parameter_Files/Evalution_result/cluster_test_results.csv | Clustering test results                     |