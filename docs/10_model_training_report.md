# Model Training Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 5 to 6 March 2026

## 1. Objective
Train and evaluate multiple models for classification, regression, and clustering tasks using cross validation and Optuna hyperparameter tuning. Multiple models are trained so the best candidate can be selected in Phase 13 — Model Selection.

## 2. Data Used
| Task           | Dataset                                         | Shape       |
|----------------|-------------------------------------------------|-------------|
| Classification | data/engineered_data/X_train_cls_engineered.csv | (27577, 23) |
| Regression     | data/engineered_data/X_train_reg_engineered.csv | (24943, 16) |
| Clustering     | data/engineered_data/X_train_cls_engineered.csv | (27577, 23) |

**Note:** Classification uses loan_int_rate as an input feature (23 features). Regression predicts loan_int_rate as its target and therefore excludes it from input features (16 features). Both tasks are independent tools serving different business purposes.

## 3. Models Trained
| Task           | Models                                              | Total |
|----------------|-----------------------------------------------------|-------|
| Classification | LogisticRegression, RandomForest, XGBoost, LightGBM | 4     |
| Regression     | LinearRegression, RandomForest, XGBoost, LightGBM   | 4     |
| Clustering     | KMeans, GaussianMixture (GMM)                       | 2     |
Total models trained: 10

## 4. Cross Validation Results
### 4.1 Classification (StratifiedKFold, 5 folds, scoring=F1)

| Model              | F1 Mean | F1 Std  | ROC-AUC Mean | ROC-AUC Std | Time   |
|--------------------|---------|---------|--------------|-------------|--------|
| LogisticRegression | 0.6151  | ±0.0104 | 0.8630       | ±0.0050     | 3.26s  |
| RandomForest       | 0.8077  | ±0.0072 | 0.9308       | ±0.0039     | 15.89s |
| XGBoost            | 0.8237  | ±0.0077 | 0.9425       | ±0.0032     | 1.37s  |
| LightGBM           | 0.8100  | ±0.0085 | 0.9438       | ±0.0033     | 1.44s  |

### 4.2 Regression (KFold, 5 folds, scoring=R2)
| Model            | R2 Mean | R2 Std  | RMSE Mean | RMSE Std| Time    |
|------------------|---------|---------|-----------|---------|---------|
| LinearRegression | 0.8757  | ±0.0026 | 1.1394    | ±0.0112 | 2.90s   |
| RandomForest     | 0.9058  | ±0.0026 | 0.9920    | ±0.0121 | 100.11s |
| XGBoost          | 0.9014  | ±0.0024 | 1.0147    | ±0.0104 | 1.83s   |
| LightGBM         | 0.9050  | ±0.0022 | 0.9960    | ±0.0099 | 1.87s   |
Decision: LogisticRegression and LinearRegression skipped for tuning — too weak compared to ensemble models.

## 5. Hyperparameter Tuning — Optuna
Method: Optuna Bayesian optimization, 20 trials per model
### 5.1 Classification Best Parameters

| Model        | Best F1 | Key Parameters                                                         |
|--------------|---------|------------------------------------------------------------------------|
| RandomForest | 0.8072  | n_estimators=217, max_depth=15, min_samples_split=5, max_features=sqrt |
| XGBoost      | 0.8318  | n_estimators=184, max_depth=7, learning_rate=0.178, subsample=0.948    |
| LightGBM     | 0.8249  | n_estimators=242, max_depth=7, learning_rate=0.154, num_leaves=79      |

### 5.2 Regression Best Parameters
| Model        | Best R2 | Key Parameters                                                         |
|--------------|---------|------------------------------------------------------------------------|
| RandomForest | 0.9057  | n_estimators=171, max_depth=15, min_samples_split=8, max_features=sqrt |
| XGBoost      | 0.9053  | n_estimators=236, max_depth=6, learning_rate=0.051, subsample=0.932    |
| LightGBM     | 0.9050  | n_estimators=184, max_depth=6, learning_rate=0.043, num_leaves=40      |

### 5.3 Clustering Best Parameters
| Model  | Best Silhouette | Key Parameters                            |
|--------|-----------------|-------------------------------------------|
| KMeans | 0.1553          | n_clusters=2, init=k-means++, n_init=24   |
| GMM    | 0.2332          | n_components=2, covariance_type=spherical |
Note: n_clusters overridden to 2 for KMeans and GMM based on elbow method and silhouette analysis. Hierarchical clustering dropped — no predict() method, unstable on unseen data.

## 6. Final Validation Results
### 6.1 Classification

| Model              | Accuracy | F1     | Precision | Recall | ROC-AUC |
|--------------------|----------|--------|-----------|--------|---------| 
| RandomForest       | 0.9206   | 0.7909 | 0.9291    | 0.6886 | 0.9251  |
| XGBoost            | 0.9264   | 0.8133 | 0.9095    | 0.7355 | 0.9397  |
| LightGBM           | 0.9218   | 0.8107 | 0.8592    | 0.7674 | 0.9379  |
| LogisticRegression | 0.8020   | 0.6271 | 0.5320    | 0.7636 | 0.8630  |
Best Model: XGBoost — Best F1 (0.8133) and ROC-AUC (0.9397)

### 6.2 Regression
| Model            | R2     | RMSE   | MAE    |
|------------------|--------|--------|--------|
| RandomForest     | 0.9137 | 0.9842 | 0.7829 |
| XGBoost          | 0.9123 | 0.9919 | 0.7906 |
| LightGBM         | 0.9115 | 0.9963 | 0.7939 |
| LinearRegression | 0.8826 | 1.1476 | 0.9045 |
Best Model: RandomForest — Best R2 (0.9137), RMSE (0.9842), MAE (0.7829)

### 6.3 Clustering
| Model | Silhouette | Davies-Bouldin | Calinski-Harabasz | N Clusters | Cluster Sizes     |
|-------|------------|----------------|-------------------|------------|-------------------|
| KMeans| 0.1651     | 2.8308         | 238.1061          | 2          | {0: 804, 1: 1640} |
| GMM   | 0.2287     | 4.2214         | 115.7651          | 2          | {0: 660, 1: 1784} |
Best Model: GMM — Best Silhouette (0.2287)

## 7. Best Models Summary
| Task           | Best Model   | Key Metric                |
|----------------|--------------|---------------------------|
| Classification | XGBoost      | F1=0.8133, ROC-AUC=0.9397 |
| Regression     | RandomForest | R2=0.9137, RMSE=0.9842    |
| Clustering     | GMM          | Silhouette=0.2287         |


## 8. Benchmarks vs Baseline (Phase 9)
### Classification
| Metric  | Baseline | Phase 10 Best | Improvement |
|---------|----------|---------------|-------------|
| F1      | 0.712    | 0.813         | +0.101      |
| ROC-AUC | 0.870    | 0.940         | +0.070      |

### Regression
| Metric | Baseline | Phase 10 Best | Improvement |
|--------|----------|---------------|-------------|
| R2     | 0.910    | 0.914         | +0.004      |
| RMSE   | 1.007    | 0.984         | -0.023      |

### Clustering
| Metric     | Baseline | Phase 10 Best | Improvement |
|------------|----------|---------------|-------------|
| Silhouette | 0.155    | 0.229         | +0.074      |
All Phase 10 models beat their respective baselines.

## 9. Key Decisions Made
| Decision                            | Reason                                     |
|-------------------------------------|--------------------------------------------|
| Dropped Hierarchical clustering     | No predict() method — not production ready |
| Used k=2 for KMeans and GMM         | Elbow method + silhouette analysis         |
| Skipped tuning LogisticRegression   | CV F1=0.615 — too weak                     |
| Skipped tuning LinearRegression     | CV R2=0.876 — too weak                     |
| Used Optuna over RandomizedSearchCV | Bayesian optimization — smarter and faster |
| Used scale_pos_weight for XGBoost   | Handles class imbalance (78/22 split)      |

## 10. Saved Files
| File                                                         | Description                       |
|--------------------------------------------------------------|-----------------------------------|
| models/Final_Models/cls_RandomForest.joblib                  | Classification RandomForest       |
| models/Final_Models/cls_XGBoost.joblib                       | Classification XGBoost            |
| models/Final_Models/cls_LightGBM.joblib                      | Classification LightGBM           |
| models/Final_Models/cls_LogisticRegression.joblib            | Classification LogisticRegression |
| models/Final_Models/reg_RandomForest.joblib                  | Regression RandomForest           |
| models/Final_Models/reg_XGBoost.joblib                       | Regression XGBoost                |
| models/Final_Models/reg_LightGBM.joblib                      | Regression LightGBM               |
| models/Final_Models/reg_LinearRegression.joblib              | Regression LinearRegression       |
| models/Final_Models/cluster_KMeans.joblib                    | Clustering KMeans                 |
| models/Final_Models/cluster_GMM.joblib                       | Clustering GMM                    |
| models/Project_Parameter_Files/training_results.json         | All results and best params       |
| models/Project_Parameter_Files/cls_val_results.csv           | Classification val results CSV    |
| models/Project_Parameter_Files/reg_val_results.csv           | Regression val results CSV        |
| models/Project_Parameter_Files/class_weights.pkl             | Class weight for Imbalanced data  |