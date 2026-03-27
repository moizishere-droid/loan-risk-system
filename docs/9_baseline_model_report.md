# Baseline Model Training Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: March 2026
# Phase: 9 — Baseline Model Training

## 1. Objective
Identify and train baseline models for classification, regression, and clustering tasks to establish minimum performance benchmarks. These benchmarks must be beaten by Phase 10 model training.

## 2. Data Used
| Task           | Dataset                                         | Shape       |
|----------------|-------------------------------------------------|-------------|
| Classification | data/engineered_data/X_train_cls_engineered.csv | (27577, 23) |
| Regression     | data/engineered_data/X_train_reg_engineered.csv | (24943, 16) |
| Clustering     | data/engineered_data/X_train_cls_engineered.csv | (27577, 23) |

## 3. Evaluation Metrics
| Task           | Metrics                                                                |
|----------------|------------------------------------------------------------------------|
| Classification | accuracy_score, f1_score, precision_score, recall_score, roc_auc_score |
| Regression     | r2_score, root_mean_squared_error, mean_absolute_error                 |
| Clustering     | silhouette_score, davies_bouldin_score, calinski_harabasz_score        |

## 4. Class Imbalance Handling
| Class | Label       | Distribution | Weight                |
|-------|-------------|--------------|-----------------------|
| 0     | Non-Default | 78.13%       | 0.6399 — downweighted |
| 1     | Default     | 21.87%       | 2.2867 — upweighted   |
Class weights applied to all classification models using class_weight parameter. Model pays 3.6x more attention to default cases.

## 5. Baseline Models Defined
| Task           | Models                                                      |
|----------------|-------------------------------------------------------------|
| Classification | DummyClassifier, LogisticRegression, DecisionTreeClassifier |
| Regression     | DummyRegressor, LinearRegression, DecisionTreeRegressor     |
| Clustering     | KMeans, DBSCAN, AgglomerativeClustering                     |

## 6. Results — Classification
| Model                  | Accuracy | F1     | Precision | Recall | ROC-AUC | Time  |
|------------------------|----------|--------|-----------|------ -|---------|-------|
| DummyClassifier        | 0.7819   | 0.0000 | 0.0000    | 0.0000 | 0.5000  | 0.04s |
| LogisticRegression     | 0.8020   | 0.6271 | 0.5320    | 0.7636 | 0.8630  | 0.13s |
| DecisionTreeClassifier | 0.8764   | 0.7118 | 0.7243    | 0.6998 | 0.8703  | 0.27s |
Best Baseline: DecisionTreeClassifier — Best F1 (0.7118) and ROC-AUC (0.8703)

## 7. Results — Regression
| Model                 | R2      | RMSE   | MAE    | Time   |
|-----------------------|---------|--------|--------|--------|
| DummyRegressor        | -0.0004 | 3.3505 | 2.8002 | 0.007s |
| LinearRegression      | 0.8826  | 1.1476 | 0.9045 | 0.10s  |
| DecisionTreeRegressor | 0.9096  | 1.0073 | 0.8022 | 0.30s  |
Best Baseline: DecisionTreeRegressor — Best R2 (0.9096) and RMSE (1.0073)

## 8. Results — Clustering
| Model                   | Silhouette | Davies-Bouldin | Calinski-Harabasz | N Clusters | Time   |
|-------------------------|------------|----------------|-------------------|------------|--------|
| KMeans                  | 0.1479     | 2.3775         | 2480.8676         | 3          | 20.01s |
| DBSCAN                  | -0.4251    | 1.7363         | 4.0750            | 102        | 14.91s |
| AgglomerativeClustering | 0.1553     | 2.4686         | 1910.2081         | 3          | 59.15s |
Best Baseline: AgglomerativeClustering — Best Silhouette (0.1553)
Note: DBSCAN failed badly — found 102 clusters with silhouette = -0.425. Default eps value was not suitable for this dataset.

## 9. Benchmarks for Phase 10
| Task           | Metric     | Baseline Value | Target    |
|------|---------|------------|----------------|-----------|
| Classification | F1         | 0.712          | Must beat |
| Classification | ROC-AUC    | 0.870          | Must beat |
| Regression     | R2         | 0.910          | Must beat |
| Regression     | RMSE       | 1.007          | Must beat |
| Clustering     | Silhouette | 0.155          | Must beat |

## 10. Saved Files
| File                                                 | Description                         |
|------------------------------------------------------|-------------------------------------|
| models/Baseline_Results/baseline_results.json        | All baseline results and metrics    |
| models/Baseline_Results/baseline_cls_results.csv     | Classification baseline results CSV |
| models/Baseline_Results/baseline_reg_results.csv     | Regression baseline results CSV     |
| models/Baseline_Results/baseline_cluster_results.csv | Clustering baseline results CSV     |