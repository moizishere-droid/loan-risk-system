# Project : End-to-End Loan Risk Assessment System (Tabular ML)

This project builds a production-style machine learning system for financial risk assessment using a single applicant dataset. The goal is to simulate how banks evaluate loan applications by combining multiple machine learning approaches into three independent but complementary tools.

The system includes three core models:

1. Loan Default Risk Classifier (Classification)
Predicts whether a loan application should be approved or rejected based on applicant financial and demographic features, including the assigned interest rate.


2. Interest Rate Predictor (Regression)
Estimates the interest rate a bank would assign to a borrower based on their profile. Since interest rate reflects credit risk, a higher predicted rate signals a riskier applicant.


3. Customer Segmentation Model (Clustering)
Groups applicants into behavioral or risk segments using unsupervised learning. These segments can be used for differentiated loan policies, offers, or risk strategies.


The project follows a full industry lifecycle including data validation, preprocessing pipelines, feature engineering, multi-model training, cross-validation, explainability using SHAP, model comparison and selection, ONNX model conversion, Database, API development with FastAPI, containerization using Docker, deployment to cloud, and monitoring/logging setup.

Each tool operates independently and serves a different stage of the loan management lifecycle, together forming a comprehensive loan risk assessment platform.

This project demonstrates end-to-end ownership of a tabular machine learning system, covering supervised and unsupervised learning, deployment, and production considerations.