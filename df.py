from huggingface_hub import HfApi

import os
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Abdulmoiz123/loan-risk-models"

# Upload ONNX models
api.upload_file(path_or_fileobj="models_onnx/cls_LightGBM.onnx",     path_in_repo="models_onnx/cls_LightGBM.onnx",     repo_id=repo_id)
api.upload_file(path_or_fileobj="models_onnx/reg_RandomForest.onnx", path_in_repo="models_onnx/reg_RandomForest.onnx", repo_id=repo_id)
api.upload_file(path_or_fileobj="models_onnx/cluster_GMM.onnx",      path_in_repo="models_onnx/cluster_GMM.onnx",      repo_id=repo_id)

# Upload production pipelines
api.upload_file(path_or_fileobj="models/Production_pipelines/cls_pipeline.joblib",    path_in_repo="models/Production_pipelines/cls_pipeline.joblib",    repo_id=repo_id)
api.upload_file(path_or_fileobj="models/Production_pipelines/cls_fe_pipeline.joblib", path_in_repo="models/Production_pipelines/cls_fe_pipeline.joblib", repo_id=repo_id)
api.upload_file(path_or_fileobj="models/Production_pipelines/reg_pipeline.joblib",    path_in_repo="models/Production_pipelines/reg_pipeline.joblib",    repo_id=repo_id)
api.upload_file(path_or_fileobj="models/Production_pipelines/reg_fe_pipeline.joblib", path_in_repo="models/Production_pipelines/reg_fe_pipeline.joblib", repo_id=repo_id)

# Upload final models (for SHAP)
api.upload_file(path_or_fileobj="models/Final_Models/cls_LightGBM.joblib", path_in_repo="models/Final_Models/cls_LightGBM.joblib", repo_id=repo_id)
api.upload_file(path_or_fileobj="models/Final_Models/reg_RandomForest.joblib", path_in_repo="models/Final_Models/reg_RandomForest.joblib", repo_id=repo_id)

# Upload feature lists
api.upload_file(path_or_fileobj="models/Project_Parameter_Files/feature_lists.json", path_in_repo="models/Project_Parameter_Files/feature_lists.json", repo_id=repo_id)

# Upload engineered data (for SHAP background)
api.upload_file(path_or_fileobj="data/engineered_data/X_train_cls_engineered.csv", path_in_repo="data/engineered_data/X_train_cls_engineered.csv", repo_id=repo_id)
api.upload_file(path_or_fileobj="data/engineered_data/X_train_reg_engineered.csv", path_in_repo="data/engineered_data/X_train_reg_engineered.csv", repo_id=repo_id)

print("✅ All files uploaded")