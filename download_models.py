"""
download_models.py
Downloads all model files from Hugging Face Hub at container startup.
Called by Dockerfile.api before starting uvicorn.
"""

import os
from huggingface_hub import hf_hub_download


REPO_ID   = "Abdulmoiz123/loan-risk-models"
REPO_TYPE = "model"
HF_TOKEN  = os.getenv("HF_TOKEN")  # set as environment variable on Render


# Files to download — (repo path, local path)
FILES = [
    # ONNX models
    ("models_onnx/cls_LightGBM.onnx",                        "models_onnx/cls_LightGBM.onnx"),
    ("models_onnx/reg_RandomForest.onnx",                     "models_onnx/reg_RandomForest.onnx"),
    ("models_onnx/cluster_GMM.onnx",                          "models_onnx/cluster_GMM.onnx"),

    # Production pipelines
    ("models/Production_pipelines/cls_pipeline.joblib",       "models/Production_pipelines/cls_pipeline.joblib"),
    ("models/Production_pipelines/cls_fe_pipeline.joblib",    "models/Production_pipelines/cls_fe_pipeline.joblib"),
    ("models/Production_pipelines/reg_pipeline.joblib",       "models/Production_pipelines/reg_pipeline.joblib"),
    ("models/Production_pipelines/reg_fe_pipeline.joblib",    "models/Production_pipelines/reg_fe_pipeline.joblib"),

    # Final models (for SHAP)
    ("models/Final_Models/cls_LightGBM.joblib",               "models/Final_Models/cls_LightGBM.joblib"),
    ("models/Final_Models/reg_RandomForest.joblib",            "models/Final_Models/reg_RandomForest.joblib"),

    # Feature lists
    ("models/Project_Parameter_Files/feature_lists.json",     "models/Project_Parameter_Files/feature_lists.json"),

    # Engineered data (SHAP background)
    ("data/engineered_data/X_train_cls_engineered.csv",       "data/engineered_data/X_train_cls_engineered.csv"),
    ("data/engineered_data/X_train_reg_engineered.csv",       "data/engineered_data/X_train_reg_engineered.csv"),
]


def download_all():
    print("Downloading model files from Hugging Face...")

    for repo_path, local_path in FILES:
        # Skip if already exists
        if os.path.exists(local_path):
            print(f"   SKIP  {local_path} (already exists)")
            continue

        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"   DOWN  {repo_path}")
        hf_hub_download(
            repo_id   = REPO_ID,
            repo_type = REPO_TYPE,
            filename  = repo_path,
            token     = HF_TOKEN,
            local_dir = "."
        )
        print(f"   DONE  {local_path}")

    print("\n✅ All model files ready")


if __name__ == "__main__":
    download_all()