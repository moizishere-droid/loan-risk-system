import os
import json
import joblib
import shap
import pandas as pd
import onnxruntime as rt

import warnings
warnings.filterwarnings('ignore')


# PATHS
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR  = os.path.join(BASE_DIR, 'models_onnx')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'engineered_data')


# GLOBAL VARIABLES
# ONNX sessions
_cls_session     = None
_reg_session     = None
_cluster_session = None
# Preprocessing + FE pipelines
_cls_pipeline    = None
_reg_pipeline    = None
_cls_fe_pipeline = None
_reg_fe_pipeline = None
# SHAP explainers
_cls_explainer   = None
_reg_explainer   = None
# Feature lists
_cls_features    = None
_reg_features    = None
# Load status
_loaded = False


# LOAD FUNCTION — called once at API startup
def load_all_models():
    """
    Load all models, pipelines, and SHAP explainers.
    Called once at FastAPI startup via lifespan.
    """
    global _cls_session, _reg_session, _cluster_session
    global _cls_pipeline, _reg_pipeline
    global _cls_fe_pipeline, _reg_fe_pipeline
    global _cls_explainer, _reg_explainer
    global _cls_features, _reg_features
    global _loaded

    print("Loading models...")

    # ONNX sessions
    _cls_session     = rt.InferenceSession(os.path.join(ONNX_DIR, 'cls_LightGBM.onnx'))
    _reg_session     = rt.InferenceSession(os.path.join(ONNX_DIR, 'reg_RandomForest.onnx'))
    _cluster_session = rt.InferenceSession(os.path.join(ONNX_DIR, 'cluster_GMM.onnx'))
    print(" DONE --> ONNX sessions loaded")

    # Preprocessing pipelines
    _cls_pipeline = joblib.load(os.path.join(MODEL_DIR, 'Production_pipelines', 'cls_pipeline.joblib'))
    _reg_pipeline = joblib.load(os.path.join(MODEL_DIR, 'Production_pipelines', 'reg_pipeline.joblib'))
    print(" DONE --> Preprocessing pipelines loaded")

    # FE pipelines
    _cls_fe_pipeline = joblib.load(os.path.join(MODEL_DIR, 'Production_pipelines', 'cls_fe_pipeline.joblib'))
    _reg_fe_pipeline = joblib.load(os.path.join(MODEL_DIR, 'Production_pipelines', 'reg_fe_pipeline.joblib'))
    print(" DONE --> FE pipelines loaded")

    # Feature lists
    with open(os.path.join(MODEL_DIR, 'Project_Parameter_Files/feature_lists.json'), 'r') as f:
        feature_lists = json.load(f)

    _cls_features = feature_lists['classification']['features']
    _reg_features = feature_lists['regression']['features']
    print("DONE --> Feature lists loaded")

    # SHAP explainers
    X_train_cls  = pd.read_csv(os.path.join(DATA_DIR, 'X_train_cls_engineered.csv'))
    X_train_reg  = pd.read_csv(os.path.join(DATA_DIR, 'X_train_reg_engineered.csv'))

    cls_model    = joblib.load(os.path.join(MODEL_DIR, 'Final_Models', 'cls_LightGBM.joblib'))
    reg_model    = joblib.load(os.path.join(MODEL_DIR, 'Final_Models', 'reg_RandomForest.joblib'))

    _cls_explainer = shap.TreeExplainer(cls_model, X_train_cls)
    _reg_explainer = shap.TreeExplainer(reg_model, X_train_reg)
    print("DONE --> SHAP explainers loaded")

    _loaded = True
    print("\n All models loaded successfully")


# GETTERS — used by pipeline.py
def is_loaded()          : return _loaded
def get_cls_session()    : return _cls_session
def get_reg_session()    : return _reg_session
def get_cluster_session(): return _cluster_session
def get_cls_pipeline()   : return _cls_pipeline
def get_reg_pipeline()   : return _reg_pipeline
def get_cls_fe_pipeline(): return _cls_fe_pipeline
def get_reg_fe_pipeline(): return _reg_fe_pipeline
def get_cls_explainer()  : return _cls_explainer
def get_reg_explainer()  : return _reg_explainer
def get_cls_features()   : return _cls_features
def get_reg_features()   : return _reg_features



def get_model_status() -> dict:
    """
    Returns load status of all models.
    Used by /health endpoint.
    """
    return {
        'cls_onnx'        : _cls_session     is not None,
        'reg_onnx'        : _reg_session     is not None,
        'cluster_onnx'    : _cluster_session is not None,
        'cls_pipeline'    : _cls_pipeline    is not None,
        'reg_pipeline'    : _reg_pipeline    is not None,
        'cls_fe_pipeline' : _cls_fe_pipeline is not None,
        'reg_fe_pipeline' : _reg_fe_pipeline is not None,
        'cls_explainer'   : _cls_explainer   is not None,
        'reg_explainer'   : _reg_explainer   is not None,
    }