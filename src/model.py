import pandas as pd
import numpy as np
import joblib
import os
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# IMPORTS
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils.class_weight import compute_class_weight


# BEST PARAMS — from Phase 10 Optuna tuning in Notebook

CLS_BEST_PARAMS = {
    'RandomForest': {
        'n_estimators':      217,
        'max_depth':         15,
        'min_samples_split': 5,
        'min_samples_leaf':  1,
        'max_features':      'sqrt'
    },
    'XGBoost': {
        'n_estimators':     184,
        'max_depth':        7,
        'learning_rate':    0.17874952361375027,
        'subsample':        0.9484244741385437,
        'colsample_bytree': 0.7018992424964301
    },
    'LightGBM': {
        'n_estimators':  242,
        'max_depth':     7,
        'learning_rate': 0.1540738920567998,
        'num_leaves':    79,
        'subsample':     0.8583481396183521
    }
}

REG_BEST_PARAMS = {
    'RandomForest': {
        'n_estimators':      171,
        'max_depth':         15,
        'min_samples_split': 8,
        'min_samples_leaf':  2,
        'max_features':      'sqrt'
    },
    'XGBoost': {
        'n_estimators':     236,
        'max_depth':        6,
        'learning_rate':    0.05098353526886846,
        'subsample':        0.9324172850947218,
        'colsample_bytree': 0.8345534165318191
    },
    'LightGBM': {
        'n_estimators':  184,
        'max_depth':     6,
        'learning_rate': 0.042521487869093835,
        'num_leaves':    40,
        'subsample':     0.7029242979243545
    }
}

CLUSTER_BEST_PARAMS = {
    'KMeans': {
        'n_clusters': 2,
        'init':       'k-means++',
        'n_init':     24
    },
    'GMM': {
        'n_components':    2,
        'covariance_type': 'spherical'
    }
}


# DATA LOADING

def load_config():
    """Load project config from config/config.yaml"""
    base_dir = Path(__file__).resolve().parent.parent
    config_file = base_dir / 'config' / 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config, base_dir


def load_engineered_splits():
    """
    Load all engineered splits for training.
    Returns:
        dict: All X and y splits for cls, reg tasks
    """
    config, base_dir = load_config()
    engineered_path = base_dir / config['paths']['engineered_data']

    # Classification
    X_train_cls = pd.read_csv(engineered_path / 'X_train_cls_engineered.csv')
    X_val_cls   = pd.read_csv(engineered_path / 'X_val_cls_engineered.csv')
    X_test_cls  = pd.read_csv(engineered_path / 'X_test_cls_engineered.csv')
    y_train_cls = pd.read_csv(engineered_path / 'y_train_cls_engineered.csv')
    y_val_cls   = pd.read_csv(engineered_path / 'y_val_cls_engineered.csv')
    y_test_cls  = pd.read_csv(engineered_path / 'y_test_cls_engineered.csv')

    # Regression
    X_train_reg = pd.read_csv(engineered_path / 'X_train_reg_engineered.csv')
    X_val_reg   = pd.read_csv(engineered_path / 'X_val_reg_engineered.csv')
    X_test_reg  = pd.read_csv(engineered_path / 'X_test_reg_engineered.csv')
    y_train_reg = pd.read_csv(engineered_path / 'y_train_reg_engineered.csv')
    y_val_reg   = pd.read_csv(engineered_path / 'y_val_reg_engineered.csv')
    y_test_reg  = pd.read_csv(engineered_path / 'y_test_reg_engineered.csv')

    print("    All engineered splits loaded")
    print(f"   Classification train: {X_train_cls.shape}")
    print(f"   Regression train:     {X_train_reg.shape}")

    return {
        'cls': {
            'X_train_cls': X_train_cls, 'X_val_cls': X_val_cls, 'X_test_cls': X_test_cls,
            'y_train_cls': y_train_cls, 'y_val_cls': y_val_cls, 'y_test_cls': y_test_cls
        },
        'reg': {
            'X_train_reg': X_train_reg, 'X_val_reg': X_val_reg, 'X_test_reg': X_test_reg,
            'y_train_reg': y_train_reg, 'y_val_reg': y_val_reg, 'y_test_reg': y_test_reg
        }
    }


# CLASS WEIGHTS
def get_class_weights(y_train_cls):
    """
    Compute class weights for imbalanced classification.
    Args:
        y_train_cls: training labels
    Returns:
        dict: {class: weight}
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_cls),
        y=y_train_cls.values.ravel()
    )
    class_weight_dict = dict(zip(np.unique(y_train_cls.values.ravel()), weights))
    print(f"   Class weights computed: {class_weight_dict}")
    return class_weight_dict


# BUILD MODELS
def build_cls_models(class_weight_dict):
    """
    Build all classification models with best tuned params.
    Args:
        class_weight_dict: class weights for imbalance handling
    Returns:
        dict: {model_name: model}
    """
    models = {
        'LogisticRegression': LogisticRegression(
            class_weight=class_weight_dict,
            random_state=42,
            max_iter=1000),

        'RandomForest': RandomForestClassifier(
            **CLS_BEST_PARAMS['RandomForest'],
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1),

        'XGBoost': XGBClassifier(
            **CLS_BEST_PARAMS['XGBoost'],
            scale_pos_weight=2.28,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1),

        'LightGBM': LGBMClassifier(
            **CLS_BEST_PARAMS['LightGBM'],
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,
            verbose=-1)
    }

    print(" Classification models built:")
    for name in models:
        print(f"   - {name}")
    return models


def build_reg_models():
    """
    Build all regression models with best tuned params.
    Returns:
        dict: {model_name: model}
    """
    models = {
        'LinearRegression': LinearRegression(
            n_jobs=-1),

        'RandomForest': RandomForestRegressor(
            **REG_BEST_PARAMS['RandomForest'],
            random_state=42,
            n_jobs=-1),

        'XGBoost': XGBRegressor(
            **REG_BEST_PARAMS['XGBoost'],
            random_state=42,
            n_jobs=-1),

        'LightGBM': LGBMRegressor(
            **REG_BEST_PARAMS['LightGBM'],
            random_state=42,
            n_jobs=-1,
            verbose=-1)
    }

    print("  Regression models built:")
    for name in models:
        print(f"   - {name}")
    return models


def build_cluster_models():
    """
    Build all clustering models with best tuned params.
    Returns:
        dict: {model_name: model}
    """
    models = {
        'KMeans': KMeans(
            **CLUSTER_BEST_PARAMS['KMeans'],
            random_state=42),

        'GMM': GaussianMixture(
            **CLUSTER_BEST_PARAMS['GMM'],
            random_state=42)
    }

    print("  Clustering models built:")
    for name in models:
        print(f"   - {name}")
    return models


# TRAIN MODELS

def train_cls_model(model, X_train_cls, y_train_cls):
    """
    Train a single classification model.
    Args:
        model: sklearn compatible classifier
        X_train_cls: training features
        y_train_cls: training labels
    Returns:
        trained model
    """
    import time
    start = time.time()
    model.fit(X_train_cls, y_train_cls.values.ravel())
    elapsed = round(time.time() - start, 2)
    print(f"    Trained in {elapsed}s")
    return model


def train_reg_model(model, X_train_reg, y_train_reg):
    """
    Train a single regression model.
    Args:
        model: sklearn compatible regressor
        X_train_reg: training features
        y_train_reg: training labels
    Returns:
        trained model
    """
    import time
    start = time.time()
    model.fit(X_train_reg, y_train_reg.values.ravel())
    elapsed = round(time.time() - start, 2)
    print(f"    Trained in {elapsed}s")
    return model


def train_cluster_model(model, X_train_cls):
    """
    Train a single clustering model.
    Args:
        model: sklearn compatible clustering model
        X_train_cls: training features
    Returns:
        trained model
    """
    import time
    start = time.time()
    model.fit(X_train_cls)
    elapsed = round(time.time() - start, 2)
    print(f"    Trained in {elapsed}s")
    return model


def train_all_models(data):
    """
    Train all models for all 3 tasks.
    Args:
        data: dict from load_engineered_splits()
    Returns:
        dict: all trained models
    """
    print("TRAINING ALL MODELS")

    # Class weights
    class_weight_dict = get_class_weights(data['cls']['y_train_cls'])

    # Build models
    cls_models     = build_cls_models(class_weight_dict)
    reg_models     = build_reg_models()
    cluster_models = build_cluster_models()

    # Train classification
    print("\n--- Classification ---")
    trained_cls = {}
    for name, model in cls_models.items():
        print(f"Training {name}...")
        trained_cls[name] = train_cls_model(
            model,
            data['cls']['X_train_cls'],
            data['cls']['y_train_cls'])

    # Train regression
    print("\n--- Regression ---")
    trained_reg = {}
    for name, model in reg_models.items():
        print(f"Training {name}...")
        trained_reg[name] = train_reg_model(
            model,
            data['reg']['X_train_reg'],
            data['reg']['y_train_reg'])

    # Train clustering
    print("\n--- Clustering ---")
    trained_cluster = {}
    for name, model in cluster_models.items():
        print(f"Training {name}...")
        trained_cluster[name] = train_cluster_model(
            model,
            data['cls']['X_train_cls'])

    print("\n All models trained successfully")
    return {
        'cls':     trained_cls,
        'reg':     trained_reg,
        'cluster': trained_cluster
    }


# SAVE AND LOAD MODELS
def save_model(model, name, task_type):
    """
    Save a trained model to models/ directory.
    Args:
        model: trained model
        name: model name (e.g. 'XGBoost')
        task_type: 'cls', 'reg', or 'cluster'
    """
    config, base_dir = load_config()
    models_path = base_dir / 'models/Final_Models'
    os.makedirs(models_path, exist_ok=True)

    filename = f"{task_type}_{name}.joblib"
    filepath = models_path / filename
    joblib.dump(model, filepath)
    print(f" Saved {filename}")


def load_model(name, task_type):
    """
    Load a trained model from models/ directory.
    Args:
        name: model name (e.g. 'XGBoost')
        task_type: 'cls', 'reg', or 'cluster'
    Returns:
        loaded model
    """
    config, base_dir = load_config()
    models_path = base_dir / 'models/Final_Models'

    filename = f"{task_type}_{name}.joblib"
    filepath = models_path / filename
    model = joblib.load(filepath)
    print(f" Loaded {filename}")
    return model


def save_all_models(trained_models):
    """
    Save all trained models to models/ directory.
    Args:
        trained_models: dict from train_all_models()
    """
    print("\n--- Saving all models ---")
    for name, model in trained_models['cls'].items():
        save_model(model, name, 'cls')
    for name, model in trained_models['reg'].items():
        save_model(model, name, 'reg')
    for name, model in trained_models['cluster'].items():
        save_model(model, name, 'cluster')
    print(" All models saved")


def load_all_models():
    """
    Load all trained models from models/ directory.
    Returns:
        dict: all loaded models
    """
    print("\n--- Loading all models ---")
    cls_names     = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']
    reg_names     = ['LinearRegression', 'RandomForest', 'XGBoost', 'LightGBM']
    cluster_names = ['KMeans', 'GMM']

    return {
        'cls':     {name: load_model(name, 'cls') for name in cls_names},
        'reg':     {name: load_model(name, 'reg') for name in reg_names},
        'cluster': {name: load_model(name, 'cluster') for name in cluster_names}
    }


# MAIN — full training run
if __name__ == "__main__":

    # Step 1: Load data
    data = load_engineered_splits()

    # Step 2: Train all models
    trained_models = train_all_models(data)

    # Step 3: Save all models
    save_all_models(trained_models)

    print("\n model.py complete — all models trained and saved")