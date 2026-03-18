import numpy as np
import pandas as pd
import onnxruntime as rt
import os

import warnings
warnings.filterwarnings('ignore')

# CONSTANTS
BEST_THRESHOLD_CLS = 0.6

CLUSTER_LABELS = {
    'cluster_GMM': {
        0: 'High Value Borrower',
        1: 'Standard Borrower'
    }
}


# LOAD ONNX SESSIONS
def load_onnx_sessions(onnx_dir):
    """
    Load all ONNX inference sessions.
    Args:
        onnx_dir: path to models_onnx directory
    Returns:
        dict: ONNX sessions for cls, reg, cluster
    """
    sessions = {}

    models = {
        'cls':     'cls_LightGBM.onnx',
        'reg':     'reg_RandomForest.onnx',
        'cluster': 'cluster_GMM.onnx'
    }

    for task, filename in models.items():
        path = os.path.join(onnx_dir, filename)
        if os.path.exists(path):
            sessions[task] = rt.InferenceSession(path)
            print(f" Loaded {filename}")
        else:
            print(f" Not found: {path}")
    return sessions


def load_cls_session(onnx_dir):
    """Load classification ONNX session."""
    path = os.path.join(onnx_dir, 'cls_LightGBM.onnx')
    return rt.InferenceSession(path)


def load_reg_session(onnx_dir):
    """Load regression ONNX session."""
    path = os.path.join(onnx_dir, 'reg_RandomForest.onnx')
    return rt.InferenceSession(path)


def load_cluster_session(onnx_dir):
    """Load clustering ONNX session."""
    path = os.path.join(onnx_dir, 'cluster_GMM.onnx')
    return rt.InferenceSession(path)


# PREPROCESSING FOR ONNX
def to_float32(X):
    """
    Convert dataframe or array to float32 numpy array.
    ONNX Runtime requires float32 input.
    Args:
        X: pd.DataFrame or np.ndarray
    Returns:
        np.ndarray: float32 array
    """
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32)
    return np.array(X).astype(np.float32)


# INFERENCE FUNCTIONS
def predict_cls_onnx(session, X):
    """
    Run classification inference using ONNX Runtime.
    Args:
        session: ONNX InferenceSession for classification
        X: input features (DataFrame or array)
    Returns:
        tuple: (y_pred, y_prob)
            y_pred: binary predictions using BEST_THRESHOLD_CLS
            y_prob: default class probabilities
    """
    X_float = to_float32(X)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_float})

    # LightGBM ONNX returns list of dicts — extract class 1 prob
    y_prob = np.array([p[1] for p in outputs[1]])
    y_pred = (y_prob >= BEST_THRESHOLD_CLS).astype(int)
    return y_pred, y_prob


def predict_reg_onnx(session, X):
    """
    Run regression inference using ONNX Runtime.
    Args:
        session: ONNX InferenceSession for regression
        X: input features (DataFrame or array)
    Returns:
        np.ndarray: predicted interest rates
    """
    X_float    = to_float32(X)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_float})
    return outputs[0].ravel()


def predict_cluster_onnx(session, X):
    """
    Run clustering inference using ONNX Runtime.
    Args:
        session: ONNX InferenceSession for clustering
        X: input features (DataFrame or array)
    Returns:
        tuple: (labels, probabilities)
            labels: cluster assignments
            probabilities: soft membership probabilities
    """
    X_float    = to_float32(X)
    input_name = session.get_inputs()[0].name
    outputs      = session.run(None, {input_name: X_float})
    labels       = outputs[0].ravel()
    probabilities = outputs[1]
    return labels, probabilities

# FULL PREDICTION PIPELINE
def predict_all_onnx(sessions, X_cls, X_reg):
    """
    Run full prediction pipeline using all ONNX models.
    Args:
        sessions: dict from load_onnx_sessions()
        X_cls: classification features
        X_reg: regression features
    Returns:
        dict: all predictions
    """
    # Classification
    y_pred_cls, y_prob_cls = predict_cls_onnx(
        sessions['cls'], X_cls)

    # Regression
    y_pred_reg = predict_reg_onnx(
        sessions['reg'], X_reg)

    # Clustering
    cluster_labels, cluster_probs = predict_cluster_onnx(
        sessions['cluster'], X_cls)

    return {
        'classification': {
            'prediction':    int(y_pred_cls[0]),
            'label':         'Default' if y_pred_cls[0] == 1 else 'Non-Default',
            'decision':      'REJECTED' if y_pred_cls[0] == 1 else 'APPROVED',
            'probability':   round(float(y_prob_cls[0]), 4),
            'threshold':     BEST_THRESHOLD_CLS
        },
        'regression': {
            'predicted_interest_rate': round(float(y_pred_reg[0]), 4)
        },
        'clustering': {
            'cluster_id':    int(cluster_labels[0]),
            'cluster_label': CLUSTER_LABELS['cluster_GMM'].get(
                int(cluster_labels[0]), f'Cluster {cluster_labels[0]}'),
            'probabilities': {
                'High Value Borrower': round(float(cluster_probs[0][0]), 4),
                'Standard Borrower':   round(float(cluster_probs[0][1]), 4)
            }
        }
    }

# PRINT PREDICTION
def print_prediction(prediction):
    """Pretty print full prediction output."""
    print("LOAN RISK PREDICTION")

    cls = prediction['classification']
    reg = prediction['regression']
    clu = prediction['clustering']

    print(f"\n--- Loan Decision ---")
    print(f"  Decision     : {cls['decision']}")
    print(f"  Default Prob : {cls['probability']}")
    print(f"  Threshold    : {cls['threshold']}")

    print(f"\n--- Interest Rate ---")
    print(f"  Predicted Rate : {reg['predicted_interest_rate']}%")

    print(f"\n--- Borrower Profile ---")
    print(f"  Cluster    : {clu['cluster_label']}")
    print(f"  High Value : {clu['probabilities']['High Value Borrower']}")
    print(f"  Standard   : {clu['probabilities']['Standard Borrower']}")

# MAIN
if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    BASE_DIR = Path(__file__).resolve().parent.parent
    ONNX_DIR = str(BASE_DIR / 'models_onnx')
    DATA_DIR = str(BASE_DIR / 'data' / 'engineered_data')

    # Load sessions
    print("Loading ONNX sessions...")
    sessions = load_onnx_sessions(ONNX_DIR)

    # Load sample data
    X_val_cls = pd.read_csv(
        f'{DATA_DIR}/X_val_cls_engineered.csv')
    X_val_reg = pd.read_csv(
        f'{DATA_DIR}/X_val_reg_engineered.csv')

    # Single prediction
    print("\nRunning single prediction...")
    prediction = predict_all_onnx(
        sessions,
        X_val_cls.iloc[[0]],
        X_val_reg.iloc[[0]])

    print_prediction(prediction)

    # Batch prediction
    print("\nRunning batch prediction (5 samples)...")
    y_pred_cls, y_prob_cls = predict_cls_onnx(
        sessions['cls'], X_val_cls.iloc[:5])
    y_pred_reg = predict_reg_onnx(
        sessions['reg'], X_val_reg.iloc[:5])
    cluster_labels, cluster_probs = predict_cluster_onnx(
        sessions['cluster'], X_val_cls.iloc[:5])

    print(f"Cls predictions : {y_pred_cls}")
    print(f"Cls probs       : {y_prob_cls.round(4)}")
    print(f"Reg predictions : {y_pred_reg.round(4)}")
    print(f"Cluster labels  : {cluster_labels}")

    print("\n onnx_inference.py complete")