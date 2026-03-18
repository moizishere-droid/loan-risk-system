import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# CONSTANTS
BEST_THRESHOLD_CLS = 0.6

FINAL_MODELS = {
    'classification': {
        'model_name':  'cls_LightGBM',
        'model_file':  'cls_LightGBM.joblib',
        'threshold':    BEST_THRESHOLD_CLS,
        'metrics': {
            'test_f1':        0.8390,
            'test_auc':       0.9471,
            'test_accuracy':  0.9358,
            'test_recall':    0.7674,
            'test_precision': 0.9253
        },
        'reason': 'Wins 7/10 metrics. Higher recall critical for loan risk.'
    },
    'regression': {
        'model_name': 'reg_RandomForest',
        'model_file': 'reg_RandomForest.joblib',
        'metrics': {
            'test_r2':   0.9077,
            'test_rmse': 0.9811,
            'test_mae':  0.7697
        },
        'reason': 'Wins all 6/6 metrics. No debate.'
    },
    'clustering': {
        'model_name': 'cluster_GMM',
        'model_file': 'cluster_GMM.joblib',
        'metrics': {
            'test_silhouette':        0.2196,
            'test_davies_bouldin':    4.3057,
            'test_calinski_harabasz': 118.08
        },
        'cluster_labels': {
            0: 'High Value Borrower',
            1: 'Standard Borrower'
        },
        'reason': 'Best silhouette + soft probabilities for production.'
    }
}


# LOAD RESULTS
def load_all_results(param_dir):
    """
    Load all evaluation results from CSV files.
    Args:
        param_dir: path to Project_Parameter_Files directory
    Returns:
        dict: all val and test results
    """
    return {
        'cls': {
            'val':  pd.read_csv(
                os.path.join(param_dir, 'cls_val_results.csv'),
                index_col=0),
            'test': pd.read_csv(
                os.path.join(param_dir, 'cls_test_results.csv'),
                index_col=0)
        },
        'reg': {
            'val':  pd.read_csv(
                os.path.join(param_dir, 'reg_val_results.csv'),
                index_col=0),
            'test': pd.read_csv(
                os.path.join(param_dir, 'reg_test_results.csv'),
                index_col=0)
        },
        'cluster': {
            'val':  pd.read_csv(
                os.path.join(param_dir, 'cluster_val_results.csv'),
                index_col=0),
            'test': pd.read_csv(
                os.path.join(param_dir, 'cluster_test_results.csv'),
                index_col=0)
        }
    }


# COMPARISON TABLES
def get_cls_comparison(results):
    """
    Build classification comparison table.
    Args:
        results: dict from load_all_results()
    Returns:
        pd.DataFrame: comparison table
    """
    return pd.DataFrame({
        'Val F1':         results['cls']['val']['f1'],
        'Test F1':        results['cls']['test']['f1'],
        'Val AUC':        results['cls']['val']['roc_auc'],
        'Test AUC':       results['cls']['test']['roc_auc'],
        'Val Accuracy':   results['cls']['val']['accuracy'],
        'Test Accuracy':  results['cls']['test']['accuracy'],
        'Val Precision':  results['cls']['val']['precision'],
        'Test Precision': results['cls']['test']['precision'],
        'Val Recall':     results['cls']['val']['recall'],
        'Test Recall':    results['cls']['test']['recall'],
    }).round(4)


def get_reg_comparison(results):
    """
    Build regression comparison table.
    Args:
        results: dict from load_all_results()
    Returns:
        pd.DataFrame: comparison table
    """
    return pd.DataFrame({
        'Val R2':    results['reg']['val']['r2'],
        'Test R2':   results['reg']['test']['r2'],
        'Val RMSE':  results['reg']['val']['rmse'],
        'Test RMSE': results['reg']['test']['rmse'],
        'Val MAE':   results['reg']['val']['mae'],
        'Test MAE':  results['reg']['test']['mae'],
    }).round(4)


def get_cluster_comparison(results):
    """
    Build clustering comparison table.
    Args:
        results: dict from load_all_results()
    Returns:
        pd.DataFrame: comparison table
    """
    return pd.DataFrame({
        'Val Silhouette':         results['cluster']['val']['silhouette'],
        'Test Silhouette':        results['cluster']['test']['silhouette'],
        'Val Davies Bouldin':     results['cluster']['val']['davies_bouldin'],
        'Test Davies Bouldin':    results['cluster']['test']['davies_bouldin'],
        'Val Calinski Harabasz':  results['cluster']['val']['calinski_harabasz'],
        'Test Calinski Harabasz': results['cluster']['test']['calinski_harabasz'],
    }).round(4)


# BEST MODEL PER METRIC
def get_best_per_metric(comparison_df, higher_better_cols,
                        lower_better_cols):
    """
    Get best model per metric.
    Args:
        comparison_df: comparison dataframe
        higher_better_cols: list of columns where higher is better
        lower_better_cols: list of columns where lower is better
    Returns:
        dict: best model per metric
    """
    best = {}
    for col in higher_better_cols:
        if col in comparison_df.columns:
            best[col] = {
                'model': comparison_df[col].idxmax(),
                'value': comparison_df[col].max(),
                'direction': '↑ higher better'
            }
    for col in lower_better_cols:
        if col in comparison_df.columns:
            best[col] = {
                'model': comparison_df[col].idxmin(),
                'value': comparison_df[col].min(),
                'direction': '↓ lower better'
            }
    return best


# LOAD FINAL MODELS
def load_final_models(models_dir):
    """
    Load only the final selected models.
    Args:
        models_dir: path to models directory
    Returns:
        dict: final models
    """
    import joblib

    final = {}
    for task, config in FINAL_MODELS.items():
        model_path = os.path.join(models_dir, config['model_file'])
        if os.path.exists(model_path):
            final[task] = joblib.load(model_path)
            print(f"Loaded {config['model_name']}")
        else:
            print(f"Not found: {model_path}")
    return final


# SAVE CONFIG
def save_final_config(param_dir):
    """
    Save final model config to JSON.
    Args:
        param_dir: path to Project_Parameter_Files directory
    """
    # Convert int keys to str for JSON
    config = FINAL_MODELS.copy()
    config['clustering']['cluster_labels'] = {
        str(k): v for k, v in
        config['clustering']['cluster_labels'].items()
    }

    path = os.path.join(param_dir, 'final_model_config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Final model config saved: {path}")


def load_final_config(param_dir):
    """
    Load final model config from JSON.
    Args:
        param_dir: path to Project_Parameter_Files directory
    Returns:
        dict: final model config
    """
    path = os.path.join(param_dir, 'final_model_config.json')
    with open(path, 'r') as f:
        return json.load(f)


# PRINT SUMMARY
def print_final_summary():
    """Print final model selection summary."""
    print("FINAL SELECTED MODELS")
    for task, config in FINAL_MODELS.items():
        print(f"\n  Task    : {task.capitalize()}")
        print(f"  Model   : {config['model_name']}")
        metrics_str = ' | '.join(
            [f"{k.replace('test_', '').upper()}="
             f"{v}" for k, v in config['metrics'].items()])
        print(f"  Metrics : {metrics_str}")
        print(f"  Reason  : {config['reason']}")


# MAIN
if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR  = Path(__file__).resolve().parent.parent
    PARAM_DIR = str(BASE_DIR / 'models' / 'Project_Parameter_Files' / 'Evalution_result')
    MODEL_DIR = str(BASE_DIR / 'models' / "Final_Models")

    # Load results
    print("Loading evaluation results...")
    results = load_all_results(PARAM_DIR)

    # Print comparison tables
    print("\n--- Classification Comparison ---")
    print(get_cls_comparison(results).to_string())

    print("\n--- Regression Comparison ---")
    print(get_reg_comparison(results).to_string())

    print("\n--- Clustering Comparison ---")
    print(get_cluster_comparison(results).to_string())

    # Save config
    save_final_config(PARAM_DIR)

    # Print summary
    print_final_summary()

    # Load final models
    print("\nLoading final models...")
    final_models = load_final_models(MODEL_DIR)

    print("\nmodel_selection.py complete")