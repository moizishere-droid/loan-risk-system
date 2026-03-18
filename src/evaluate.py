import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    # Classification
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc,
    # Regression
    r2_score, mean_squared_error, mean_absolute_error,
    # Clustering
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA


# CONSTANTS
BEST_THRESHOLD_CLS = 0.6


# PREDICTION FUNCTIONS

def predict_cls(model, X):
    """
    Predict classification with best threshold.
    Args:
        model: trained classification model
        X: features dataframe
    Returns:
        tuple: (y_pred, y_prob)
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= BEST_THRESHOLD_CLS).astype(int)
    return y_pred, y_prob


def predict_reg(model, X):
    """
    Predict regression values.
    Args:
        model: trained regression model
        X: features dataframe
    Returns:
        np.array: predictions
    """
    return model.predict(X)


def predict_cluster(model, X, model_name):
    """
    Predict cluster labels.
    Args:
        model: trained clustering model
        X: features dataframe
        model_name: model name string
    Returns:
        np.array: cluster labels
    """
    return model.predict(X)


# EVALUATION FUNCTIONS

def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Evaluate classification model.
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_prob: predicted probabilities (optional)
    Returns:
        dict: evaluation metrics
    """
    results = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall':    recall_score(y_true, y_pred),
    }
    if y_prob is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_prob)
    return results


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression model.
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        dict: evaluation metrics
    """
    return {
        'r2':   r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae':  mean_absolute_error(y_true, y_pred)
    }


def evaluate_clustering(X, labels):
    """
    Evaluate clustering model.
    Args:
        X: features dataframe
        labels: cluster labels
    Returns:
        dict: evaluation metrics
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {
            'silhouette':        None,
            'davies_bouldin':    None,
            'calinski_harabasz': None,
            'n_clusters':        n_clusters
        }
    return {
        'silhouette':        silhouette_score(X, labels),
        'davies_bouldin':    davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'n_clusters':        n_clusters
    }


# CLASSIFICATION PLOTS

def plot_confusion_matrix(cls_models, X, y_true, save_path=None):
    """
    Plot confusion matrix for all classification models.
    Args:
        cls_models: dict of {name: model}
        X: features
        y_true: true labels
        save_path: path to save figure
    """
    n = len(cls_models)
    fig, axes = plt.subplots(
        (n + 1) // 2, 2,
        figsize=(14, 5 * ((n + 1) // 2)))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(cls_models.items()):
        y_pred, _ = predict_cls(model, X)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm, annot=True, fmt='d',
            ax=axes[idx], cmap='Blues',
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default'])

        axes[idx].set_title(f'{name}', fontsize=13)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.suptitle(
        f'Confusion Matrix — All Models (Threshold={BEST_THRESHOLD_CLS})',
        fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
    plt.show()


def plot_roc_curves(cls_models, X, y_true, save_path=None):
    """
    Plot ROC curves for all classification models.
    Args:
        cls_models: dict of {name: model}
        X: features
        y_true: true labels
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, model in cls_models.items():
        _, y_prob = predict_cls(model, X)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'ROC Curve — All Models (Threshold={BEST_THRESHOLD_CLS})',
        fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f" Saved ROC curve: {save_path}")
    plt.show()


# REGRESSION PLOTS

def plot_residuals(reg_models, X, y_true, save_path=None):
    """
    Plot residuals for all regression models.
    Args:
        reg_models: dict of {name: model}
        X: features
        y_true: true values
        save_path: path to save figure
    """
    n = len(reg_models)
    fig, axes = plt.subplots(
        (n + 1) // 2, 2,
        figsize=(14, 5 * ((n + 1) // 2)))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(reg_models.items()):
        y_pred = predict_reg(model, X)
        residuals = y_true.values.ravel() - y_pred

        axes[idx].scatter(y_pred, residuals,
                          alpha=0.3, color='steelblue', s=10)
        axes[idx].axhline(y=0, color='red',
                          linestyle='--', linewidth=1.5)
        axes[idx].set_xlabel('Predicted Values')
        axes[idx].set_ylabel('Residuals')
        axes[idx].set_title(f'{name}')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Residual Plot — All Regression Models',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f" Saved residual plot: {save_path}")
    plt.show()


def plot_actual_vs_predicted(reg_models, X, y_true, save_path=None):
    """
    Plot actual vs predicted for all regression models.
    Args:
        reg_models: dict of {name: model}
        X: features
        y_true: true values
        save_path: path to save figure
    """
    n = len(reg_models)
    fig, axes = plt.subplots(
        (n + 1) // 2, 2,
        figsize=(14, 5 * ((n + 1) // 2)))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(reg_models.items()):
        y_pred = predict_reg(model, X)
        y_true_vals = y_true.values.ravel()

        axes[idx].scatter(y_true_vals, y_pred,
                          alpha=0.3, color='steelblue', s=10)

        min_val = min(y_true_vals.min(), y_pred.min())
        max_val = max(y_true_vals.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val],
                       'r--', linewidth=1.5, label='Perfect Prediction')

        axes[idx].set_xlabel('Actual Values')
        axes[idx].set_ylabel('Predicted Values')
        axes[idx].set_title(f'{name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Actual vs Predicted — All Regression Models',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f" Saved actual vs predicted: {save_path}")
    plt.show()


# CLUSTERING PLOTS

def plot_cluster_pca(cluster_models, X, save_path=None):
    """
    Plot PCA visualization for all clustering models.
    Args:
        cluster_models: dict of {name: model}
        X: features
        save_path: path to save figure
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    n = len(cluster_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(cluster_models.items()):
        labels = predict_cluster(model, X, name)

        scatter = axes[idx].scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap='viridis',
            alpha=0.4, s=10)

        axes[idx].set_title(f'{name}', fontsize=13)
        axes[idx].set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
        axes[idx].set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
        plt.colorbar(scatter, ax=axes[idx], label='Cluster')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Cluster Visualization — PCA 2D',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f" Saved PCA plot: {save_path}")
    plt.show()


# SAVE RESULTS

def save_results(cls_val_results, cls_test_results,
                 reg_val_results, reg_test_results,
                 cluster_val_results, cluster_test_results,
                 save_dir):
    """
    Save all val and test results to JSON and CSV.
    Args:
        cls_val_results: classification val results dict
        cls_test_results: classification test results dict
        reg_val_results: regression val results dict
        reg_test_results: regression test results dict
        cluster_val_results: clustering val results dict
        cluster_test_results: clustering test results dict
        save_dir: directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save JSON — val and test together
    all_results = {
        'classification': {
            'val_results':    cls_val_results,
            'test_results':   cls_test_results,
            'best_threshold': BEST_THRESHOLD_CLS
        },
        'regression': {
            'val_results':  reg_val_results,
            'test_results': reg_test_results
        },
        'clustering': {
            'val_results':  cluster_val_results,
            'test_results': cluster_test_results
        }
    }

    json_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    # Save val CSVs
    pd.DataFrame(cls_val_results).T.round(4).to_csv(
        os.path.join(save_dir, 'cls_val_results.csv'))
    pd.DataFrame(reg_val_results).T.round(4).to_csv(
        os.path.join(save_dir, 'reg_val_results.csv'))
    pd.DataFrame(cluster_val_results).T.round(4).to_csv(
        os.path.join(save_dir, 'cluster_val_results.csv'))

    # Save test CSVs
    pd.DataFrame(cls_test_results).T.round(4).to_csv(
        os.path.join(save_dir, 'cls_test_results.csv'))
    pd.DataFrame(reg_test_results).T.round(4).to_csv(
        os.path.join(save_dir, 'reg_test_results.csv'))
    pd.DataFrame(cluster_test_results).T.round(4).to_csv(
        os.path.join(save_dir, 'cluster_test_results.csv'))

    print("All val and test results saved")
    print(f"   - {save_dir}/evaluation_results.json")
    print(f"   - {save_dir}/cls_val_results.csv")
    print(f"   - {save_dir}/reg_val_results.csv")
    print(f"   - {save_dir}/cluster_val_results.csv")
    print(f"   - {save_dir}/cls_test_results.csv")
    print(f"   - {save_dir}/reg_test_results.csv")
    print(f"   - {save_dir}/cluster_test_results.csv")

# FULL EVALUATION RUN

def run_full_evaluation(cls_models, reg_models, cluster_models,
                        X_val_cls, y_val_cls,
                        X_test_cls, y_test_cls,
                        X_val_reg, y_val_reg,
                        X_test_reg, y_test_reg,
                        figures_dir, results_dir):
    """
    Run complete evaluation pipeline for all models.
    """
    print("FULL MODEL EVALUATION")

    # --- Classification ---
    print("\n--- Classification ---")
    cls_val_results  = {}
    cls_test_results = {}

    for name, model in cls_models.items():
        y_pred_val,  y_prob_val  = predict_cls(model, X_val_cls)
        y_pred_test, y_prob_test = predict_cls(model, X_test_cls)
        cls_val_results[name]  = evaluate_classification(
            y_val_cls,  y_pred_val,  y_prob_val)
        cls_test_results[name] = evaluate_classification(
            y_test_cls, y_pred_test, y_prob_test)
        print(f"  {name}: Val F1={cls_val_results[name]['f1']:.4f} "
              f"Test F1={cls_test_results[name]['f1']:.4f}")

    # --- Regression ---
    print("\n--- Regression ---")
    reg_val_results  = {}
    reg_test_results = {}

    for name, model in reg_models.items():
        y_pred_val  = predict_reg(model, X_val_reg)
        y_pred_test = predict_reg(model, X_test_reg)
        reg_val_results[name]  = evaluate_regression(y_val_reg,  y_pred_val)
        reg_test_results[name] = evaluate_regression(y_test_reg, y_pred_test)
        print(f"  {name}: Val R2={reg_val_results[name]['r2']:.4f} "
              f"Test R2={reg_test_results[name]['r2']:.4f}")

    # --- Clustering ---
    print("\n--- Clustering ---")
    cluster_val_results  = {}
    cluster_test_results = {}

    for name, model in cluster_models.items():
        labels_val  = predict_cluster(model, X_val_cls, name)
        labels_test = predict_cluster(model, X_test_cls, name)
        cluster_val_results[name]  = evaluate_clustering(X_val_cls,  labels_val)
        cluster_test_results[name] = evaluate_clustering(X_test_cls, labels_test)
        print(f"  {name}: Val Sil={cluster_val_results[name]['silhouette']:.4f} "
              f"Test Sil={cluster_test_results[name]['silhouette']:.4f}")

    # --- Plots ---
    print("\n--- Generating Plots ---")
    os.makedirs(figures_dir, exist_ok=True)

    plot_confusion_matrix(
        cls_models, X_val_cls, y_val_cls,
        save_path=os.path.join(figures_dir, 'confusion_matrix_all.png'))

    plot_roc_curves(
        cls_models, X_val_cls, y_val_cls,
        save_path=os.path.join(figures_dir, 'roc_curve_all.png'))

    plot_residuals(
        reg_models, X_val_reg, y_val_reg,
        save_path=os.path.join(figures_dir, 'residual_plot_all.png'))

    plot_actual_vs_predicted(
        reg_models, X_val_reg, y_val_reg,
        save_path=os.path.join(figures_dir, 'actual_vs_predicted_all.png'))

    plot_cluster_pca(
        cluster_models, X_val_cls,
        save_path=os.path.join(figures_dir, 'cluster_pca_visualization.png'))

    # --- Save Results ---
    save_results(cls_val_results,  cls_test_results,
            reg_val_results,  reg_test_results,
            cluster_val_results, cluster_test_results,
            results_dir)
    
    print("\n Full evaluation complete")

    return {
        'cls':     {'val': cls_val_results,     'test': cls_test_results},
        'reg':     {'val': reg_val_results,     'test': reg_test_results},
        'cluster': {'val': cluster_val_results, 'test': cluster_test_results}
    }


# MAIN
if __name__ == "__main__":
    from model import load_all_models, load_engineered_splits
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    # Load models and data
    models = load_all_models()
    data   = load_engineered_splits()

    # Run evaluation
    results = run_full_evaluation(
        cls_models     = models['cls'],
        reg_models     = models['reg'],
        cluster_models = models['cluster'],
        X_val_cls  = data['cls']['X_val_cls'],
        y_val_cls  = data['cls']['y_val_cls'],
        X_test_cls = data['cls']['X_test_cls'],
        y_test_cls = data['cls']['y_test_cls'],
        X_val_reg  = data['reg']['X_val_reg'],
        y_val_reg  = data['reg']['y_val_reg'],
        X_test_reg = data['reg']['X_test_reg'],
        y_test_reg = data['reg']['y_test_reg'],
        figures_dir = str(BASE_DIR / 'logs' / 'reports' / 'figures'),
        results_dir = str(BASE_DIR / 'models' / 'Project_Parameter_Files')
    )

    print("\nevaluate.py complete")