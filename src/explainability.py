import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

# CONSTANTS

BEST_THRESHOLD_CLS = 0.6

# Cluster labels
CLUSTER_LABELS = {
    'cluster_GMM': {
        0: 'High Value Borrower',
        1: 'Standard Borrower'
    },
    'cluster_KMeans': {
        0: 'Standard Borrower',
        1: 'High Value Borrower'
    }
}

# Human readable feature names
FEATURE_NAMES_READABLE = {
    'loan_percent_income':              'Loan to Income Ratio',
    'loan_int_rate':                    'Interest Rate',
    'person_income':                    'Annual Income',
    'person_home_ownership_OWN':        'Home Ownership (Own)',
    'person_home_ownership_RENT':       'Home Ownership (Rent)',
    'loan_intent_DEBTCONSOLIDATION':    'Loan Intent (Debt Consolidation)',
    'loan_intent_VENTURE':              'Loan Intent (Venture)',
    'loan_intent_HOMEIMPROVEMENT':      'Loan Intent (Home Improvement)',
    'loan_grade_x_loan_int_rate':       'Loan Grade x Interest Rate',
    'loan_grade_x_loan_percent_income': 'Loan Grade x Loan Income Ratio',
    'loan_grade_x_loan_amnt':           'Loan Grade x Loan Amount',
    'loan_grade':                       'Loan Grade',
    'loan_amnt':                        'Loan Amount',
    'debt_burden_score':                'Debt Burden Score',
    'loan_to_age_ratio':                'Loan to Age Ratio',
    'income_per_emp_year':              'Income per Employment Year',
    'person_age':                       'Age',
    'person_emp_length':                'Employment Length',
    'income_to_loan_ratio':             'Income to Loan Ratio',
    'cb_person_default_on_file':        'Previous Default on File',
    'person_age_x_person_emp_length':   'Age x Employment Length'
}


# SHAP EXPLAINER SETUP
def build_cls_explainer(model, X_background):
    """
    Build SHAP TreeExplainer for classification model.
    Args:
        model: trained classification model
        X_background: background dataset for SHAP
    Returns:
        shap.TreeExplainer
    """
    return shap.TreeExplainer(model, X_background)


def build_reg_explainer(model, X_background):
    """
    Build SHAP TreeExplainer for regression model.
    Args:
        model: trained regression model
        X_background: background dataset for SHAP
    Returns:
        shap.TreeExplainer
    """
    return shap.TreeExplainer(model, X_background)


# SHAP VALUES
def get_cls_shap_values(explainer, X):
    """
    Get SHAP values for classification — Default class only.
    Args:
        explainer: SHAP TreeExplainer
        X: input features
    Returns:
        tuple: (shap_values_default, expected_value)
    """
    shap_values = explainer.shap_values(X)

    # Handle both list (binary) and array (multiclass) output
    if isinstance(shap_values, list):
        return shap_values[1], explainer.expected_value[1]
    return shap_values, explainer.expected_value


def get_reg_shap_values(explainer, X):
    """
    Get SHAP values for regression.
    Args:
        explainer: SHAP TreeExplainer
        X: input features
    Returns:
        tuple: (shap_values, expected_value)
    """
    shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value


# SINGLE PREDICTION EXPLANATION
def explain_cls_prediction(model, explainer, X_single,
                           feature_names, top_n=5):
    """
    Explain a single classification prediction.
    Args:
        model: trained classification model
        explainer: SHAP TreeExplainer
        X_single: single sample dataframe (1 row)
        feature_names: list of feature names
        top_n: number of top features to return
    Returns:
        dict: explanation with prediction, probability, reasons
    """
    # Prediction
    y_prob = model.predict_proba(X_single)[:, 1][0]
    y_pred = int(y_prob >= BEST_THRESHOLD_CLS)

    # SHAP values
    shap_vals, expected_val = get_cls_shap_values(
        explainer, X_single)
    shap_vals_single = shap_vals[0]

    # Top contributing features
    feat_shap = pd.Series(
        shap_vals_single, index=feature_names)
    top_increase = feat_shap.nlargest(top_n)
    top_decrease = feat_shap.nsmallest(top_n)

    # Human readable reasons
    reasons_for_default = []
    for feat, val in top_increase.items():
        readable = FEATURE_NAMES_READABLE.get(feat, feat)
        feat_val = X_single[feat].values[0]
        reasons_for_default.append({
            'feature':       feat,
            'readable_name': readable,
            'feature_value': round(float(feat_val), 4),
            'shap_impact':   round(float(val), 4),
            'direction':     'increases default risk'
        })

    reasons_against_default = []
    for feat, val in top_decrease.items():
        readable = FEATURE_NAMES_READABLE.get(feat, feat)
        feat_val = X_single[feat].values[0]
        reasons_against_default.append({
            'feature':       feat,
            'readable_name': readable,
            'feature_value': round(float(feat_val), 4),
            'shap_impact':   round(float(val), 4),
            'direction':     'decreases default risk'
        })

    return {
        'prediction':             y_pred,
        'prediction_label':       'Default' if y_pred == 1 else 'Non-Default',
        'default_probability':    round(float(y_prob), 4),
        'threshold':              BEST_THRESHOLD_CLS,
        'decision':               'REJECTED' if y_pred == 1 else 'APPROVED',
        'reasons_for_default':    reasons_for_default,
        'reasons_against_default': reasons_against_default,
        'base_rate':              round(float(expected_val), 4)
    }


def explain_reg_prediction(model, explainer, X_single,
                           feature_names, top_n=5):
    """
    Explain a single regression prediction.
    Args:
        model: trained regression model
        explainer: SHAP TreeExplainer
        X_single: single sample dataframe (1 row)
        feature_names: list of feature names
        top_n: number of top features to return
    Returns:
        dict: explanation with prediction and reasons
    """
    # Prediction
    y_pred = model.predict(X_single)[0]

    # SHAP values
    shap_vals, expected_val = get_reg_shap_values(
        explainer, X_single)
    shap_vals_single = shap_vals[0]

    # Top contributing features
    feat_shap = pd.Series(
        shap_vals_single, index=feature_names)
    top_increase = feat_shap.nlargest(top_n)
    top_decrease = feat_shap.nsmallest(top_n)

    # Human readable reasons
    reasons_high_rate = []
    for feat, val in top_increase.items():
        readable = FEATURE_NAMES_READABLE.get(feat, feat)
        feat_val = X_single[feat].values[0]
        reasons_high_rate.append({
            'feature':       feat,
            'readable_name': readable,
            'feature_value': round(float(feat_val), 4),
            'shap_impact':   round(float(val), 4),
            'direction':     'increases interest rate'
        })

    reasons_low_rate = []
    for feat, val in top_decrease.items():
        readable = FEATURE_NAMES_READABLE.get(feat, feat)
        feat_val = X_single[feat].values[0]
        reasons_low_rate.append({
            'feature':       feat,
            'readable_name': readable,
            'feature_value': round(float(feat_val), 4),
            'shap_impact':   round(float(val), 4),
            'direction':     'decreases interest rate'
        })

    return {
        'predicted_interest_rate': round(float(y_pred), 4),
        'base_rate':               round(float(expected_val), 4),
        'reasons_high_rate':       reasons_high_rate,
        'reasons_low_rate':        reasons_low_rate
    }


def explain_cluster_prediction(model, X_single, model_name):
    """
    Explain a single cluster prediction.
    Args:
        model: trained clustering model
        X_single: single sample dataframe (1 row)
        model_name: name of clustering model
    Returns:
        dict: cluster label and profile
    """
    cluster_id = int(model.predict(X_single)[0])
    label = CLUSTER_LABELS.get(model_name, {}).get(
        cluster_id, f'Cluster {cluster_id}')

    # Profile features
    profile_features = [
        'loan_percent_income', 'person_income',
        'loan_amnt', 'loan_int_rate',
        'person_age', 'person_emp_length'
    ]

    profile = {}
    for feat in profile_features:
        if feat in X_single.columns:
            readable = FEATURE_NAMES_READABLE.get(feat, feat)
            profile[readable] = round(
                float(X_single[feat].values[0]), 4)

    return {
        'cluster_id':    cluster_id,
        'cluster_label': label,
        'profile':       profile
    }


# PLOTS
def plot_shap_waterfall(shap_vals_single, expected_val,
                        X_single, feature_names,
                        title, save_path=None):
    """
    Plot SHAP waterfall for a single prediction.
    Args:
        shap_vals_single: SHAP values for single sample
        expected_val: base expected value
        X_single: single sample dataframe
        feature_names: list of feature names
        title: plot title
        save_path: path to save figure
    """
    explanation = shap.Explanation(
        values        = shap_vals_single,
        base_values   = expected_val,
        data          = X_single.values[0],
        feature_names = feature_names)

    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(explanation, show=False)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f" Saved waterfall: {save_path}")
    plt.show()


def plot_shap_summary(shap_values, X, title,
                      plot_type='dot', save_path=None):
    """
    Plot SHAP summary for global explanation.
    Args:
        shap_values: SHAP values array
        X: features dataframe
        title: plot title
        plot_type: 'dot' or 'bar'
        save_path: path to save figure
    """
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X,
        plot_type=plot_type,
        max_display=15,
        show=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved summary plot: {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names,
                            title, top_n=10,
                            color='Blues_r', save_path=None):
    """
    Plot built-in feature importance for tree models.
    Args:
        model: trained tree model
        feature_names: list of feature names
        title: plot title
        top_n: number of top features
        color: seaborn palette
        save_path: path to save figure
    """
    import seaborn as sns

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index,
                palette=color)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f" Saved feature importance: {save_path}")
    plt.show()


# PRINT EXPLANATION
def print_cls_explanation(explanation):
    """Pretty print classification explanation."""
    print("LOAN APPLICATION DECISION")
    print(f"Decision          : {explanation['decision']}")
    print(f"Default Probability: {explanation['default_probability']}")
    print(f"Threshold         : {explanation['threshold']}")

    print(f"\n--- Top Reasons FOR Default Risk ---")
    for r in explanation['reasons_for_default']:
        print(f"  {r['readable_name']:<35} "
              f"impact: +{r['shap_impact']:.3f}")

    print(f"\n--- Top Reasons AGAINST Default Risk ---")
    for r in explanation['reasons_against_default']:
        print(f"  {r['readable_name']:<35} "
              f"impact: {r['shap_impact']:.3f}")
    print("=" * 55)


def print_reg_explanation(explanation):
    """Pretty print regression explanation."""
    print("INTEREST RATE PREDICTION")
    print(f"Predicted Rate : {explanation['predicted_interest_rate']}%")
    print(f"Average Rate   : {explanation['base_rate']}%")

    print(f"\n--- Top Reasons FOR High Rate ---")
    for r in explanation['reasons_high_rate']:
        print(f"  {r['readable_name']:<35} "
              f"impact: +{r['shap_impact']:.3f}")

    print(f"\n--- Top Reasons FOR Low Rate ---")
    for r in explanation['reasons_low_rate']:
        print(f"  {r['readable_name']:<35} "
              f"impact: {r['shap_impact']:.3f}")
    print("=" * 55)


def print_cluster_explanation(explanation):
    """Pretty print cluster explanation."""
    print("BORROWER PROFILE")
    print(f"Cluster ID    : {explanation['cluster_id']}")
    print(f"Cluster Label : {explanation['cluster_label']}")
    print(f"\nProfile:")
    for feat, val in explanation['profile'].items():
        print(f"  {feat:<35} {val:.4f}")


# MAIN

if __name__ == "__main__":
    from model import load_all_models
    from pathlib import Path
    import pandas as pd

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data' / 'engineered_data'

    # Load models
    models = load_all_models()
    cls_model     = models['cls']['LightGBM']
    reg_model     = models['reg']['RandomForest']
    cluster_model = models['cluster']['GMM']

    # Load data
    X_train_cls = pd.read_csv(
        DATA_DIR / 'X_train_cls_engineered.csv')
    X_val_cls   = pd.read_csv(
        DATA_DIR / 'X_val_cls_engineered.csv')
    X_train_reg = pd.read_csv(
        DATA_DIR / 'X_train_reg_engineered.csv')
    X_val_reg   = pd.read_csv(
        DATA_DIR / 'X_val_reg_engineered.csv')

    # Build explainers
    print("Building SHAP explainers...")
    cls_explainer = build_cls_explainer(cls_model, X_train_cls)
    reg_explainer = build_reg_explainer(reg_model, X_train_reg)
    print("Explainers built")

    # Single sample explanation — first val sample
    X_sample_cls = X_val_cls.iloc[[0]]
    X_sample_reg = X_val_reg.iloc[[0]]

    # Classification explanation
    cls_exp = explain_cls_prediction(cls_model, cls_explainer,
        X_sample_cls, X_val_cls.columns.tolist())
    print_cls_explanation(cls_exp)

    # Regression explanation
    reg_exp = explain_reg_prediction(reg_model, reg_explainer,
        X_sample_reg, X_val_reg.columns.tolist())
    print_reg_explanation(reg_exp)

    # Cluster explanationD
    cluster_exp = explain_cluster_prediction(
        cluster_model, X_sample_cls, 'cluster_GMM')
    print_cluster_explanation(cluster_exp)

    print("\n explainability.py complete")