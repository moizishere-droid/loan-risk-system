import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from api.model_loader import (
    get_cls_session, get_reg_session, get_cluster_session,
    get_cls_pipeline, get_reg_pipeline,
    get_cls_fe_pipeline, get_reg_fe_pipeline,
    get_cls_explainer, get_reg_explainer,
    get_cls_features, get_reg_features
)


# CONSTANTS
BEST_THRESHOLD_CLS = 0.6
CLUSTER_LABELS = {
    0: 'High Value Borrower',
    1: 'Standard Borrower'
}
CLS_PREPROCESSED_COLS = [
    'num__person_age', 'num__person_income', 'num__person_emp_length',
    'num__loan_amnt', 'num__loan_int_rate', 'num__loan_percent_income',
    'ord__loan_grade',
    'ohe__person_home_ownership_MORTGAGE', 'ohe__person_home_ownership_OWN',
    'ohe__person_home_ownership_RENT',
    'ohe__loan_intent_DEBTCONSOLIDATION', 'ohe__loan_intent_EDUCATION',
    'ohe__loan_intent_HOMEIMPROVEMENT', 'ohe__loan_intent_MEDICAL',
    'ohe__loan_intent_PERSONAL', 'ohe__loan_intent_VENTURE',
    'remainder__cb_person_default_on_file'
]
REG_PREPROCESSED_COLS = [
    'num__person_age', 'num__person_income', 'num__person_emp_length',
    'num__loan_amnt', 'num__loan_percent_income',
    'ord__loan_grade',
    'ohe__person_home_ownership_MORTGAGE', 'ohe__person_home_ownership_OWN',
    'ohe__person_home_ownership_RENT',
    'ohe__loan_intent_DEBTCONSOLIDATION', 'ohe__loan_intent_EDUCATION',
    'ohe__loan_intent_HOMEIMPROVEMENT', 'ohe__loan_intent_MEDICAL',
    'ohe__loan_intent_PERSONAL', 'ohe__loan_intent_VENTURE',
    'remainder__loan_status', 'remainder__cb_person_default_on_file'
]
CLS_FE_INPUT_COLS = [
    'person_age', 'person_income', 'person_emp_length', 'loan_grade',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_default_on_file',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
    'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]
REG_FE_INPUT_COLS = [
    'person_age', 'person_income', 'person_emp_length', 'loan_grade',
    'loan_amnt', 'loan_status', 'loan_percent_income',
    'cb_person_default_on_file',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
    'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]


# PLAIN REASONS — MAPS
GRADE_MAP = {
    'A': 'A (Excellent)', 'B': 'B (Good)',     'C': 'C (Fair)',
    'D': 'D (Poor)',      'E': 'E (Very Poor)', 'F': 'F (Bad)',
    'G': 'G (Very Bad)'
}
INTENT_MAP = {
    'DEBTCONSOLIDATION': 'debt consolidation',
    'EDUCATION'        : 'education',
    'HOMEIMPROVEMENT'  : 'home improvement',
    'MEDICAL'          : 'medical expenses',
    'PERSONAL'         : 'personal use',
    'VENTURE'          : 'business venture'
}
READABLE_NAMES = {
    'person_age'                      : 'Age',
    'person_income'                   : 'Annual Income',
    'person_emp_length'               : 'Employment Length',
    'loan_amnt'                       : 'Loan Amount',
    'loan_int_rate'                   : 'Interest Rate',
    'loan_percent_income'             : 'Loan to Income Ratio',
    'loan_grade'                      : 'Loan Grade',
    'cb_person_default_on_file'       : 'Previous Default',
    'person_home_ownership_MORTGAGE'  : 'Home Ownership (Mortgage)',
    'person_home_ownership_OWN'       : 'Home Ownership (Own)',
    'person_home_ownership_RENT'      : 'Home Ownership (Rent)',
    'loan_intent_DEBTCONSOLIDATION'   : 'Loan Intent (Debt Consolidation)',
    'loan_intent_EDUCATION'           : 'Loan Intent (Education)',
    'loan_intent_HOMEIMPROVEMENT'     : 'Loan Intent (Home Improvement)',
    'loan_intent_MEDICAL'             : 'Loan Intent (Medical)',
    'loan_intent_PERSONAL'            : 'Loan Intent (Personal)',
    'loan_intent_VENTURE'             : 'Loan Intent (Venture)',
    'income_to_loan_ratio'            : 'Income to Loan Ratio',
    'loan_to_age_ratio'               : 'Loan to Age Ratio',
    'income_per_emp_year'             : 'Income per Employment Year',
    'loan_grade_x_loan_int_rate'      : 'Loan Grade x Interest Rate',
    'loan_grade_x_loan_percent_income': 'Loan Grade x Loan Income Ratio',
    'loan_grade_x_loan_amnt'          : 'Loan Grade x Loan Amount',
    'person_age_x_person_emp_length'  : 'Age x Employment Length',
    'debt_burden_score'               : 'Debt Burden Score',
}

# HELPER — PREPROCESS --> Ready Features For Onnx Model
def _preprocess(input_dict, pipeline, preprocessed_cols,
                fe_input_cols, fe_pipeline, feature_names):
    """
    Preprocess raw input dict into engineered features.
    Steps:
        1. dict → DataFrame
        2. preprocessing pipeline
        3. strip column prefixes
        4. reorder to FE pipeline fit order
        5. feature engineering pipeline
        6. select final features
    Returns:
        pd.DataFrame: engineered features ready for ONNX
    """
    df    = pd.DataFrame([input_dict])
    X_pre = pipeline.transform(df)
    X_pre = pd.DataFrame(X_pre, columns=preprocessed_cols)
    X_pre.columns = [
        col.split('__')[1] if '__' in col else col
        for col in X_pre.columns
    ]
    X_pre  = X_pre[fe_input_cols]
    X_fe   = fe_pipeline.transform(X_pre)
    X_fe   = pd.DataFrame(X_fe, columns=feature_names)
    return X_fe


# HELPER — INFERENCE --> output from model
def _predict_cls(X_fe):
    """Run classification ONNX inference."""
    session    = get_cls_session()
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: X_fe.values.astype(np.float32)})
    prob       = float(np.array([p[1] for p in outputs[1]])[0])
    decision   = 'REJECTED' if prob >= BEST_THRESHOLD_CLS else 'APPROVED'
    return decision, prob

def _predict_reg(X_fe):
    """Run regression ONNX inference."""
    session    = get_reg_session()
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: X_fe.values.astype(np.float32)})
    return round(float(outputs[0].ravel()[0]), 4)

def _predict_cluster(X_fe):
    """Run clustering ONNX inference."""
    session    = get_cluster_session()
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: X_fe.values.astype(np.float32)})
    cluster_id    = int(outputs[0].ravel()[0])
    cluster_label = CLUSTER_LABELS[cluster_id]
    prob_high     = round(float(outputs[1][0][0]), 4)
    prob_std      = round(float(outputs[1][0][1]), 4)
    return cluster_id, cluster_label, prob_high, prob_std


# HELPER — SHAP
def _get_cls_shap(X_fe):
    """Get SHAP explanations for classification."""
    explainer   = get_cls_explainer()
    cls_features= get_cls_features()
    shap_vals   = explainer.shap_values(X_fe)
    vals        = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    shap_series = pd.Series(vals, index=cls_features)

    reasons_for = [
        {'feature'      : f,
        'readable_name': READABLE_NAMES.get(f, f),
        'feature_value': round(float(X_fe[f].values[0]), 4),
        'shap_impact'  : round(float(v), 4),
        'direction'    : 'increases default risk'}
        for f, v in shap_series.nlargest(5).items()
    ]
    reasons_against = [
        {'feature'      : f,
        'readable_name': READABLE_NAMES.get(f, f),
        'feature_value': round(float(X_fe[f].values[0]), 4),
        'shap_impact'  : round(float(v), 4),
        'direction'    : 'decreases default risk'}
        for f, v in shap_series.nsmallest(5).items()
    ]
    return reasons_for, reasons_against

def _get_reg_shap(X_fe):
    """Get SHAP explanations for regression."""
    explainer   = get_reg_explainer()
    reg_features= get_reg_features()
    shap_vals   = explainer.shap_values(X_fe)
    vals        = shap_vals[0][0] if isinstance(shap_vals, list) else shap_vals[0]
    shap_series = pd.Series(vals, index=reg_features)

    reasons_high = [
        {'feature'      : f,
        'readable_name': READABLE_NAMES.get(f, f),
        'feature_value': round(float(X_fe[f].values[0]), 4),
        'shap_impact'  : round(float(v), 4),
        'direction'    : 'increases interest rate'}
        for f, v in shap_series.nlargest(5).items()
    ]
    reasons_low = [
        {'feature'      : f,
        'readable_name': READABLE_NAMES.get(f, f),
        'feature_value': round(float(X_fe[f].values[0]), 4),
        'shap_impact'  : round(float(v), 4),
        'direction'    : 'decreases interest rate'}
        for f, v in shap_series.nsmallest(5).items()
    ]
    return reasons_high, reasons_low


# HELPER — PLAIN REASONS
def _generate_plain_reasons(decision, default_prob, predicted_rate,
                              shap_for, shap_against,
                              shap_high, shap_low,
                              original_input, workflow):
    """Generate plain English reasons from SHAP + original values."""

    loan_amnt    = original_input.get('loan_amnt', 0)
    loan_grade   = original_input.get('loan_grade', '')
    loan_intent  = original_input.get('loan_intent', '')
    loan_pct_inc = original_input.get('loan_percent_income', 0)
    person_inc   = original_input.get('person_income', 0)
    emp_length   = original_input.get('person_emp_length', 0)
    loan_rate    = original_input.get('loan_int_rate')

    grade_label  = GRADE_MAP.get(loan_grade, loan_grade)
    intent_label = INTENT_MAP.get(loan_intent, loan_intent.lower())
    pct_inc_disp = round(loan_pct_inc * 100, 1)
    rate_val     = loan_rate if loan_rate else predicted_rate

    top_for     = [r['feature'] for r in shap_for]
    top_against = [r['feature'] for r in shap_against]
    top_high    = [r['feature'] for r in shap_high]
    top_low     = [r['feature'] for r in shap_low]

    decision_reasons = []
    rate_reasons     = []

    # Decision summary
    risk_pct = round(default_prob * 100, 1)
    if decision == 'REJECTED':
        decision_reasons.append(
            f"Your loan application was REJECTED with a {risk_pct}% "
            f"estimated default risk (threshold: {round(BEST_THRESHOLD_CLS * 100)}%)."
        )
    else:
        decision_reasons.append(
            f"Your loan application was APPROVED with only a {risk_pct}% "
            f"estimated default risk (threshold: {round(BEST_THRESHOLD_CLS * 100)}%)."
        )

    # Grade
    if any('loan_grade' in f for f in top_for):
        decision_reasons.append(
            f"Your loan grade {grade_label} is a high-risk grade, "
            f"which strongly increases your chance of default."
        )
    elif any('loan_grade' in f for f in top_against):
        decision_reasons.append(
            f"Your loan grade {grade_label} is a low-risk grade, "
            f"which works in your favor."
        )

    # Loan percent income
    if any('loan_percent_income' in f for f in top_for):
        decision_reasons.append(
            f"Your loan amount (${loan_amnt:,.0f}) is {pct_inc_disp}% "
            f"of your annual income (${person_inc:,.0f}), "
            f"which is very high and increases default risk."
        )
    elif any('loan_percent_income' in f for f in top_against):
        decision_reasons.append(
            f"Your loan-to-income ratio of {pct_inc_disp}% is manageable "
            f"relative to your income of ${person_inc:,.0f}."
        )

    # Interest rate
    if rate_val and any('loan_int_rate' in f for f in top_for):
        if decision == 'REJECTED':
            decision_reasons.append(
                f"Your interest rate of {round(rate_val, 2)}% is high, "
                f"which significantly increases default risk."
            )
        else:
            decision_reasons.append(
                f"Your interest rate of {round(rate_val, 2)}% has a "
                f"minor upward effect on default risk."
            )

    # Grade x rate
    if any('loan_grade_x_loan_int_rate' in f for f in top_for):
        decision_reasons.append(
            f"The combination of grade {loan_grade} and a high "
            f"interest rate is the strongest signal for default risk."
        )

    # Home ownership
    if any('person_home_ownership_RENT' in f for f in top_for):
        decision_reasons.append(
            "Renting your home is associated with slightly higher "
            "default risk compared to homeowners."
        )
    elif any('person_home_ownership' in f for f in top_against):
        decision_reasons.append(
            "Owning your home or having a mortgage is a positive "
            "financial stability signal."
        )

    # Loan intent
    if any('loan_intent' in f for f in top_for):
        decision_reasons.append(
            f"Borrowing for {intent_label} is associated with "
            f"slightly higher default rates."
        )
    elif any('loan_intent' in f for f in top_against):
        decision_reasons.append(
            f"Borrowing for {intent_label} is associated with "
            f"lower default rates."
        )

    # Income
    if any(f == 'person_income' for f in top_against):
        decision_reasons.append(
            f"Your annual income of ${person_inc:,.0f} helps "
            f"reduce default risk."
        )

    # Employment
    if any(f == 'person_emp_length' for f in top_against):
        decision_reasons.append(
            f"Your {emp_length:.0f} years of employment history "
            f"is a positive factor."
        )

    # Rate reasons
    if workflow == 'new_applicant' and predicted_rate:
        rate_reasons.append(
            f"Based on your profile, the estimated interest "
            f"rate is {round(predicted_rate, 2)}%."
        )
        if any('loan_grade' in f for f in top_high):
            rate_reasons.append(
                f"Your loan grade {grade_label} is the primary driver "
                f"of your interest rate — lower grades receive higher rates."
            )
        if any('loan_grade_x_loan_amnt' in f for f in top_high):
            rate_reasons.append(
                f"The combination of grade {loan_grade} and loan amount "
                f"${loan_amnt:,.0f} significantly raises your rate."
            )
        if any('debt_burden_score' in f for f in top_high):
            rate_reasons.append(
                "Your overall debt burden relative to income and "
                "loan grade is high, increasing your rate."
            )
        if any(f == 'person_income' for f in top_low):
            rate_reasons.append(
                f"Your income of ${person_inc:,.0f} slightly "
                f"lowers your estimated rate."
            )

    return {
        'decision_reasons': decision_reasons,
        'rate_reasons'    : rate_reasons
    }


# WORKFLOW 1 — NEW APPLICANT
def run_new_applicant(input_data: dict) -> dict:
    """
    New Applicant workflow.
    Steps: Regression → Classification → Clustering → SHAP → Plain Reasons
    Args:
        input_data: dict with 10 fields (no loan_int_rate)
    Returns:
        dict: full prediction result
    """
    cls_features = get_cls_features()
    reg_features = get_reg_features()

    # Step 1 — Regression
    reg_input = {**input_data, 'loan_status': 0}
    X_fe_reg  = _preprocess(reg_input, get_reg_pipeline(),
                             REG_PREPROCESSED_COLS, REG_FE_INPUT_COLS,
                             get_reg_fe_pipeline(), reg_features)
    predicted_rate = _predict_reg(X_fe_reg)

    # Step 2 — Classification
    cls_input = {**input_data,
                 'loan_int_rate': predicted_rate,
                 'loan_status'  : 0}
    X_fe_cls  = _preprocess(cls_input, get_cls_pipeline(),
                             CLS_PREPROCESSED_COLS, CLS_FE_INPUT_COLS,
                             get_cls_fe_pipeline(), cls_features)
    decision, prob = _predict_cls(X_fe_cls)

    # Step 3 — Clustering
    cluster_id, cluster_label, prob_high, prob_std = _predict_cluster(X_fe_cls)

    # Step 4 — SHAP
    reasons_for,     reasons_against = _get_cls_shap(X_fe_cls)
    reasons_high,    reasons_low     = _get_reg_shap(X_fe_reg)

    # Step 5 — Plain reasons
    plain_reasons = _generate_plain_reasons(
        decision       = decision,
        default_prob   = prob,
        predicted_rate = predicted_rate,
        shap_for       = reasons_for,
        shap_against   = reasons_against,
        shap_high      = reasons_high,
        shap_low       = reasons_low,
        original_input = input_data,
        workflow       = 'new_applicant'
    )

    return {
        'workflow'       : 'new_applicant',
        'prediction_data': {
            'classification': {
                'decision'               : decision,
                'default_probability'    : round(prob, 4),
                'threshold'              : BEST_THRESHOLD_CLS,
                'reasons_for_default'    : reasons_for,
                'reasons_against_default': reasons_against
            },
            'regression': {
                'predicted_interest_rate': predicted_rate,
                'reasons_high_rate'      : reasons_high,
                'reasons_low_rate'       : reasons_low
            },
            'clustering': {
                'cluster_id'   : cluster_id,
                'cluster_label': cluster_label,
                'probabilities': {
                    'High Value Borrower': prob_high,
                    'Standard Borrower'  : prob_std
                }
            },
            'plain_reasons': plain_reasons
        },
        'input_data': input_data
    }


# WORKFLOW 2 — EXISTING LOAN
def run_existing_loan(input_data: dict) -> dict:
    """
    Existing Loan workflow.
    Steps: Classification → Clustering → SHAP → Plain Reasons
    Regression is skipped — loan_int_rate already provided.
    Args:
        input_data: dict with 11 fields (with loan_int_rate)
    Returns:
        dict: full prediction result
    """
    cls_features = get_cls_features()

    # Step 1 — Classification
    cls_input = {**input_data, 'loan_status': 0}
    X_fe_cls  = _preprocess(cls_input, get_cls_pipeline(),
                             CLS_PREPROCESSED_COLS, CLS_FE_INPUT_COLS,
                             get_cls_fe_pipeline(), cls_features)
    decision, prob = _predict_cls(X_fe_cls)

    # Step 2 — Clustering
    cluster_id, cluster_label, prob_high, prob_std = _predict_cluster(X_fe_cls)

    # Step 3 — SHAP
    reasons_for, reasons_against = _get_cls_shap(X_fe_cls)

    # Step 4 — Plain reasons
    plain_reasons = _generate_plain_reasons(
        decision       = decision,
        default_prob   = prob,
        predicted_rate = None,
        shap_for       = reasons_for,
        shap_against   = reasons_against,
        shap_high      = [],
        shap_low       = [],
        original_input = input_data,
        workflow       = 'existing_loan'
    )

    return {
        'workflow'       : 'existing_loan',
        'prediction_data': {
            'classification': {
                'decision'               : decision,
                'default_probability'    : round(prob, 4),
                'threshold'              : BEST_THRESHOLD_CLS,
                'reasons_for_default'    : reasons_for,
                'reasons_against_default': reasons_against
            },
            'clustering': {
                'cluster_id'   : cluster_id,
                'cluster_label': cluster_label,
                'probabilities': {
                    'High Value Borrower': prob_high,
                    'Standard Borrower'  : prob_std
                }
            },
            'plain_reasons': plain_reasons
        },
        'input_data': input_data
    }