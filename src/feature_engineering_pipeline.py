import pandas as pd
import numpy as np
import joblib
import os
import yaml
import json
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# CUSTOM TRANSFORMERS

class RatioFeatureCreator(BaseEstimator, TransformerMixin):
    """Create ratio features from existing numerical features"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ratio 1: income_to_loan_ratio
        X['income_to_loan_ratio'] = X['person_income'] / (X['loan_amnt'] + 1)

        # Ratio 2: loan_to_age_ratio
        X['loan_to_age_ratio'] = X['loan_amnt'] / (X['person_age'] + 1)

        # Ratio 3: income_per_emp_year
        X['income_per_emp_year'] = X['person_income'] / (X['person_emp_length'] + 1)

        return X


class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """Create interaction features — different for cls and reg"""

    def __init__(self, task_type='cls'):
        self.task_type = task_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Interaction 2: loan_grade × loan_percent_income
        X['loan_grade_x_loan_percent_income'] = X['loan_grade'] * X['loan_percent_income']

        # Interaction 3: person_age × person_emp_length
        X['person_age_x_person_emp_length'] = X['person_age'] * X['person_emp_length']

        if self.task_type == 'cls':
            # Classification only — loan_int_rate is a feature
            X['loan_grade_x_loan_int_rate'] = X['loan_grade'] * X['loan_int_rate']
        else:
            # Regression only — loan_int_rate is target, use loan_amnt instead
            X['loan_grade_x_loan_amnt'] = X['loan_grade'] * X['loan_amnt']

        return X


class BinningFeatureCreator(BaseEstimator, TransformerMixin):
    """Create binning features using scaled value ranges"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Bins based on scaled ranges (-inf to -0.5 = low, -0.5 to 0.5 = mid, 0.5 to inf = high)
        X['age_group'] = pd.cut(
            X['person_age'],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=[0, 1, 2], right=False
        ).astype(float)

        X['income_group'] = pd.cut(
            X['person_income'],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=[0, 1, 2], right=False
        ).astype(float)

        X['loan_size_group'] = pd.cut(
            X['loan_amnt'],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=[0, 1, 2], right=False
        ).astype(float)

        return X


class DomainFeatureCreator(BaseEstimator, TransformerMixin):
    """Create domain knowledge features for loan risk"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Use loan_grade directly (already encoded as 0-6)
        loan_grade_num = X['loan_grade']

        # Feature 1: High risk borrower flag
        X['is_high_risk'] = np.where(
            (loan_grade_num >= 4) & (X['loan_percent_income'] > 0.3),
            1, 0)

        # Feature 2: Experienced borrower flag
        X['is_experienced_borrower'] = np.where(
            (X['person_emp_length'] > 5) & (X['person_age'] > 25),
            1, 0)

        # Feature 3: Debt burden score
        X['debt_burden_score'] = (
            X['loan_amnt'] * loan_grade_num) / (X['person_income'] + 1)

        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drop specified columns"""

    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols)


class NewFeatureScaler(BaseEstimator, TransformerMixin):
    """Scale new numerical features created during FE"""

    def __init__(self, cols_to_scale):
        self.cols_to_scale = cols_to_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        cols = [c for c in self.cols_to_scale if c in X.columns]
        self.scaler.fit(X[cols])
        self.fitted_cols = cols
        return self

    def transform(self, X):
        X = X.copy()
        X[self.fitted_cols] = self.scaler.transform(X[self.fitted_cols])
        return X


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Drop low variance features"""

    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.vt = VarianceThreshold(threshold=threshold)

    def fit(self, X, y=None):
        self.vt.fit(X)
        self.selected_cols = X.columns[self.vt.get_support()].tolist()
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.vt.transform(X),
            columns=self.selected_cols,
            index=X.index)


class FinalFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep only final selected features"""

    def __init__(self, final_features):
        self.final_features = final_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = [c for c in self.final_features if c in X.columns]
        return X[cols]


# NEW NUMERICAL FEATURES — for scaling

NEW_NUMERICAL_CLS = [
    'income_to_loan_ratio',
    'loan_to_age_ratio',
    'income_per_emp_year',
    'loan_grade_x_loan_int_rate',
    'loan_grade_x_loan_percent_income',
    'person_age_x_person_emp_length',
    'debt_burden_score'
]

NEW_NUMERICAL_REG = [
    'income_to_loan_ratio',
    'loan_to_age_ratio',
    'income_per_emp_year',
    'loan_grade_x_loan_amnt',
    'loan_grade_x_loan_percent_income',
    'person_age_x_person_emp_length',
    'debt_burden_score'
]

# Final features after all selection
FINAL_FEATURES_CLS = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_default_on_file',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
    'loan_intent_PERSONAL', 'loan_intent_VENTURE',
    'income_to_loan_ratio', 'loan_to_age_ratio', 'income_per_emp_year',
    'loan_grade_x_loan_int_rate', 'loan_grade_x_loan_percent_income',
    'person_age_x_person_emp_length', 'debt_burden_score'
]

FINAL_FEATURES_REG = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_grade', 'loan_amnt', 'loan_percent_income',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'income_to_loan_ratio', 'loan_to_age_ratio', 'income_per_emp_year',
    'loan_grade_x_loan_amnt', 'loan_grade_x_loan_percent_income',
    'person_age_x_person_emp_length', 'debt_burden_score'
]


# BUILD PIPELINES

def build_cls_fe_pipeline():
    """Complete Feature Engineering pipeline for classification"""
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        # Step 1: Create ratio features
        ('ratio_features', RatioFeatureCreator()),

        # Step 2: Create interaction features (cls version)
        ('interaction_features', InteractionFeatureCreator(task_type='cls')),

        # Step 3: Create binning features
        ('binning_features', BinningFeatureCreator()),

        # Step 4: Create domain knowledge features
        ('domain_features', DomainFeatureCreator()),

        # Step 5: Scale new numerical features
        ('new_feature_scaler', NewFeatureScaler(cols_to_scale=NEW_NUMERICAL_CLS)),

        # Step 6: Drop low variance features
        ('variance_filter', VarianceFilter(threshold=0.01)),

        # Step 7: Keep only final selected features
        ('final_selector', FinalFeatureSelector(final_features=FINAL_FEATURES_CLS)),
    ])

    return pipeline


def build_reg_fe_pipeline():
    """Complete Feature Engineering pipeline for regression"""
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        # Step 1: Create ratio features
        ('ratio_features', RatioFeatureCreator()),

        # Step 2: Create interaction features (reg version)
        ('interaction_features', InteractionFeatureCreator(task_type='reg')),

        # Step 3: Create binning features
        ('binning_features', BinningFeatureCreator()),

        # Step 4: Create domain knowledge features
        ('domain_features', DomainFeatureCreator()),

        # Step 5: Scale new numerical features
        ('new_feature_scaler', NewFeatureScaler(cols_to_scale=NEW_NUMERICAL_REG)),

        # Step 6: Drop low variance features
        ('variance_filter', VarianceFilter(threshold=0.01)),

        # Step 7: Keep only final selected features
        ('final_selector', FinalFeatureSelector(final_features=FINAL_FEATURES_REG)),
    ])

    return pipeline


# USAGE

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    config_file = BASE_DIR / "config" / "config.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
    processed_path = os.path.join(root_dir, config['paths']['processed_data'])

    # Load preprocessed splits
    X_train_cls = pd.read_csv(os.path.join(processed_path, 'X_train_cls_processed.csv'))
    X_val_cls   = pd.read_csv(os.path.join(processed_path, 'X_val_cls_processed.csv'))
    X_test_cls  = pd.read_csv(os.path.join(processed_path, 'X_test_cls_processed.csv'))

    X_train_reg = pd.read_csv(os.path.join(processed_path, 'X_train_reg_processed.csv'))
    X_val_reg   = pd.read_csv(os.path.join(processed_path, 'X_val_reg_processed.csv'))
    X_test_reg  = pd.read_csv(os.path.join(processed_path, 'X_test_reg_processed.csv'))

    # Build pipelines
    cls_fe_pipeline = build_cls_fe_pipeline()
    reg_fe_pipeline = build_reg_fe_pipeline()

    # Fit on train only
    X_train_cls_fe = cls_fe_pipeline.fit_transform(X_train_cls)
    X_train_reg_fe = reg_fe_pipeline.fit_transform(X_train_reg)

    # Transform val and test
    X_val_cls_fe   = cls_fe_pipeline.transform(X_val_cls)
    X_test_cls_fe  = cls_fe_pipeline.transform(X_test_cls)
    X_val_reg_fe   = reg_fe_pipeline.transform(X_val_reg)
    X_test_reg_fe  = reg_fe_pipeline.transform(X_test_reg)

    # Save pipelines separately
    models_path = BASE_DIR / "models"
    joblib.dump(cls_fe_pipeline, models_path / "cls_fe_pipeline.joblib")
    joblib.dump(reg_fe_pipeline, models_path / "reg_fe_pipeline.joblib")

    print("Classification FE pipeline saved")
    print("Regression FE pipeline saved")
    print(f"Classification train shape: {X_train_cls_fe.shape}")
    print(f"Regression train shape: {X_train_reg_fe.shape}")