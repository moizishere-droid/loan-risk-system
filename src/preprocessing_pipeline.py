import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# CUSTOM TRANSFORMERS

# This is the function for my Cross col logic fixer
class CrossColumnFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): # train dataset only
        return self
    def transform(self, X): # test and val and new dataset
        X = X.copy()
        X['person_emp_length'] = X.apply(
            lambda row: min(row['person_emp_length'], row['person_age'] - 18)
            if pd.notna(row['person_emp_length']) else row['person_emp_length'], axis=1)
        X['cb_person_cred_hist_length'] = X.apply(
            lambda row: min(row['cb_person_cred_hist_length'], row['person_age'] - 18)
            if pd.notna(row['cb_person_cred_hist_length']) else row['cb_person_cred_hist_length'], axis=1)
        X['loan_percent_income'] = X['loan_amnt'] / (X['person_income'] + 1)
        return X

# This is the function for Capping the outlier
class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bounds = {}
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = {'lower': Q1 - 1.5 * IQR, 'upper': Q3 + 1.5 * IQR}
        return self
    def transform(self, X):
        X = X.copy()
        for col, bound in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=bound['lower'], upper=bound['upper'])
        return X

# This is the function for my binnary mapping (Y/N -> 1/0)
class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, column, mapping):
        self.column = column
        self.mapping = mapping
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].map(self.mapping)
        return X

# This is the function for Drop highly correlataed features
class CorrelatedFeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.cols_to_drop:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)
        return X

# This is the function for optimizing the memory usage by dtype
class DTypeOptimizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=['float64']).columns:
            X[col] = X[col].astype('float32')
        for col in X.select_dtypes(include=['int64']).columns:
            X[col] = X[col].astype('int32')
        return X


# COLUMN TRANSFORMER
def build_column_transformer_cls():
    numerical_cols = ['person_age', 'person_income', 'person_emp_length',
                      'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    ordinal_cols = ['loan_grade']
    ohe_cols = ['person_home_ownership', 'loan_intent']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E', 'F', 'G']]))
    ])
    # OTHER excluded from categories - handles rare category
    ohe_pipeline = Pipeline([
        ('ohe', OneHotEncoder(
            categories=[
                ['MORTGAGE', 'OWN', 'RENT'],
                ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT',
                 'MEDICAL', 'PERSONAL', 'VENTURE']
            ],
            sparse_output=False,
            handle_unknown='ignore'
        ))
    ])
    return ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
        ('ohe', ohe_pipeline, ohe_cols)
    ], remainder='passthrough')


def build_column_transformer_reg():
    numerical_cols = ['person_age', 'person_income', 'person_emp_length',
                      'loan_amnt', 'loan_percent_income']
    ordinal_cols = ['loan_grade']
    ohe_cols = ['person_home_ownership', 'loan_intent']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E', 'F', 'G']]))
    ])
    ohe_pipeline = Pipeline([
        ('ohe', OneHotEncoder(
            categories=[
                ['MORTGAGE', 'OWN', 'RENT'],
                ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT',
                 'MEDICAL', 'PERSONAL', 'VENTURE']
            ],
            sparse_output=False,
            handle_unknown='ignore'
        ))
    ])
    return ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
        ('ohe', ohe_pipeline, ohe_cols)
    ], remainder='passthrough')


# FULL PIPELINES

def build_classification_pipeline():
    return Pipeline([
        ('cross_col_fix', CrossColumnFixer()),
        ('iqr_capper', IQRCapper()),
        ('corr_dropper', CorrelatedFeatureDropper(cols_to_drop=['cb_person_cred_hist_length'])),
        ('binary_mapper', BinaryMapper(column='cb_person_default_on_file', mapping={'Y': 1, 'N': 0})),
        ('col_transformer', build_column_transformer_cls()),
    ])


def build_regression_pipeline():
    return Pipeline([
        ('cross_col_fix', CrossColumnFixer()),
        ('iqr_capper', IQRCapper()),
        ('corr_dropper', CorrelatedFeatureDropper(cols_to_drop=['cb_person_cred_hist_length'])),
        ('binary_mapper', BinaryMapper(column='cb_person_default_on_file', mapping={'Y': 1, 'N': 0})),
        ('col_transformer', build_column_transformer_reg()),
    ])


# USAGE
if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    config_file = BASE_DIR / "config" / "config.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
    splits_path = os.path.join(root_dir, config['paths']['splits_data'])

    # Load raw splits
    X_train_cls = pd.read_csv(os.path.join(splits_path, 'X_train_cls.csv'))
    X_val_cls   = pd.read_csv(os.path.join(splits_path, 'X_val_cls.csv'))
    X_test_cls  = pd.read_csv(os.path.join(splits_path, 'X_test_cls.csv'))
    X_train_reg = pd.read_csv(os.path.join(splits_path, 'X_train_reg.csv'))
    X_val_reg   = pd.read_csv(os.path.join(splits_path, 'X_val_reg.csv'))
    X_test_reg  = pd.read_csv(os.path.join(splits_path, 'X_test_reg.csv'))

    # Build and fit pipelines
    cls_pipeline = build_classification_pipeline()
    reg_pipeline = build_regression_pipeline()

    X_train_cls_processed = cls_pipeline.fit_transform(X_train_cls)
    X_train_reg_processed = reg_pipeline.fit_transform(X_train_reg)

    X_val_cls_processed   = cls_pipeline.transform(X_val_cls)
    X_test_cls_processed  = cls_pipeline.transform(X_test_cls)
    X_val_reg_processed   = reg_pipeline.transform(X_val_reg)
    X_test_reg_processed  = reg_pipeline.transform(X_test_reg)

    # Save pipelines
    models_path = BASE_DIR / "models"
    joblib.dump(cls_pipeline, models_path / "cls_pipeline.joblib")
    joblib.dump(reg_pipeline, models_path / "reg_pipeline.joblib")

    print("Classification pipeline saved")
    print("Regression pipeline saved")
    print(f"Classification train shape: {X_train_cls_processed.shape}")
    print(f"Regression train shape: {X_train_reg_processed.shape}")
