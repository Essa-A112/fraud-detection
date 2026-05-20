"""
Model training utilities for credit card fraud detection.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train all three models: Logistic Regression, Random Forest, XGBoost.

    Returns:
        dict with keys 'logistic_regression', 'random_forest', 'xgboost'
    """
    models = {
        'logistic_regression': train_logistic_regression(X_train, y_train),
        'random_forest': train_random_forest(X_train, y_train),
        'xgboost': train_xgboost(X_train, y_train),
    }
    return models


def save_model(model, path: str) -> None:
    """Persist a trained model to disk using joblib."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load a persisted model from disk using joblib."""
    return joblib.load(path)
