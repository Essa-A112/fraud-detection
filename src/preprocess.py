"""
Preprocessing pipeline for credit card fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data and drop rows with NaN values."""
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def split_features_target(df: pd.DataFrame, target_col: str = 'Class') -> tuple:
    """Split DataFrame into features X and target y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Stratified train/test split to preserve class ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple:
    """Apply SMOTE oversampling to balance the training set."""
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Fit StandardScaler on training data and transform both train and test sets.

    Returns:
        scaled_train (np.ndarray), scaled_test (np.ndarray), scaler (StandardScaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def full_pipeline(path: str) -> dict:
    """
    Run the complete preprocessing pipeline.

    Steps:
        1. Load data and drop NaN
        2. Split features and target
        3. Stratified train/test split
        4. Apply SMOTE on training set only
        5. Scale features (fit on train, transform both)

    Returns:
        dict with keys: X_train_res, X_test_scaled, y_train_res, y_test, scaler
    """
    df = load_data(path)
    X, y = split_features_target(df, target_col='Class')
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_res, X_test)
    return {
        'X_train_res': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_res': y_train_res,
        'y_test': y_test,
        'scaler': scaler,
    }
