"""
Model evaluation utilities for credit card fraud detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute evaluation metrics for a single model.

    Returns:
        dict with keys: precision, recall, f1, roc_auc, confusion_matrix (list), classification_report (str)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
    }


def compare_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models on key metrics.

    Args:
        models: dict mapping model name -> fitted model

    Returns:
        DataFrame with columns: Model, Precision, Recall, F1, AUC
    """
    rows = []
    for name, model in models.items():
        metrics = compute_metrics(model, X_test, y_test)
        rows.append({
            'Model': name,
            'Precision': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'F1': round(metrics['f1'], 4),
            'AUC': round(metrics['roc_auc'], 4),
        })
    return pd.DataFrame(rows)


def plot_roc_curves(models: dict, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None) -> None:
    """
    Plot ROC curves for all models on a single figure.

    Args:
        models: dict mapping model name -> fitted model
        save_path: if provided, save figure to this path
    """
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Fraud Detection Models', fontsize=14)
    ax.legend(loc='lower right')
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_confusion_matrices(models: dict, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None) -> None:
    """
    Plot side-by-side confusion matrix heatmaps for all models.

    Args:
        models: dict mapping model name -> fitted model
        save_path: if provided, save figure to this path
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
        )
        ax.set_title(f'{name}', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)

    plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
