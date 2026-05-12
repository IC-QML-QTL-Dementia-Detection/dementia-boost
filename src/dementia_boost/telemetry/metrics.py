from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsAnalyzer:
    """
    A stateless utility class that calculates standard classification metrics
    from raw probabilities and ground-truth labels.
    """

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Calculates Accuracy, Precision, Recall, F1, AUC, and the Confusion Matrix.

        Args:
            y_true (np.ndarray): The ground-truth binary labels.
            y_prob (np.ndarray): The raw probabilities from the model.
            threshold (float): The cutoff point to convert probabilities
                to binary predictions.

        Returns:
            dict[str, Any]: A dictionary containing the calculated metrics.
        """
        y_pred = (y_prob >= threshold).astype(int)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division="warn")),
            "recall": float(recall_score(y_true, y_pred, zero_division="warn")),
            "f1_score": float(f1_score(y_true, y_pred, zero_division="warn")),
            "auc": float(roc_auc_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
