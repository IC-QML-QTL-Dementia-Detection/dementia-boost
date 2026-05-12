from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """DTO for a single model's evaluation metrics."""

    run_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    confusion_matrix: list[list[int]]


@dataclass
class AggregateMetrics:
    """DTO for statistical aggregations across multiple runs."""

    metric_name: str
    mean: float
    std: float
    min_val: float
    max_val: float


class MetricsAnalyzer:
    """
    A utility class that handles standard classification metrics
    calculations from raw probabilities and serialization.
    """

    @staticmethod
    def calculate_metrics(
        run_id: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """
        Calculates Accuracy, Precision, Recall, F1, AUC, and the Confusion Matrix.

        Args:
            run_id (str): A unique identifier for this run (e.g., "seed_42").
            y_true (np.ndarray): The ground-truth binary labels.
            y_prob (np.ndarray): The raw probabilities from the model.
            threshold (float): The cutoff point to convert probabilities
                to binary predictions.

        Returns:
            dict[str, Any]: A dictionary containing the calculated metrics.
        """
        y_pred = (y_prob >= threshold).astype(int)

        return EvaluationResult(
            run_id=run_id,
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division="warn")),
            recall=float(recall_score(y_true, y_pred, zero_division="warn")),
            f1_score=float(f1_score(y_true, y_pred, zero_division="warn")),
            auc=float(roc_auc_score(y_true, y_prob)),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        )
