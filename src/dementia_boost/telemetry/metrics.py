"""Classification metric computation, multi-run aggregation, and JSON serialization.

This module provides data transfer objects (`EvaluationResult`, `AggregateMetrics`)
and the `MetricsAnalyzer` class to calculate standard binary classification
metrics (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix) and serialize
summaries to disk.
"""

import json
from dataclasses import asdict, dataclass

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
    """Data transfer object storing a single model run's evaluation metrics.

    Attributes:
        run_id: Unique identifier for the experiment run.
        accuracy: Accuracy score (TP + TN) / total.
        precision: Precision score TP / (TP + FP).
        recall: Recall score TP / (TP + FN).
        f1_score: Harmonic mean of precision and recall.
        auc: Area under the Receiver Operating Characteristic (ROC) curve.
        confusion_matrix: 2x2 confusion matrix as a nested integer list.
        y_true: Ground-truth binary classification labels.
        y_prob: Predicted model probability values.
    """

    run_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    confusion_matrix: list[list[int]]
    y_true: list[int]
    y_prob: list[float]


@dataclass
class AggregateMetrics:
    """Data transfer object storing statistical aggregations across multiple runs.

    Attributes:
        metric_name: Name of the classification metric being summarized.
        mean: Sample mean across all evaluation runs.
        std: Sample standard deviation across all evaluation runs.
        min_val: Minimum metric score observed across runs.
        max_val: Maximum metric score observed across runs.
    """

    metric_name: str
    mean: float
    std: float
    min_val: float
    max_val: float


class MetricsAnalyzer:
    """Utility engine for computing and aggregating classification metrics.

    Provides stateless static methods to calculate standard binary metrics from
    model probability predictions, compute cross-run statistics (mean, std, min,
    max), and serialize evaluation payloads to JSON files.
    """

    @staticmethod
    def calculate_metrics(
        run_id: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """Calculates Accuracy, Precision, Recall, F1, AUC, and Confusion Matrix.

        Converts raw probabilities to discrete binary labels using `threshold`
        and evaluates classification scores via scikit-learn.

        Args:
            run_id: Unique identifier for this evaluation run.
            y_true: 1D array of ground-truth binary labels.
            y_prob: 1D array of raw prediction probabilities.
            threshold: Probability decision boundary for positive class
                assignment. Defaults to 0.5.

        Returns:
            An EvaluationResult instance populated with all computed metrics.
        """
        y_pred = (y_prob >= threshold).astype(int)

        return EvaluationResult(
            run_id=run_id,
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),  # type: ignore
            recall=float(recall_score(y_true, y_pred, zero_division=0)),  # type: ignore
            f1_score=float(f1_score(y_true, y_pred, zero_division=0)),  # type: ignore
            auc=float(roc_auc_score(y_true, y_prob)),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
            y_true=y_true.astype(int).tolist(),
            y_prob=y_prob.astype(float).tolist(),
        )

    @staticmethod
    def aggregate_results(
        results: list[EvaluationResult],
    ) -> dict[str, AggregateMetrics]:
        """Calculates summary statistics across multiple independent runs.

        Computes mean, standard deviation, minimum, and maximum across all
        primary metrics (accuracy, precision, recall, f1_score, auc).

        Args:
            results: List of EvaluationResult instances to aggregate.

        Returns:
            A dictionary mapping each metric name to its AggregateMetrics
            summary object. Returns an empty dict if `results` is empty.
        """
        if not results:
            return {}

        metrics_keys = ["accuracy", "precision", "recall", "f1_score", "auc"]
        aggregated = {}

        for key in metrics_keys:
            values = [getattr(res, key) for res in results]
            aggregated[key] = AggregateMetrics(
                metric_name=key,
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                min_val=float(np.min(values)),
                max_val=float(np.max(values)),
            )

        return aggregated

    @staticmethod
    def save_to_json(
        individual_results: list[EvaluationResult],
        aggregated: dict[str, AggregateMetrics],
        filepath: str,
    ) -> None:
        """Serializes evaluation metrics and aggregate statistics to a JSON file.

        The exported JSON structure contains:
        - `aggregated_statistics`: mapping metric names to summary statistics.
        - `individual_runs`: list of per-run evaluation metric dictionaries.

        Args:
            individual_results: List of per-run EvaluationResult objects.
            aggregated: Dictionary mapping metric names to AggregateMetrics objects.
            filepath: Destination file path on disk.
        """
        payload = {
            "aggregated_statistics": {k: asdict(v) for k, v in aggregated.items()},
            "individual_runs": [asdict(res) for res in individual_results],
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=4)
