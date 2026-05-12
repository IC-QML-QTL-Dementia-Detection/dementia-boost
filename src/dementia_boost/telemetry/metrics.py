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
    """
    DTO for a single model's evaluation metrics.

    Attributes:
        run_id (str): Unique identifier for the experiment run.
        accuracy (float): Accuracy score (TP + TN) / total.
        precision (float): Precision score TP / (TP + FP).
        recall (float): Recall score TP / (TP + FN).
        f1_score (float): Harmonic mean of precision and recall.
        auc (float): Area under the ROC curve.
        confusion_matrix (list[list[int]]): 2x2 confusion matrix as nested list.
    """

    run_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    confusion_matrix: list[list[int]]


@dataclass
class AggregateMetrics:
    """
    DTO for statistical aggregations across multiple runs.

    Attributes:
        metric_name (str): Name of the metric being aggregated.
        mean (float): Mean value of the metric across all runs.
        std (float): Standard deviation of the metric across runs.
        min_val (float): Minimum value of the metric across runs.
        max_val (float): Maximum value of the metric across runs.
    """

    metric_name: str
    mean: float
    std: float
    min_val: float
    max_val: float


class MetricsAnalyzer:
    """
    Utility class for classification metrics calculation and aggregation.

    Provides static methods to compute standard binary classification metrics
    from probabilities, aggregate results over multiple runs, and save the
    results to a JSON file.
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
            EvaluationResult: An object containing the calculated metrics.
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

    @staticmethod
    def aggregate_results(
        results: list[EvaluationResult],
    ) -> dict[str, AggregateMetrics]:
        """
        Calculates mean, std, min, and max for all primary metrics across multiple runs.

        Args:
            results (list[EvaluationResult]): A list of objects containing the
                calculated metrics.

        Returns:
            dict[str, AggregateMetrics]: A dictionary mapping each metric name
                to its aggregated statistics (mean, std, min, max).
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
        """
        Serializes all results to a JSON file for future plotting/analysis.

        The JSON structure contains two top-level keys:
        - "aggregated_statistics": a dictionary mapping metric names to
          the aggregated statistics (mean, std, min, max).
        - "individual_runs": a list of dictionaries, each representing one
          EvaluationResult object.

        Args:
            individual_results (list[EvaluationResult]): List of per‑run results.
            aggregated (dict[str, AggregateMetrics]): Aggregated statistics as produced
                by the `aggregate_results` method.
            filepath (str): Destination file path.
        """
        payload = {
            "aggregated_statistics": {k: asdict(v) for k, v in aggregated.items()},
            "individual_runs": [asdict(res) for res in individual_results],
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=4)
