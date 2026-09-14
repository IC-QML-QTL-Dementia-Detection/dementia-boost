"""Publication-ready visualization generators for evaluation metrics and ROC curves.

This module provides `MetricsVisualizer` to generate Seaborn boxplots/stripplots,
multi-run comparative and isolated ROC curves, and annotated confusion matrix
heatmaps from serialized JSON evaluation payloads.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve


class MetricsVisualizer:
    """Generates publication-quality charts and plots from JSON evaluation metrics.

    Reads telemetry JSON files containing individual and aggregate experiment
    runs and renders metric distributions, ROC curves, and confusion matrices.

    Attributes:
        output_dir: Destination directory path where plots will be saved.
    """

    def __init__(self, output_dir: str = "./data/results/plots") -> None:
        """Initializes the visualizer and ensures output directory exists.

        Args:
            output_dir: Path to directory where generated plots will be saved.
                Defaults to "./data/results/plots".
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        sns.set_theme(style="whitegrid")

    def plot_metric_distributions(self, json_filepath: str, prefix: str) -> None:
        """Generates boxplots with jittered Seaborn stripplots across all runs.

        Visualizes distributions for accuracy, precision, recall, f1_score, and
        auc across multi-seed runs, saving the resulting figure as a PNG.

        Args:
            json_filepath: Path to the JSON file containing evaluation results.
            prefix: Prefix string used in the output filename and plot title.
        """
        with open(json_filepath) as f:
            data = json.load(f)

        metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc"]
        metric_labels = [key.capitalize() for key in metric_keys]

        records = []
        for run in data["individual_runs"]:
            for key, label in zip(metric_keys, metric_labels, strict=True):
                records.append(
                    {
                        "Run": run["run_id"],
                        "Metric": label,
                        "Score": run[key],
                    }
                )

        df = pd.DataFrame(records)

        fig, ax = plt.subplots(figsize=(10, 6))

        box_data = [
            df.loc[df["Metric"] == label, "Score"].to_numpy() for label in metric_labels
        ]
        box_positions = range(len(metric_labels))
        box_result = ax.boxplot(
            box_data,
            positions=box_positions,
            tick_labels=metric_labels,
            orientation="vertical",
            patch_artist=True,
        )
        palette = sns.color_palette("Set2", n_colors=len(metric_labels))
        for patch, color in zip(box_result["boxes"], palette, strict=True):
            patch.set_facecolor(color)

        sns.stripplot(
            data=df,
            x="Metric",
            y="Score",
            order=metric_labels,
            color=".25",
            size=6,
            jitter=True,
            ax=ax,
        )

        ax.set_title(f"{prefix.capitalize()} Model Metrics Distribution across Seeds")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")

        save_path = os.path.join(self.output_dir, f"{prefix}_distributions.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_comparative_roc(self, json_filepath: str, prefix: str) -> None:
        """Plots comparative ROC curves for all runs on a single figure.

        Extracts ground truth (`y_true`) and predicted probabilities (`y_prob`)
        for each run, computes the ROC coordinates, and plots overlay curves
        along with the random baseline diagonal.

        Args:
            json_filepath: Path to the JSON file containing evaluation results.
            prefix: Prefix string used in the output filename and plot title.
        """
        with open(json_filepath) as f:
            data = json.load(f)

        plt.figure(figsize=(8, 8))

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

        for run in data["individual_runs"]:
            fpr, tpr, _ = roc_curve(run["y_true"], run["y_prob"])
            auc_score = run["auc"]
            plt.plot(
                fpr,
                tpr,
                alpha=0.5,
                label=f"{run['run_id']} (AUC = {auc_score:.2f})",
            )

        plt.title(f"{prefix.capitalize()} ROC Curves Across Seeds")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize="small",
            borderaxespad=0,
        )

        save_path = os.path.join(self.output_dir, f"{prefix}_roc_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_isolated_roc(self, json_filepath: str, run_id: str, prefix: str) -> None:
        """Generates a standalone ROC curve for a specific model run.

        Args:
            json_filepath: Path to the JSON file containing evaluation results.
            run_id: Unique identifier for the specific run to plot.
            prefix: Prefix string used in the output filename and plot title.

        Raises:
            ValueError: If `run_id` is not found in the JSON file.
        """
        with open(json_filepath) as f:
            data = json.load(f)

        run_data = self._get_run_data(data, run_id)

        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

        fpr, tpr, _ = roc_curve(run_data["y_true"], run_data["y_prob"])
        auc_score = run_data["auc"]
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            linewidth=2,
            label=f"{run_id} (AUC={auc_score:.4f})",
        )

        plt.title(f"{prefix.capitalize()} ROC Curve: {run_id}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        save_path = os.path.join(self.output_dir, f"{prefix}_isolated_roc_{run_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(
        self,
        json_filepath: str,
        run_id: str,
        prefix: str,
    ) -> None:
        """Renders an annotated heatmap of the confusion matrix for a run.

        Args:
            json_filepath: Path to the JSON file containing evaluation results.
            run_id: Unique identifier for the specific run to plot.
            prefix: Prefix string used in the output filename and plot title.

        Raises:
            ValueError: If `run_id` is not found in the JSON file.
        """
        with open(json_filepath) as f:
            data = json.load(f)

        run_data = self._get_run_data(data, run_id)
        cm = np.array(run_data["confusion_matrix"])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non-Demented", "Demented"],
            yticklabels=["Non-Demented", "Demented"],
        )

        plt.title(f"{prefix.capitalize()} Confusion Matrix: {run_id}")
        plt.ylabel("Actual Diagnosis")
        plt.xlabel("Predicted Diagnosis")

        save_path = os.path.join(self.output_dir, f"{prefix}_cm_{run_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _get_run_data(self, data: dict, run_id: str) -> dict:
        """Extracts the evaluation record matching the specified run ID.

        Args:
            data: Parsed dictionary from the evaluation JSON file.
            run_id: Unique identifier for the target run.

        Returns:
            Dictionary containing metrics and predictions for `run_id`.

        Raises:
            ValueError: If no entry matching `run_id` exists in `individual_runs`.
        """
        run_data = next(
            (run for run in data.get("individual_runs", []) if run["run_id"] == run_id),
            None,
        )
        if not run_data:
            raise ValueError(f"Run ID '{run_id}' not found in the provided JSON.")
        return run_data
