import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve


class MetricsVisualizer:
    """
    Generates plots from evaluation results stored in JSON format.

    The visualizer reads a JSON file containing individual run metrics
    and produces plots.
    """

    def __init__(self, output_dir: str = "./data/results/plots"):
        """
        Initializes the visualizer and creates the output directory.

        Args:
            output_dir (str): Path to the directory where plots will be saved.
                Defaults to "./data/results/plots". The directory is created
                if it does not exist.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        sns.set_theme(style="whitegrid")

    def plot_metric_distributions(self, json_filepath: str, prefix: str) -> None:
        """
        Creates boxplots and stripplots for all primary metrics across runs.

        The metrics included are: accuracy, precision, recall, f1_score, auc.
        The function reads the JSON file, reshapes the data, and saves a
        combined plot to the output directory.

        Args:
            json_filepath (str): Path to the JSON file containing evaluation results.
                The file must have an "individual_runs" key, and each run must
                contain the metrics listed above.
            prefix (str): Prefix used in the output filename and plot title.
        """
        with open(json_filepath) as f:
            data = json.load(f)

        records = []
        for run in data["individual_runs"]:
            for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
                records.append(
                    {
                        "Run": run["run_id"],
                        "Metric": metric.capitalize(),
                        "Score": run[metric],
                    }
                )

        df = pd.DataFrame(records)

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x="Metric",
            y="Score",
            hue="Metric",
            palette="Set2",
            legend=False,
        )
        sns.stripplot(data=df, x="Metric", y="Score", color=".25", size=6, jitter=True)

        plt.title(f"{prefix.capitalize()} Model Metrics Distribution across Seeds")
        plt.ylim(-0.05, 1.05)

        save_path = os.path.join(self.output_dir, f"{prefix}_distributions.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_comparative_roc(self, json_filepath: str, prefix: str) -> None:
        """
        Plots ROC curves for all runs on a single figure.

        The JSON file is expected to contain, for each run, the ground truth
        labels ("y_true") and predicted probabilities ("y_prob") in addition
        to the usual metrics.
        These fields are required to compute the ROC curve coordinates.

        Args:
            json_filepath (str): Path to the JSON file containing evaluation results.
                The file must have an "individual_runs" list, and each run
                dictionary must include "y_true", "y_prob", "run_id", and "auc".
            prefix (str): Prefix used in the output filename and plot title.
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
        """
        Generates a single ROC curve for a specific run and saves it.

        This method extracts the ground truth labels and predicted probabilities
        for the given run ID, computes its ROC curve, and creates a standalone
        figure.
        The run must exist in the JSON file and contain "y_true" and "y_prob" fields.

        Args:
            json_filepath (str): Path to the JSON file containing evaluation results.
            run_id (str): Unique identifier of the run to plot.
            prefix (str): Prefix used in the output filename and plot title.

        Raises:
            ValueError: If the provided run_id is not found in the JSON data.
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
        """
        Generates and saves a heatmap of the confusion matrix for a specific run.

        The confusion matrix is extracted from the JSON data for the given run_id
        and plotted as a seaborn heatmap. The plot uses the predefined class labels
        "Non-Demented" and "Demented".

        Args:
            json_filepath (str): Path to the JSON file containing evaluation results.
                The file must have an "individual_runs" list, and the specified run
                must contain a "confusion_matrix" key with a 2x2 integer matrix.
            run_id (str): Unique identifier of the run whose confusion matrix will
                be plotted.
            prefix (str): Prefix used in the output filename and plot title.

        Raises:
            ValueError: If the provided run_id is not found in the JSON data.
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
        """
        Retrieves the data dictionary for a specific run from the JSON structure.

        This helper method searches the "individual_runs" list for an entry with
        the matching run_id.

        Args:
            data (dict): The full JSON data as a dictionary, expected to contain
                an "individual_runs" key with a list of run dictionaries.
            run_id (str): The unique identifier of the run to retrieve.

        Returns:
            dict: The run dictionary corresponding to the given run_id.

        Raises:
            ValueError: If no run with the specified run_id is found.
        """
        run_data = next(
            (run for run in data.get("individual_runs", []) if run["run_id"] == run_id),
            None,
        )
        if not run_data:
            raise ValueError(f"Run ID '{run_id}' not found in the provided JSON.")
        return run_data
