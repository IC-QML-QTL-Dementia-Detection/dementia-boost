import json
import os

import matplotlib.pyplot as plt
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
        sns.boxplot(data=df, x="Metric", y="Score", palette="Set2")
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
        plt.legend(loc="lower right")

        save_path = os.path.join(self.output_dir, f"{prefix}_roc_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
