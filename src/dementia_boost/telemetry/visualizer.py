"""Publication-ready visualization generators for evaluation metrics and ROC curves.

This module provides `MetricsVisualizer` to generate Seaborn boxplots/stripplots,
multi-run comparative and isolated ROC curves, and annotated confusion matrix
heatmaps from serialized JSON evaluation payloads, plus single-run, multi-seed,
and cross-paradigm loss curves from serialized `TrainingHistory` files.
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve

from dementia_boost.telemetry.metrics import MetricsAnalyzer, TrainingHistory


class MetricsVisualizer:
    """Generates publication-quality charts and plots from JSON evaluation metrics.

    Reads telemetry JSON files containing individual and aggregate experiment
    runs and renders metric distributions, ROC curves, confusion matrices, and
    training loss curves.

    Attributes:
        CHANCE_LEVEL_LOSS: Binary cross-entropy of a chance-level predictor on
            balanced classes, ln 2.
        output_dir: Destination directory path where plots will be saved.
    """

    CHANCE_LEVEL_LOSS: float = math.log(2)

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

    def plot_loss_curve(self, history_path: str, prefix: str) -> None:
        """Plots loss and accuracy against epoch for a single training run.

        Draws train and validation series on two panels. Validation points
        appear only for evaluated epochs. Dashed vertical lines mark learning
        rate decay boundaries read from the history config, and the loss panel
        carries a dotted reference at the chance-level loss.

        Args:
            history_path: Path to a JSON file written by
                `MetricsAnalyzer.save_history`.
            prefix: Prefix string used in the output filename and plot title.
        """
        history = MetricsAnalyzer.load_history(history_path)
        records = history.epochs
        evaluated = [r for r in records if r.val_loss is not None]

        panels = (
            (
                "Loss",
                [r.train_loss for r in records],
                [r.val_loss for r in evaluated],
            ),
            (
                "Accuracy",
                [r.train_acc for r in records],
                [r.val_acc for r in evaluated],
            ),
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (label, train_values, val_values) in zip(axes, panels, strict=True):
            ax.plot([r.epoch for r in records], train_values, label="Train")
            if evaluated:
                ax.plot(
                    [r.epoch for r in evaluated],
                    val_values,
                    marker="o",
                    label="Validation",
                )
            self._mark_lr_decays(ax, history)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)

        axes[0].axhline(
            self.CHANCE_LEVEL_LOSS,
            color="gray",
            linestyle=":",
            label="Chance level (ln 2)",
        )
        for ax in axes:
            ax.legend()

        fig.suptitle(f"{prefix.capitalize()} Training Curves: {history.run_id}")

        save_path = os.path.join(self.output_dir, f"{prefix}_{history.run_id}_loss.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_loss_distribution(self, history_dir: str, prefix: str) -> None:
        """Plots the loss across every seed of one sweep.

        Aggregates all `*.json` histories in `history_dir`. For train and
        validation loss it draws the per-epoch median with an interquartile
        band over faint per-seed lines. The median is used so a single outlier
        seed does not distort the summary.

        Args:
            history_dir: Directory holding one history JSON file per seed.
            prefix: Prefix string used in the output filename and plot title.

        Raises:
            ValueError: If `history_dir` contains no history files.
        """
        histories = self._load_histories(history_dir)
        if not histories:
            raise ValueError(f"No training histories found in '{history_dir}'.")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette(n_colors=2)
        for field, label, color in (
            ("train_loss", "Train", colors[0]),
            ("val_loss", "Validation", colors[1]),
        ):
            self._plot_median_band(
                ax,
                self._series_matrix(histories, field),
                color=color,
                label=label,
                show_runs=True,
            )

        ax.axhline(
            self.CHANCE_LEVEL_LOSS,
            color="gray",
            linestyle=":",
            label="Chance level (ln 2)",
        )
        ax.set_title(
            f"{prefix.capitalize()} Loss Across {len(histories)} Seeds (median and IQR)"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        save_path = os.path.join(self.output_dir, f"{prefix}_loss_distribution.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_loss_comparison(self, history_dirs: dict[str, str]) -> None:
        """Overlays median validation loss curves across paradigms.

        Directories without any history file are skipped, so the plot can be
        regenerated while only some paradigms have been trained.

        Args:
            history_dirs: Mapping from a display label (for example "CTL") to
                the directory holding that paradigm's history JSON files.

        Raises:
            ValueError: If none of the directories contains a history file.
        """
        loaded = {
            label: histories
            for label, directory in history_dirs.items()
            if (histories := self._load_histories(directory))
        }
        if not loaded:
            raise ValueError("No training histories found in any provided directory.")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette(n_colors=len(loaded))
        for (label, histories), color in zip(loaded.items(), colors, strict=True):
            self._plot_median_band(
                ax,
                self._series_matrix(histories, "val_loss"),
                color=color,
                label=f"{label} (n={len(histories)})",
                show_runs=False,
            )

        ax.axhline(
            self.CHANCE_LEVEL_LOSS,
            color="gray",
            linestyle=":",
            label="Chance level (ln 2)",
        )
        ax.set_title("Validation Loss Across Paradigms (median and IQR)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.legend()

        save_path = os.path.join(self.output_dir, "loss_comparison.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _load_histories(history_dir: str) -> list[TrainingHistory]:
        """Loads every `*.json` training history in a directory.

        Args:
            history_dir: Directory holding history JSON files.

        Returns:
            Histories sorted by file name. Empty if the directory is missing
            or holds no JSON files.
        """
        return [
            MetricsAnalyzer.load_history(str(path))
            for path in sorted(Path(history_dir).glob("*.json"))
        ]

    @staticmethod
    def _series_matrix(histories: list[TrainingHistory], field: str) -> np.ndarray:
        """Stacks one per-epoch field of several histories into a matrix.

        Args:
            histories: Training histories, one per seed.
            field: Name of the `EpochRecord` attribute to extract.

        Returns:
            A `(n_runs, max_epochs)` array where epochs a run did not reach, or
            did not evaluate, are NaN.
        """
        matrix = np.full(
            (len(histories), max(len(h.epochs) for h in histories)), np.nan
        )
        for row, history in enumerate(histories):
            for record in history.epochs:
                value = getattr(record, field)
                if value is not None:
                    matrix[row, record.epoch - 1] = value
        return matrix

    @staticmethod
    def _plot_median_band(
        ax: Axes,
        matrix: np.ndarray,
        color: tuple[float, float, float],
        label: str,
        show_runs: bool,
    ) -> None:
        """Draws the per-epoch median and interquartile band of a series matrix.

        Epochs where every run is NaN (for example unevaluated epochs) are
        dropped, so the median line connects only observed epochs. Draws
        nothing if the matrix holds no value at all.

        Args:
            ax: Axes to draw on.
            matrix: `(n_runs, n_epochs)` array from `_series_matrix`.
            color: RGB color shared by the line, band, and per-run lines.
            label: Legend label of the median line.
            show_runs: Whether to draw each run as a faint background line.
        """
        observed = ~np.isnan(matrix).all(axis=0)
        if not observed.any():
            return

        epochs = np.flatnonzero(observed) + 1
        values = matrix[:, observed]

        if show_runs:
            for row in values:
                finite = ~np.isnan(row)
                ax.plot(
                    epochs[finite],
                    row[finite],
                    color=color,
                    alpha=0.15,
                    linewidth=0.8,
                )

        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        ax.fill_between(epochs, q1, q3, color=color, alpha=0.25)
        ax.plot(epochs, np.nanmedian(values, axis=0), color=color, label=label)

    @staticmethod
    def _mark_lr_decays(ax: Axes, history: TrainingHistory) -> None:
        """Draws dashed vertical lines where the StepLR schedule decays.

        Uses `lr_step_size` from the history config and draws nothing when it
        is absent. Each line sits between the last epoch at the old rate and
        the first at the new one.

        Args:
            ax: Axes to draw on.
            history: Training history providing the config and epoch count.
        """
        step_size = history.config.get("lr_step_size")
        if not step_size:
            return

        for index, boundary in enumerate(
            range(step_size, len(history.epochs), step_size)
        ):
            ax.axvline(
                boundary + 0.5,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="LR decay" if index == 0 else None,
            )

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
