"""Readable names for plots meant for presentation (the `production` mode).

The default labels of the plots are the full configuration labels
(`qiskit_qtl | zz_feature_map | 6q x 4L | lr 0.0001`), which say exactly what
ran. For a poster or a paper the legends use the short names below instead, each
word capitalised. Edit the tables to rename; a name that is not in a table is
shown as it is.
"""

from collections.abc import Mapping

from dementia_boost.core.identity import Paradigm

PARADIGM_NAMES: dict[str, str] = {
    Paradigm.BASELINE: "Baseline",
    Paradigm.CTL: "CTL",
    Paradigm.PL_QTL: "QTL (PennyLane)",
    Paradigm.QISKIT_QTL: "QTL (Qiskit)",
}

EMBEDDING_NAMES: dict[str, str] = {
    "paper": "Angle Embedding",
    "rx": "RX Embedding",
    "zz_feature_map": "ZZ-Feature Map",
    "zz_feature_map_linear": "ZZ-Feature Map (Linear)",
}

METRIC_NAMES: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "auc": "AUC-ROC",
}


def production_label(paradigm: str, ansatz: str | None) -> str:
    """Builds the short presentation name of a configuration.

    Args:
        paradigm: The paradigm name (a `Paradigm` value).
        ansatz: The embedding name of a quantum head, or None for the baseline
            and CTL.

    Returns:
        For example `QTL (Qiskit), ZZ-Feature Map`, or `CTL`.
    """
    name = PARADIGM_NAMES.get(paradigm, paradigm)
    if ansatz is None:
        return name
    return f"{name}, {EMBEDDING_NAMES.get(ansatz, ansatz)}"


def production_labels(
    configurations: Mapping[str, tuple[str, str | None]],
) -> dict[str, str]:
    """Maps configuration IDs to presentation names, keeping them unique.

    Two configurations that differ only in a hyperparameter the short name does
    not show would get the same name; their `config_id` is then appended to
    each, so a legend never merges two configurations.

    Args:
        configurations: Mapping of `config_id` to `(paradigm, ansatz)`.

    Returns:
        Mapping of `config_id` to its presentation name.
    """
    names = {
        config: production_label(paradigm, ansatz)
        for config, (paradigm, ansatz) in configurations.items()
    }
    counts = {name: list(names.values()).count(name) for name in names.values()}
    return {
        config: name if counts[name] == 1 else f"{name} [{config}]"
        for config, name in names.items()
    }
