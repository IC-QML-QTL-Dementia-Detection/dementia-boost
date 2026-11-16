"""Readable names for plots meant for presentation (the `production` mode).

The default labels of the plots are the full configuration labels
(`qiskit_qtl | zz_feature_map | 6q x 4L | lr 0.0001`), which say exactly what
ran. For a poster or a paper the legends use the short names below instead, each
word capitalised. Edit the tables to rename; a name that is not in a table is
shown as it is.
"""

from collections.abc import Mapping
from typing import Any

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

BLOCK_NAMES: dict[str, str] = {
    "paper": "Paper Block",
}

REUPLOAD_NAME: str = "Re-uploading"

METRIC_NAMES: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "auc": "AUC-ROC",
}


def production_label(spec: Mapping[str, Any]) -> str:
    """Builds the short presentation name of a configuration.

    The variational block and re-uploading are named only when they differ
    from the default (the base paper's block, embedding applied once).

    Args:
        spec: A serialised run specification (`RunSpec.to_dict`), with or
            without the seed.

    Returns:
        For example `QTL (Qiskit), ZZ-Feature Map`, or `CTL`.
    """
    paradigm: str = spec["paradigm"]
    parts: list[str] = [PARADIGM_NAMES.get(paradigm, paradigm)]
    ansatz: str | None = spec.get("ansatz")
    if ansatz is not None:
        parts.append(EMBEDDING_NAMES.get(ansatz, ansatz))
    block: str | None = spec.get("block")
    if block is not None:
        parts.append(BLOCK_NAMES.get(block, block))
    if spec.get("reupload"):
        parts.append(REUPLOAD_NAME)
    return ", ".join(parts)


def production_labels(
    configurations: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Maps configuration IDs to presentation names, keeping them unique.

    Two configurations that differ only in a hyperparameter the short name does
    not show would get the same name; their `config_id` is then appended to
    each, so a legend never merges two configurations.

    Args:
        configurations: Mapping of `config_id` to its serialised spec.

    Returns:
        Mapping of `config_id` to its presentation name.
    """
    names = {config: production_label(spec) for config, spec in configurations.items()}
    counts = {name: list(names.values()).count(name) for name in names.values()}
    return {
        config: name if counts[name] == 1 else f"{name} [{config}]"
        for config, name in names.items()
    }
