"""Telemetry, metric aggregation, logging, and publication-ready visualization.

This module provides structured data transfer objects (DTOs), metrics evaluation
and multi-run aggregation utilities, console/file loggers, and plotting tools
for ROC curves, metric distributions, and confusion matrices.
"""

from .logger import setup_logger
from .metrics import AggregateMetrics, EvaluationResult, MetricsAnalyzer
from .visualizer import MetricsVisualizer

__all__ = [
    "AggregateMetrics",
    "EvaluationResult",
    "MetricsAnalyzer",
    "MetricsVisualizer",
    "setup_logger",
]
