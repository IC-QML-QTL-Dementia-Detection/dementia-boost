"""Telemetry, metric aggregation, logging, and model selection.

This module provides structured data transfer objects (DTOs), metrics evaluation
and multi-run aggregation utilities, console/file loggers, model selection on
validation metrics, and the comparative report. Plotting lives in
`dementia_boost.viz`.
"""

from .logger import setup_logger
from .metrics import AggregateMetrics, EvaluationResult, MetricsAnalyzer

__all__ = [
    "AggregateMetrics",
    "EvaluationResult",
    "MetricsAnalyzer",
    "setup_logger",
]
