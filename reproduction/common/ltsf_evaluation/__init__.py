"""Scale-safe, model-neutral utilities for long-term forecasting comparisons."""

from .core import (
    ForecastResult,
    TrainingScaler,
    assert_aligned_labels,
    evaluate_models,
    load_normalized_archive,
    metric_summary,
)

__all__ = [
    "ForecastResult",
    "TrainingScaler",
    "assert_aligned_labels",
    "evaluate_models",
    "load_normalized_archive",
    "metric_summary",
]
