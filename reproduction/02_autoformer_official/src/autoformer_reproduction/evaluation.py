"""Autoformer adapter for the shared scale-safe LSTF evaluation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ltsf_evaluation import (
    ForecastResult,
    TrainingScaler,
    assert_aligned_labels,
    evaluate_models as _evaluate_models,
    load_normalized_archive,
    metric_summary,
)


def load_official_result(run_dir: str | Path) -> ForecastResult:
    """Load the one normalized Autoformer result directory written by a run."""

    run_dir = Path(run_dir)
    candidates = [
        path
        for path in (run_dir / "results").glob("*")
        if path.is_dir() and (path / "pred.npy").is_file() and (path / "true.npy").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "official run must contain exactly one result directory with pred.npy and true.npy; "
            f"found {len(candidates)}"
        )
    return ForecastResult(
        np.load(candidates[0] / "pred.npy", allow_pickle=False),
        np.load(candidates[0] / "true.npy", allow_pickle=False),
        "normalized",
    )


def load_baseline_result(npz_path: str | Path) -> tuple[ForecastResult, list[str]]:
    """Load normalized predictions from the audited legacy baseline runner."""

    return load_normalized_archive(npz_path, require_scale=False)


def evaluate_models(
    official: ForecastResult,
    baselines: Mapping[str, ForecastResult],
    scaler: TrainingScaler,
    columns: list[str],
) -> dict[str, dict[str, object]]:
    """Keep Autoformer's comparison CLI API while using model-neutral metrics."""

    return _evaluate_models("Autoformer", official, baselines, scaler, columns)


__all__ = [
    "ForecastResult",
    "TrainingScaler",
    "assert_aligned_labels",
    "evaluate_models",
    "load_baseline_result",
    "load_official_result",
    "metric_summary",
]
