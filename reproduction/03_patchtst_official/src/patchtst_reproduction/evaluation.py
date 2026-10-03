"""PatchTST-specific artifact loading over the shared LSTF evaluation core."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ltsf_evaluation import (
    ForecastResult,
    TrainingScaler,
    evaluate_models as _evaluate_models,
    load_normalized_archive,
)


def load_patchtst_result(run_dir: str | Path) -> tuple[ForecastResult, list[str]]:
    """Load one PatchTST run's explicitly normalized prediction archive."""

    return load_normalized_archive(Path(run_dir) / "predictions.npz", require_scale=True)


def evaluate_models(
    patchtst: ForecastResult,
    baselines: Mapping[str, ForecastResult],
    scaler: TrainingScaler,
    columns: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Evaluate PatchTST and baselines under one shared label contract."""

    return _evaluate_models("PatchTST", patchtst, baselines, scaler, columns)
