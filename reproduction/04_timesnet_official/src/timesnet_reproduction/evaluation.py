"""TimesNet artifact loading and strict shared LSTF comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ltsf_evaluation import (
    ForecastResult,
    TrainingScaler,
    evaluate_models as _evaluate_models,
    load_normalized_archive,
)


def load_timesnet_result(run_dir: str | Path) -> tuple[ForecastResult, list[str]]:
    """Load one TimesNet run's explicitly normalized prediction archive."""

    return load_normalized_archive(Path(run_dir) / "predictions.npz", require_scale=True)


def evaluate_models(
    timesnet: ForecastResult,
    baselines: Mapping[str, ForecastResult],
    scaler: TrainingScaler,
    columns: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Evaluate TimesNet only after shared label verification succeeds."""

    return _evaluate_models("TimesNet", timesnet, baselines, scaler, columns)


def compare_archives(
    *,
    timesnet_archive: str | Path,
    seasonal_archive: str | Path,
    dlinear_archive: str | Path,
    csv_path: str | Path,
    train_end: int = 34_560,
) -> dict[str, object]:
    """Build serializable metrics only after column and label alignment pass."""

    scaler = TrainingScaler.from_csv(csv_path, train_end=train_end)
    timesnet, timesnet_columns = load_normalized_archive(timesnet_archive, require_scale=True)
    seasonal_naive, seasonal_columns = load_normalized_archive(seasonal_archive)
    dlinear, dlinear_columns = load_normalized_archive(dlinear_archive)
    if timesnet_columns != scaler.columns:
        raise ValueError("TimesNet column order does not match the ETTm1 CSV")
    if seasonal_columns != scaler.columns or dlinear_columns != scaler.columns:
        raise ValueError("baseline column order does not match the ETTm1 CSV")

    models = evaluate_models(
        timesnet,
        {"Seasonal Naive": seasonal_naive, "DLinear": dlinear},
        scaler,
        scaler.columns,
    )
    return {
        "protocol": {
            "dataset": str(csv_path),
            "train_end": train_end,
            "input_length": 96,
            "prediction_length": int(timesnet.target.shape[1]),
            "channels": int(timesnet.target.shape[2]),
            "timesnet_source_commit": "2665a3143dae12d1cbcc31ddd396bbff48773bce",
            "timesnet_freq": "h",
        },
        "alignment": {
            "passed": True,
            "shape": list(timesnet.target.shape),
            "compared_against": ["Seasonal Naive", "DLinear"],
        },
        "models": models,
        "limitations": [
            "This is a single-run comparison; TimesNet uses the authors' fixed seed 2021.",
            "This project fixes L=96 and H=96; it is not a reproduction of a paper table using another context length.",
            "TimesNet uses the 2023 TSLib pin and freq=h, the implicit default of the authors' ETTm1 script.",
            "A plotted test window is qualitative; global test-set metrics determine the table.",
        ],
    }


__all__ = ["compare_archives", "evaluate_models", "load_timesnet_result"]
