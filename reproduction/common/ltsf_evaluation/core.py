"""Explicit-scale forecast artifacts and label-verified metric computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


Scale = Literal["normalized", "original_scale"]


@dataclass(frozen=True)
class ForecastResult:
    """Prediction and target arrays on one declared common scale."""

    prediction: np.ndarray
    target: np.ndarray
    scale: Scale

    def __post_init__(self) -> None:
        prediction = _forecast_array(self.prediction, "prediction")
        target = _forecast_array(self.target, "target")
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {prediction.shape} does not match target shape {target.shape}"
            )
        if self.scale not in {"normalized", "original_scale"}:
            raise ValueError(f"unsupported forecast scale: {self.scale!r}")
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "target", target)


@dataclass(frozen=True)
class TrainingScaler:
    """Feature-wise standardization parameters fitted only on training rows."""

    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_csv(cls, csv_path: str | Path, train_end: int = 34560) -> "TrainingScaler":
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"ETTm1 CSV does not exist: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "date" not in frame.columns:
            raise ValueError("ETTm1 CSV requires a date column")
        columns = [column for column in frame.columns if column != "date"]
        if train_end <= 0 or len(frame) < train_end:
            raise ValueError(f"CSV needs at least {train_end} rows for train-only scaling")
        values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("ETTm1 values contain NaN or infinite values")
        mean = values[:train_end].mean(axis=0)
        scale = values[:train_end].std(axis=0)
        return cls(columns, mean, np.where(scale == 0.0, 1.0, scale))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = _array_with_channels(values, len(self.columns), "values")
        return (values - self.mean) / self.scale

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        values = _array_with_channels(values, len(self.columns), "values")
        return values * self.scale + self.mean


def load_normalized_archive(
    npz_path: str | Path,
    *,
    expected_columns: Sequence[str] | None = None,
    require_scale: bool = False,
) -> tuple[ForecastResult, list[str]]:
    """Load a normalized archive and reject scale or channel-order ambiguity.

    Legacy baseline archives do not declare ``scale``. They may be read with
    ``require_scale=False`` because their producer is separately audited. New
    model exporters must set ``require_scale=True`` and save ``scale=normalized``.
    """

    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"forecast archive does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"prediction", "target", "columns"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"forecast archive is missing arrays: {sorted(missing)}")
        if "scale" in archive.files:
            declared_scale = _scalar_string(archive["scale"], "scale")
            if declared_scale != "normalized":
                raise ValueError(
                    f"forecast archive must declare normalized scale, got {declared_scale!r}"
                )
        elif require_scale:
            raise ValueError("forecast archive must declare normalized scale")
        result = ForecastResult(archive["prediction"], archive["target"], "normalized")
        columns = [str(column) for column in archive["columns"].tolist()]
    if len(columns) != result.target.shape[-1]:
        raise ValueError("archive column count does not match prediction channels")
    if expected_columns is not None and columns != list(expected_columns):
        raise ValueError("archive column order does not match expected column order")
    return result, columns


def assert_aligned_labels(
    candidate: ForecastResult,
    baselines: Mapping[str, ForecastResult],
    scaler: TrainingScaler,
) -> None:
    """Reject comparison candidates that do not share the exact normalized labels."""

    candidate_target = _normalized_target(candidate, scaler)
    for name, result in baselines.items():
        target = _normalized_target(result, scaler)
        if target.shape != candidate_target.shape:
            raise ValueError(
                f"labels do not align for {name}: shape {target.shape} != {candidate_target.shape}"
            )
        if not np.allclose(target, candidate_target, rtol=1e-5, atol=1e-6):
            maximum = float(np.max(np.abs(target - candidate_target)))
            raise ValueError(f"labels do not align for {name}: max normalized difference={maximum}")


def metric_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute global MSE, MAE, and RMSE from identically shaped arrays."""

    prediction = _forecast_array(prediction, "prediction")
    target = _forecast_array(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    difference = prediction - target
    mse = float(np.mean(np.square(difference)))
    return {"mse": mse, "mae": float(np.mean(np.abs(difference))), "rmse": float(np.sqrt(mse))}


def evaluate_models(
    candidate_name: str,
    candidate: ForecastResult,
    baselines: Mapping[str, ForecastResult],
    scaler: TrainingScaler,
    columns: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Evaluate one named candidate and baselines after strict label verification."""

    if not candidate_name.strip():
        raise ValueError("candidate model name must not be empty")
    columns = list(columns)
    if columns != scaler.columns:
        raise ValueError("comparison columns do not match the train-scaler CSV column order")
    if candidate_name in baselines:
        raise ValueError(f"candidate model name duplicates baseline: {candidate_name}")
    assert_aligned_labels(candidate, baselines, scaler)
    named_results: dict[str, ForecastResult] = {candidate_name: candidate, **dict(baselines)}
    output: dict[str, dict[str, object]] = {}
    for name, result in named_results.items():
        normalized_prediction, normalized_target = _as_normalized(result, scaler)
        original_prediction, original_target = _as_original_scale(result, scaler)
        output[name] = {
            "normalized": metric_summary(normalized_prediction, normalized_target),
            "original_scale": metric_summary(original_prediction, original_target),
            "per_variable": {
                column: metric_summary(
                    original_prediction[..., index : index + 1],
                    original_target[..., index : index + 1],
                )
                for index, column in enumerate(columns)
            },
        }
    return output


def _as_normalized(result: ForecastResult, scaler: TrainingScaler) -> tuple[np.ndarray, np.ndarray]:
    if result.scale == "normalized":
        return result.prediction, result.target
    return scaler.normalize(result.prediction), scaler.normalize(result.target)


def _as_original_scale(result: ForecastResult, scaler: TrainingScaler) -> tuple[np.ndarray, np.ndarray]:
    if result.scale == "original_scale":
        return result.prediction, result.target
    return scaler.denormalize(result.prediction), scaler.denormalize(result.target)


def _normalized_target(result: ForecastResult, scaler: TrainingScaler) -> np.ndarray:
    return result.target if result.scale == "normalized" else scaler.normalize(result.target)


def _forecast_array(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [windows, horizon, channels], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _array_with_channels(values: np.ndarray, channels: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != channels:
        raise ValueError(f"{name} channel count does not match the train scaler")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _scalar_string(values: np.ndarray, name: str) -> str:
    values = np.asarray(values)
    if values.size != 1:
        raise ValueError(f"archive {name} must contain one value")
    return str(values.item())
