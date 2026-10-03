"""Scale-safe evaluation for the Autoformer and existing baseline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd


Scale = Literal["normalized", "original_scale"]


@dataclass(frozen=True)
class ForecastResult:
    """Prediction and label arrays on a declared common scale."""

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
    """Feature-wise standardization parameters fitted solely on train rows."""

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
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(columns, mean, scale)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = _array_with_channels(values, len(self.columns), "values")
        return (values - self.mean) / self.scale

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        values = _array_with_channels(values, len(self.columns), "values")
        return values * self.scale + self.mean


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
            f"official run must contain exactly one result directory with pred.npy and true.npy; found {len(candidates)}"
        )
    return ForecastResult(
        np.load(candidates[0] / "pred.npy", allow_pickle=False),
        np.load(candidates[0] / "true.npy", allow_pickle=False),
        "normalized",
    )


def load_baseline_result(npz_path: str | Path) -> tuple[ForecastResult, list[str]]:
    """Load normalized predictions produced by the existing baseline runner."""

    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"baseline prediction archive does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"prediction", "target", "columns"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"baseline archive is missing arrays: {sorted(missing)}")
        result = ForecastResult(archive["prediction"], archive["target"], "normalized")
        columns = [str(column) for column in archive["columns"].tolist()]
    if len(columns) != result.target.shape[-1]:
        raise ValueError("baseline column count does not match prediction channels")
    return result, columns


def assert_aligned_labels(
    official: ForecastResult,
    baselines: Mapping[str, ForecastResult],
    scaler: TrainingScaler,
) -> None:
    """Reject comparison candidates that do not share the exact normalized labels."""

    official_target = _normalized_target(official, scaler)
    for name, result in baselines.items():
        candidate = _normalized_target(result, scaler)
        if candidate.shape != official_target.shape:
            raise ValueError(
                f"labels do not align for {name}: shape {candidate.shape} != {official_target.shape}"
            )
        if not np.allclose(candidate, official_target, rtol=1e-5, atol=1e-6):
            maximum = float(np.max(np.abs(candidate - official_target)))
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
    official: ForecastResult,
    baselines: Mapping[str, ForecastResult],
    scaler: TrainingScaler,
    columns: list[str],
) -> dict[str, dict[str, object]]:
    """Return both scale views and per-variable errors after label verification."""

    if columns != scaler.columns:
        raise ValueError("comparison columns do not match the train-scaler CSV column order")
    assert_aligned_labels(official, baselines, scaler)
    named_results: dict[str, ForecastResult] = {"Autoformer": official, **dict(baselines)}
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
