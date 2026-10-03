"""Chronological ETTm1 loading, train-only scaling, and forecasting windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitBoundaries:
    """Exclusive end indices for chronological train, validation, and test targets."""

    train_end: int
    validation_end: int
    test_end: int


class Standardizer:
    """Feature-wise population standardization fitted on a training matrix only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("standardizer expects a non-empty [time, channels] array")
        if not np.isfinite(values).all():
            raise ValueError("standardizer cannot fit NaN or infinite values")
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean, scale = self._parameters()
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != mean.shape[0]:
            raise ValueError("value channel count does not match fitted standardizer")
        return (values - mean) / scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        mean, scale = self._parameters()
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != mean.shape[0]:
            raise ValueError("value channel count does not match fitted standardizer")
        return values * scale + mean

    def _parameters(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("standardizer must be fitted before transformation")
        return self.mean_, self.scale_


class WindowDataset:
    """Forecasting windows whose labels remain inside one target interval."""

    def __init__(
        self,
        values: np.ndarray,
        input_length: int,
        prediction_length: int,
        target_begin: int,
        target_end: int,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("window values must have shape [time, channels]")
        if input_length <= 0 or prediction_length <= 0:
            raise ValueError("input and prediction lengths must be positive")
        if target_begin < input_length or target_end > len(values) or target_begin >= target_end:
            raise ValueError("target interval is outside the available chronological data")

        last_start = target_end - prediction_length
        if last_start < target_begin:
            raise ValueError("target interval cannot produce a complete forecasting window")

        self.values = values
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.target_starts = np.arange(target_begin, last_start + 1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        target_start = int(self.target_starts[index])
        history = self.values[target_start - self.input_length : target_start]
        target = self.values[target_start : target_start + self.prediction_length]
        return history, target


@dataclass(frozen=True)
class PreparedETTm1:
    columns: list[str]
    scaler: Standardizer
    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    boundaries: SplitBoundaries


def prepare_ettm1(
    csv_path: str | Path,
    input_length: int,
    prediction_length: int,
    boundaries: SplitBoundaries | None = None,
) -> PreparedETTm1:
    """Validate ETTm1 CSV and return leakage-safe chronological windows."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"ETTm1 CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    _validate_timestamps(frame)
    values, columns = _numeric_values(frame)
    if not np.isfinite(values).all():
        raise ValueError("ETTm1 values contain NaN or infinite numbers")

    if boundaries is None:
        boundaries = SplitBoundaries(34560, 46080, 57600)
    _validate_boundaries(boundaries, len(values))

    scaler = Standardizer().fit(values[: boundaries.train_end])
    normalized = scaler.transform(values).astype(np.float32)

    train = WindowDataset(
        normalized,
        input_length,
        prediction_length,
        input_length,
        boundaries.train_end,
    )
    validation = WindowDataset(
        normalized,
        input_length,
        prediction_length,
        boundaries.train_end,
        boundaries.validation_end,
    )
    test = WindowDataset(
        normalized,
        input_length,
        prediction_length,
        boundaries.validation_end,
        boundaries.test_end,
    )
    return PreparedETTm1(columns, scaler, train, validation, test, boundaries)


def _validate_timestamps(frame: pd.DataFrame) -> None:
    if "date" not in frame.columns:
        raise ValueError("ETTm1 CSV requires a date column")
    timestamps = pd.to_datetime(frame["date"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("date column contains an unparseable timestamp")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("date timestamps must be strictly increasing")


def _numeric_values(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [column for column in frame.columns if column != "date"]
    if not columns:
        raise ValueError("ETTm1 CSV requires at least one numeric value column")
    try:
        numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("all non-date ETTm1 columns must be numeric") from error
    return numeric.to_numpy(dtype=np.float64), columns


def _validate_boundaries(boundaries: SplitBoundaries, row_count: int) -> None:
    if not (0 < boundaries.train_end < boundaries.validation_end < boundaries.test_end <= row_count):
        raise ValueError("split boundaries must be strictly increasing and inside the CSV")
