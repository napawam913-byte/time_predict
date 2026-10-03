"""Forecast metrics evaluated only after returning to the raw data scale."""

import numpy as np

from ltsf_baseline.data import Standardizer


def original_scale_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    scaler: Standardizer,
    column_names: list[str],
) -> dict[str, float | dict[str, float]]:
    """Inverse-transform equally shaped forecast arrays before calculating errors."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("prediction and target must have shape [batch, time, channels]")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.shape[-1] != len(column_names):
        raise ValueError("column names must match the forecast channel count")

    raw_prediction = scaler.inverse_transform(prediction)
    raw_target = scaler.inverse_transform(target)
    error = raw_prediction - raw_target
    mae_by_channel = np.mean(np.abs(error), axis=(0, 1))
    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "per_variable_mae": {
            column: float(value) for column, value in zip(column_names, mae_by_channel, strict=True)
        },
    }
