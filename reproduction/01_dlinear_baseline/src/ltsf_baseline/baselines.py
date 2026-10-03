"""Simple forecasting baselines used to calibrate the DLinear experiment."""

import torch


def seasonal_naive(
    history: torch.Tensor, prediction_length: int, period: int
) -> torch.Tensor:
    """Repeat the last fully observed seasonal cycle without recursive rollout."""

    if history.ndim != 3:
        raise ValueError("seasonal naive history must have shape [batch, time, channels]")
    if prediction_length <= 0 or period <= 0:
        raise ValueError("prediction length and period must be positive")
    if period > history.shape[1] or prediction_length % period != 0:
        raise ValueError("prediction length must be a whole available seasonal period")
    return history[:, -period:, :].repeat(1, prediction_length // period, 1)
