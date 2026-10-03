"""Reference-compatible shared-channel DLinear for multivariate LSTF."""

import torch
from torch import nn


class MovingAverage(nn.Module):
    """Endpoint-replicated moving average over the time dimension."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel size must be a positive odd number")
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("moving average values must have shape [batch, time, channels]")
        if values.shape[1] == 0:
            raise ValueError("moving average cannot operate on an empty time dimension")

        if self.padding:
            beginning = values[:, :1, :].repeat(1, self.padding, 1)
            end = values[:, -1:, :].repeat(1, self.padding, 1)
            values = torch.cat((beginning, values, end), dim=1)
        return self.pool(values.permute(0, 2, 1)).permute(0, 2, 1)


def decompose(values: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return residual seasonal component and endpoint-smoothed trend."""

    trend = MovingAverage(kernel_size)(values)
    return values - trend, trend


class DLinear(nn.Module):
    """DLinear with one linear seasonal and trend projection per shared channel set."""

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        channels: int,
        individual: bool = False,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        if input_length <= 0 or prediction_length <= 0 or channels <= 0:
            raise ValueError("input length, prediction length, and channels must be positive")
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.channels = channels
        self.individual = individual
        self.moving_average = MovingAverage(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                nn.Linear(input_length, prediction_length) for _ in range(channels)
            )
            self.linear_trend = nn.ModuleList(
                nn.Linear(input_length, prediction_length) for _ in range(channels)
            )
        else:
            self.linear_seasonal = nn.Linear(input_length, prediction_length)
            self.linear_trend = nn.Linear(input_length, prediction_length)
        self._initialize_linear_layers()

    def _initialize_linear_layers(self) -> None:
        layers = list(self.linear_seasonal) + list(self.linear_trend) if self.individual else [
            self.linear_seasonal,
            self.linear_trend,
        ]
        for layer in layers:
            nn.init.constant_(layer.weight, 1.0 / self.input_length)
            nn.init.zeros_(layer.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self._validate_input(values)
        trend = self.moving_average(values)
        seasonal = values - trend
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            outputs = [
                self.linear_seasonal[channel](seasonal[:, channel, :])
                + self.linear_trend[channel](trend[:, channel, :])
                for channel in range(self.channels)
            ]
            combined = torch.stack(outputs, dim=1)
        else:
            combined = self.linear_seasonal(seasonal) + self.linear_trend(trend)
        return combined.permute(0, 2, 1)

    def _validate_input(self, values: torch.Tensor) -> None:
        if values.ndim != 3:
            raise ValueError("DLinear input must have shape [batch, time, channels]")
        if values.shape[1] != self.input_length:
            raise ValueError(
                f"DLinear expected input length {self.input_length}, got {values.shape[1]}"
            )
        if values.shape[2] != self.channels:
            raise ValueError(
                f"DLinear expected {self.channels} channels, got {values.shape[2]}"
            )
