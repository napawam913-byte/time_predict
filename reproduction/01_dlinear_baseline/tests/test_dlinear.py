import torch

from ltsf_baseline.dlinear import DLinear, decompose


def test_decomposition_recombines_exactly():
    values = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)

    seasonal, trend = decompose(values, kernel_size=5)

    assert torch.allclose(seasonal + trend, values)


def test_dlinear_maps_batch_time_channel_to_requested_horizon():
    model = DLinear(
        input_length=8,
        prediction_length=3,
        channels=2,
        individual=False,
        kernel_size=25,
    )

    assert model(torch.zeros(4, 8, 2)).shape == (4, 3, 2)
