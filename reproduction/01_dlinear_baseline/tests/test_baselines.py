import torch

from ltsf_baseline.baselines import seasonal_naive


def test_seasonal_naive_repeats_the_last_complete_period():
    history = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    prediction = seasonal_naive(history, prediction_length=4, period=4)

    assert torch.equal(prediction, history)
