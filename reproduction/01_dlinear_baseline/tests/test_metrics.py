import numpy as np

from ltsf_baseline.data import Standardizer
from ltsf_baseline.metrics import original_scale_metrics


def test_metrics_are_calculated_after_inverse_transform():
    scaler = Standardizer().fit(np.array([[8.0], [12.0]]))
    prediction = np.array([[[0.0], [1.0]]])
    target = np.array([[[1.0], [0.0]]])

    metrics = original_scale_metrics(prediction, target, scaler, ["OT"])

    assert metrics["mse"] == 4.0
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert metrics["per_variable_mae"] == {"OT": 2.0}
