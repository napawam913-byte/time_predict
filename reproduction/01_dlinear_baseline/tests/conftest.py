import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ettm1_csv(tmp_path):
    dates = pd.date_range("2024-01-01 00:00", periods=12, freq="15min")
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "a": np.arange(12, dtype=float),
            "b": np.arange(12, dtype=float) * 10.0,
        }
    )
    path = tmp_path / "ETTm1.csv"
    frame.to_csv(path, index=False)
    return path
