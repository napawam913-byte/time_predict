import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ltsf_baseline.runner import _normalized_mse, run_experiment


@pytest.fixture
def tiny_csv(tmp_path):
    points = np.arange(140, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(points), freq="15min"),
            "load_a": np.sin(2 * np.pi * points / 8) + 0.01 * points,
            "load_b": np.cos(2 * np.pi * points / 8) - 0.02 * points,
        }
    )
    path = tmp_path / "tiny_ettm1.csv"
    frame.to_csv(path, index=False)
    return path


def tiny_config(
    dataset_path,
    model_name="dlinear",
    input_length=8,
    prediction_length=2,
    max_epochs=2,
    patience=1,
    batch_size=4,
    period=8,
):
    return {
        "dataset_path": str(dataset_path),
        "input_length": input_length,
        "prediction_length": prediction_length,
        "seed": 2026,
        "model": {
            "name": model_name,
            "individual": False,
            "moving_average_window": 5,
            "period": period,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "max_epochs": max_epochs,
            "patience": patience,
        },
        "split_boundaries": {"train_end": 80, "validation_end": 110, "test_end": 140},
    }


def test_dlinear_cpu_run_writes_reproducible_artifacts(tiny_csv, tmp_path):
    config = tiny_config(dataset_path=tiny_csv)

    result = run_experiment(config, project_root=tmp_path, run_name="tiny_dlinear")

    run_dir = tmp_path / "runs" / "tiny_dlinear"
    assert set(result) >= {"mse", "mae", "rmse", "per_variable_mae"}
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "history.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.npz").is_file()
    assert (run_dir / "best.pt").is_file()
    assert torch.load(run_dir / "best.pt", weights_only=True)["model_state"]


def test_existing_run_name_is_rejected(tiny_csv, tmp_path):
    config = tiny_config(dataset_path=tiny_csv, max_epochs=1)
    run_experiment(config, project_root=tmp_path, run_name="already_exists")

    with pytest.raises(FileExistsError, match="already_exists"):
        run_experiment(config, project_root=tmp_path, run_name="already_exists")


def test_unknown_model_does_not_create_an_empty_run_directory(tiny_csv, tmp_path):
    config = tiny_config(dataset_path=tiny_csv)
    config["model"]["name"] = "unknown_model"

    with pytest.raises(ValueError, match="unknown_model"):
        run_experiment(config, project_root=tmp_path, run_name="invalid_model")

    assert not (tmp_path / "runs" / "invalid_model").exists()


def test_validation_mse_weights_a_partial_final_batch_by_its_element_count():
    histories = torch.zeros(3, 1, 1)
    targets = torch.tensor([[[0.0]], [[0.0]], [[3.0]]])
    loader = DataLoader(TensorDataset(histories, targets), batch_size=2, shuffle=False)

    mse = _normalized_mse(nn.Identity(), loader, nn.MSELoss())

    assert mse == pytest.approx(3.0)


def test_seasonal_naive_run_has_metrics_but_no_checkpoint(tiny_csv, tmp_path):
    config = tiny_config(
        dataset_path=tiny_csv,
        model_name="seasonal_naive",
        input_length=8,
        prediction_length=8,
        period=8,
    )

    run_experiment(config, project_root=tmp_path, run_name="tiny_naive")

    run_dir = tmp_path / "runs" / "tiny_naive"
    assert (run_dir / "metrics.json").is_file()
    assert not (run_dir / "best.pt").exists()
