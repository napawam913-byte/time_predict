"""Tests for scale-safe, label-verified model comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoformer_reproduction.evaluation import (
    ForecastResult,
    TrainingScaler,
    assert_aligned_labels,
    evaluate_models,
    load_baseline_result,
    load_official_result,
)


def _write_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "HUFL": [1.0, 3.0, 5.0, 7.0],
            "OT": [10.0, 30.0, 50.0, 70.0],
        }
    ).to_csv(path, index=False)


def test_evaluate_models_reports_both_scales_and_per_variable_metrics(tmp_path: Path) -> None:
    """One shared train-only scaler makes normalized and raw MSE auditable."""

    csv_path = tmp_path / "ETTm1.csv"
    _write_csv(csv_path)
    scaler = TrainingScaler.from_csv(csv_path, train_end=2)
    target_raw = np.array([[[3.0, 30.0]]])
    prediction_raw = np.array([[[4.0, 20.0]]])
    target_normalized = scaler.normalize(target_raw)
    prediction_normalized = scaler.normalize(prediction_raw)
    official = ForecastResult(prediction_normalized, target_normalized, "normalized")
    baseline = ForecastResult(prediction_raw, target_raw, "original_scale")

    metrics = evaluate_models(
        official,
        {"Seasonal Naive": baseline, "DLinear": baseline},
        scaler,
        ["HUFL", "OT"],
    )

    assert metrics["Autoformer"]["normalized"]["mse"] == pytest.approx(1.0)
    assert metrics["Autoformer"]["original_scale"]["mse"] == pytest.approx(50.5)
    assert metrics["DLinear"]["per_variable"]["OT"]["mse"] == pytest.approx(100.0)
    assert metrics["Seasonal Naive"]["per_variable"]["HUFL"]["mae"] == pytest.approx(1.0)


def test_label_alignment_rejects_a_single_mismatched_target_value(tmp_path: Path) -> None:
    """The comparison must fail before ranking models with shifted test windows."""

    csv_path = tmp_path / "ETTm1.csv"
    _write_csv(csv_path)
    scaler = TrainingScaler.from_csv(csv_path, train_end=2)
    target_raw = np.array([[[3.0, 30.0]]])
    official = ForecastResult(scaler.normalize(target_raw), scaler.normalize(target_raw), "normalized")
    shifted_baseline = ForecastResult(target_raw, target_raw + np.array([[[0.0, 1.0]]]), "original_scale")

    with pytest.raises(ValueError, match="labels do not align"):
        assert_aligned_labels(official, {"DLinear": shifted_baseline}, scaler)


def test_loaders_reject_ambiguous_official_results_and_preserve_baseline_columns(tmp_path: Path) -> None:
    """Result discovery cannot silently select an arbitrary author output folder."""

    run_dir = tmp_path / "autoformer-run/results"
    for name in ("first", "second"):
        result_dir = run_dir / name
        result_dir.mkdir(parents=True)
        np.save(result_dir / "pred.npy", np.zeros((1, 1, 2)))
        np.save(result_dir / "true.npy", np.zeros((1, 1, 2)))

    with pytest.raises(ValueError, match="exactly one"):
        load_official_result(run_dir.parent)

    baseline_path = tmp_path / "baseline.npz"
    np.savez_compressed(
        baseline_path,
        prediction=np.zeros((1, 1, 2)),
        target=np.zeros((1, 1, 2)),
        columns=np.array(["HUFL", "OT"]),
    )
    result, columns = load_baseline_result(baseline_path)
    assert result.scale == "original_scale"
    assert columns == ["HUFL", "OT"]
