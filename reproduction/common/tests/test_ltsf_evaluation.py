"""Model-neutral scale and label-alignment contracts for LSTF results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ltsf_evaluation import (
    ForecastResult,
    TrainingScaler,
    assert_aligned_labels,
    evaluate_models,
    load_normalized_archive,
)


def _write_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "HUFL": [1.0, 3.0, 5.0, 7.0],
            "OT": [10.0, 30.0, 50.0, 70.0],
        }
    ).to_csv(path, index=False)


def test_evaluate_models_names_the_candidate_model_without_autoformer_coupling(tmp_path: Path) -> None:
    """A PatchTST candidate gets its own name while metrics retain both scales."""

    csv_path = tmp_path / "ETTm1.csv"
    _write_csv(csv_path)
    scaler = TrainingScaler.from_csv(csv_path, train_end=2)
    target_raw = np.array([[[3.0, 30.0]]])
    prediction_raw = np.array([[[4.0, 20.0]]])
    candidate = ForecastResult(
        scaler.normalize(prediction_raw), scaler.normalize(target_raw), "normalized"
    )
    baseline = ForecastResult(prediction_raw, target_raw, "original_scale")

    metrics = evaluate_models(
        "PatchTST",
        candidate,
        {"DLinear": baseline},
        scaler,
        ["HUFL", "OT"],
    )

    assert set(metrics) == {"PatchTST", "DLinear"}
    assert metrics["PatchTST"]["normalized"]["mse"] == pytest.approx(1.0)
    assert metrics["PatchTST"]["original_scale"]["mse"] == pytest.approx(50.5)


def test_normalized_archive_requires_declared_normalized_scale_and_column_order(tmp_path: Path) -> None:
    """A new model archive cannot silently change either scale or channel meaning."""

    archive_path = tmp_path / "patchtst.npz"
    values = np.zeros((1, 2, 2))
    np.savez_compressed(
        archive_path,
        prediction=values,
        target=values,
        columns=np.array(["HUFL", "OT"]),
        scale=np.array("normalized"),
    )

    result, columns = load_normalized_archive(
        archive_path, expected_columns=["HUFL", "OT"], require_scale=True
    )
    assert result.scale == "normalized"
    assert columns == ["HUFL", "OT"]

    with pytest.raises(ValueError, match="column order"):
        load_normalized_archive(
            archive_path, expected_columns=["OT", "HUFL"], require_scale=True
        )

    wrong_scale_path = tmp_path / "wrong-scale.npz"
    np.savez_compressed(
        wrong_scale_path,
        prediction=values,
        target=values,
        columns=np.array(["HUFL", "OT"]),
        scale=np.array("original_scale"),
    )
    with pytest.raises(ValueError, match="normalized"):
        load_normalized_archive(wrong_scale_path, require_scale=True)


def test_label_alignment_rejects_a_single_mismatched_normalized_target(tmp_path: Path) -> None:
    """One shifted value must prevent a misleading cross-model ranking."""

    csv_path = tmp_path / "ETTm1.csv"
    _write_csv(csv_path)
    scaler = TrainingScaler.from_csv(csv_path, train_end=2)
    target_raw = np.array([[[3.0, 30.0]]])
    candidate = ForecastResult(scaler.normalize(target_raw), scaler.normalize(target_raw), "normalized")
    shifted = ForecastResult(
        scaler.normalize(target_raw), scaler.normalize(target_raw + np.array([[[0.0, 1.0]]])), "normalized"
    )

    with pytest.raises(ValueError, match="labels do not align"):
        assert_aligned_labels(candidate, {"DLinear": shifted}, scaler)
