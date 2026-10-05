"""End-to-end contracts for the label-verified TimesNet comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from timesnet_reproduction.evaluation import compare_archives, load_timesnet_result


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PYTHON = PROJECT_ROOT / "reproduction" / "02_autoformer_official" / ".venv/bin/python"
SCRIPT = EXPERIMENT_ROOT / "scripts" / "compare_ettm1_l96_h96.py"


def _write_inputs(
    tmp_path: Path,
    *,
    mismatch: bool = False,
    swapped_columns: bool = False,
    changed_horizon: bool = False,
) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "ETTm1.csv"
    pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "HUFL": [1.0, 3.0, 5.0, 7.0],
            "OT": [10.0, 30.0, 50.0, 70.0],
        }
    ).to_csv(csv_path, index=False)
    prediction = np.array([[[2.0, 2.0]]])
    target = np.array([[[1.0, 3.0]]])
    timesnet_run = tmp_path / "timesnet-run"
    timesnet_run.mkdir()
    np.savez_compressed(
        timesnet_run / "predictions.npz",
        prediction=prediction,
        target=target,
        columns=np.array(["HUFL", "OT"]),
        scale=np.array("normalized"),
    )
    horizon_target = (
        target[:, :0, :]
        if changed_horizon
        else target + np.array([[[0.0, 0.1 if mismatch else 0.0]]])
    )
    baseline_prediction = prediction[:, :0, :] if changed_horizon else prediction
    seasonal = tmp_path / "seasonal.npz"
    dlinear = tmp_path / "dlinear.npz"
    columns = np.array(["OT", "HUFL"] if swapped_columns else ["HUFL", "OT"])
    for path in (seasonal, dlinear):
        np.savez_compressed(path, prediction=baseline_prediction, target=horizon_target, columns=columns)
    return csv_path, timesnet_run, seasonal, dlinear


def _invoke(tmp_path: Path, **kwargs: bool) -> tuple[subprocess.CompletedProcess[str], Path]:
    csv_path, timesnet_run, seasonal, dlinear = _write_inputs(tmp_path, **kwargs)
    output_dir = tmp_path / "comparison"
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            [str(PROJECT_ROOT / "reproduction" / "common"), str(EXPERIMENT_ROOT / "src")]
        ),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib-cache"),
    }
    result = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--timesnet-run",
            str(timesnet_run),
            "--seasonal-naive",
            str(seasonal),
            "--dlinear",
            str(dlinear),
            "--csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--train-end",
            "2",
            "--test-start",
            "2",
            "--input-length",
            "1",
            "--window-index",
            "0",
            "--variable",
            "OT",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output_dir


def test_timesnet_result_loader_requires_declared_normalized_scale(tmp_path: Path) -> None:
    """TimesNet artifacts must state their scale before cross-model comparison."""

    _, timesnet_run, _, _ = _write_inputs(tmp_path)
    result, columns = load_timesnet_result(timesnet_run)

    assert result.scale == "normalized"
    assert columns == ["HUFL", "OT"]


def test_comparison_writes_metrics_only_after_strict_label_alignment(tmp_path: Path) -> None:
    """The public function and CLI expose one successfully aligned comparison."""

    csv_path, timesnet_run, seasonal, dlinear = _write_inputs(tmp_path)
    payload = compare_archives(
        timesnet_archive=timesnet_run / "predictions.npz",
        seasonal_archive=seasonal,
        dlinear_archive=dlinear,
        csv_path=csv_path,
        train_end=2,
    )
    assert payload["alignment"]["passed"] is True
    assert set(payload["models"]) == {"TimesNet", "Seasonal Naive", "DLinear"}

    result, output_dir = _invoke(tmp_path / "cli")
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "ettm1_l96_h96_initial_metrics.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "ettm1_l96_h96_initial_metrics.md").read_text(encoding="utf-8")
    assert metrics["alignment"]["passed"] is True
    assert "标准化 MSE" in markdown
    assert "各变量原始尺度 MSE" in markdown
    assert (output_dir / "ettm1_l96_h96_initial_prediction.png").is_file()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mismatch": True}, "labels do not align"),
        ({"swapped_columns": True}, "column order"),
        ({"changed_horizon": True}, "labels do not align"),
    ],
)
def test_comparison_rejects_column_shape_and_value_misalignment(
    tmp_path: Path, kwargs: dict[str, bool], message: str
) -> None:
    """No report can rank forecasts that use changed labels or channel meaning."""

    result, output_dir = _invoke(tmp_path, **kwargs)

    assert result.returncode != 0
    assert message in result.stderr
    assert not (output_dir / "ettm1_l96_h96_initial_metrics.json").exists()
