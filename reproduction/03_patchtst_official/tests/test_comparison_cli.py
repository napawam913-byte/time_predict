"""End-to-end contracts for label-verified PatchTST comparison artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from patchtst_reproduction.evaluation import load_patchtst_result


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
    patchtst_run = tmp_path / "patchtst-run"
    patchtst_run.mkdir()
    np.savez_compressed(
        patchtst_run / "predictions.npz",
        prediction=prediction,
        target=target,
        columns=np.array(["HUFL", "OT"]),
        scale=np.array("normalized"),
    )
    horizon_target = target[:, :0, :] if changed_horizon else target + np.array([[[0.0, 0.1 if mismatch else 0.0]]])
    baseline_prediction = prediction[:, :0, :] if changed_horizon else prediction
    seasonal = tmp_path / "seasonal.npz"
    dlinear = tmp_path / "dlinear.npz"
    columns = np.array(["OT", "HUFL"] if swapped_columns else ["HUFL", "OT"])
    for path in (seasonal, dlinear):
        np.savez_compressed(path, prediction=baseline_prediction, target=horizon_target, columns=columns)
    return csv_path, patchtst_run, seasonal, dlinear


def _invoke(tmp_path: Path, **kwargs: bool) -> tuple[subprocess.CompletedProcess[str], Path]:
    csv_path, patchtst_run, seasonal, dlinear = _write_inputs(tmp_path, **kwargs)
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
            "--patchtst-run",
            str(patchtst_run),
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


def test_patchtst_result_loader_requires_declared_normalized_scale(tmp_path: Path) -> None:
    """New PatchTST artifacts cannot retain the legacy baseline scale ambiguity."""

    _, patchtst_run, _, _ = _write_inputs(tmp_path)
    result, columns = load_patchtst_result(patchtst_run)

    assert result.scale == "normalized"
    assert columns == ["HUFL", "OT"]


def test_comparison_cli_writes_metrics_after_verified_label_alignment(tmp_path: Path) -> None:
    """The report names PatchTST and exposes both scales plus per-variable MSE."""

    result, output_dir = _invoke(tmp_path)

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "ettm1_l96_h96_initial_metrics.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "ettm1_l96_h96_initial_metrics.md").read_text(encoding="utf-8")
    assert metrics["alignment"]["passed"] is True
    assert set(metrics["models"]) == {"PatchTST", "Seasonal Naive", "DLinear"}
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
def test_comparison_cli_refuses_invalid_baseline_artifacts(
    tmp_path: Path, kwargs: dict[str, bool], message: str
) -> None:
    """Changed labels, channel meaning, or horizons must fail before ranking models."""

    result, output_dir = _invoke(tmp_path, **kwargs)

    assert result.returncode != 0
    assert message in result.stderr
    assert not (output_dir / "ettm1_l96_h96_initial_metrics.json").exists()
