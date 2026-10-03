"""End-to-end checks for the artifact-producing comparison command."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = EXPERIMENT_ROOT / ".venv/bin/python"
SCRIPT = EXPERIMENT_ROOT / "scripts/compare_ettm1_l96_h96.py"


def _write_inputs(tmp_path: Path, mismatch: bool = False) -> tuple[Path, Path, Path, Path]:
    csv_path = tmp_path / "ETTm1.csv"
    pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "HUFL": [1.0, 3.0, 5.0, 7.0],
            "OT": [10.0, 30.0, 50.0, 70.0],
        }
    ).to_csv(csv_path, index=False)
    official_dir = tmp_path / "official/results/setting"
    official_dir.mkdir(parents=True)
    np.save(official_dir / "true.npy", np.array([[[1.0, 3.0]]]))
    np.save(official_dir / "pred.npy", np.array([[[2.0, 2.0]]]))
    target = np.array([[[3.0, 50.0 + (1.0 if mismatch else 0.0)]]])
    prediction = np.array([[[4.0, 40.0]]])
    seasonal = tmp_path / "seasonal.npz"
    dlinear = tmp_path / "dlinear.npz"
    for path in (seasonal, dlinear):
        np.savez_compressed(
            path,
            prediction=prediction,
            target=target,
            columns=np.array(["HUFL", "OT"]),
        )
    return csv_path, official_dir.parents[1], seasonal, dlinear


def _invoke(tmp_path: Path, mismatch: bool = False) -> tuple[subprocess.CompletedProcess[str], Path]:
    csv_path, official_dir, seasonal, dlinear = _write_inputs(tmp_path, mismatch=mismatch)
    output_dir = tmp_path / "comparison"
    environment = os.environ | {
        "PYTHONPATH": str(EXPERIMENT_ROOT / "src"),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib-cache"),
    }
    result = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--autoformer-run",
            str(official_dir),
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


def test_comparison_cli_writes_mse_table_only_after_label_alignment(tmp_path: Path) -> None:
    """The report makes all three models' MSE visible after an actual alignment check."""

    result, output_dir = _invoke(tmp_path)

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "ettm1_l96_h96_initial_metrics.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "ettm1_l96_h96_initial_metrics.md").read_text(encoding="utf-8")
    assert metrics["alignment"]["passed"] is True
    assert set(metrics["models"]) == {"Autoformer", "Seasonal Naive", "DLinear"}
    assert "标准化 MSE" in markdown
    assert (output_dir / "ettm1_l96_h96_initial_prediction.png").is_file()


def test_comparison_cli_refuses_to_write_metrics_for_misaligned_labels(tmp_path: Path) -> None:
    """A shifted baseline must fail rather than receive a misleading MSE ranking."""

    result, output_dir = _invoke(tmp_path, mismatch=True)

    assert result.returncode != 0
    assert "labels do not align" in result.stderr
    assert not (output_dir / "ettm1_l96_h96_initial_metrics.json").exists()
