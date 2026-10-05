#!/usr/bin/env python3
"""Create a label-verified ETTm1 96 -> 96 TimesNet comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for source in (
    PROJECT_ROOT / "reproduction" / "common",
    PROJECT_ROOT / "reproduction" / "04_timesnet_official" / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from ltsf_evaluation import TrainingScaler, load_normalized_archive
from timesnet_reproduction.evaluation import compare_archives, load_timesnet_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesnet-run", type=Path, required=True)
    parser.add_argument("--seasonal-naive", type=Path, required=True)
    parser.add_argument("--dlinear", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-end", type=int, default=34_560)
    parser.add_argument("--test-start", type=int, default=46_080)
    parser.add_argument("--input-length", type=int, default=96)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--variable", default="OT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timesnet_archive = args.timesnet_run / "predictions.npz"
    payload = compare_archives(
        timesnet_archive=timesnet_archive,
        seasonal_archive=args.seasonal_naive,
        dlinear_archive=args.dlinear,
        csv_path=args.csv,
        train_end=args.train_end,
    )
    scaler = TrainingScaler.from_csv(args.csv, train_end=args.train_end)
    timesnet, _ = load_timesnet_result(args.timesnet_run)
    seasonal_naive, _ = load_normalized_archive(args.seasonal_naive)
    dlinear, _ = load_normalized_archive(args.dlinear)
    if args.variable not in scaler.columns:
        raise ValueError(f"unknown plotting variable: {args.variable}")
    if not 0 <= args.window_index < timesnet.target.shape[0]:
        raise ValueError(f"window index is outside [0, {timesnet.target.shape[0]})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ettm1_l96_h96_initial_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "ettm1_l96_h96_initial_metrics.md").write_text(
        _metrics_markdown(payload, scaler.columns), encoding="utf-8"
    )
    _plot_window(
        args.csv,
        scaler,
        timesnet,
        seasonal_naive,
        dlinear,
        args.test_start,
        args.input_length,
        args.window_index,
        args.variable,
        args.output_dir / "ettm1_l96_h96_initial_prediction.png",
    )


def _metrics_markdown(payload: dict[str, object], columns: list[str]) -> str:
    models = payload["models"]
    assert isinstance(models, dict)
    lines = [
        "# ETTm1 `96 → 96` 初步比较",
        "",
        "标签对齐：**通过**。下表的每个模型均在同一测试窗口、同一标签上计算。",
        "",
        "| 模型 | 标准化 MSE | 标准化 MAE | 原始尺度 MSE | 原始尺度 MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in models.items():
        assert isinstance(result, dict)
        normalized = result["normalized"]
        original = result["original_scale"]
        assert isinstance(normalized, dict) and isinstance(original, dict)
        lines.append(
            f"| {name} | {normalized['mse']:.6f} | {normalized['mae']:.6f} | "
            f"{original['mse']:.6f} | {original['mae']:.6f} |"
        )
    lines.extend(["", "## 各变量原始尺度 MSE", "", "| 模型 | " + " | ".join(columns) + " |"])
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")
    for name, result in models.items():
        assert isinstance(result, dict)
        per_variable = result["per_variable"]
        assert isinstance(per_variable, dict)
        values = " | ".join(f"{per_variable[column]['mse']:.6f}" for column in columns)
        lines.append(f"| {name} | {values} |")
    lines.extend(["", "## 限制", ""])
    limitations = payload["limitations"]
    assert isinstance(limitations, list)
    lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def _plot_window(
    csv_path: Path,
    scaler: TrainingScaler,
    timesnet,
    seasonal_naive,
    dlinear,
    test_start: int,
    input_length: int,
    window_index: int,
    variable: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(csv_path)
    values = frame[scaler.columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    variable_index = scaler.columns.index(variable)
    target_start = test_start + window_index
    history = values[target_start - input_length : target_start, variable_index]
    actual = scaler.denormalize(timesnet.target)[window_index, :, variable_index]
    timesnet_prediction = scaler.denormalize(timesnet.prediction)[window_index, :, variable_index]
    seasonal_prediction = scaler.denormalize(seasonal_naive.prediction)[window_index, :, variable_index]
    dlinear_prediction = scaler.denormalize(dlinear.prediction)[window_index, :, variable_index]

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(np.arange(input_length), history, label="Observed history", color="black")
    future_x = np.arange(input_length, input_length + len(actual))
    axis.plot(future_x, actual, label="Ground truth", color="tab:red")
    axis.plot(future_x, timesnet_prediction, label="TimesNet", color="tab:blue")
    axis.plot(future_x, seasonal_prediction, label="Seasonal Naive", color="tab:green")
    axis.plot(future_x, dlinear_prediction, label="DLinear", color="tab:purple")
    axis.axvline(input_length - 0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_title(f"ETTm1 {variable}: verified test window {window_index}")
    axis.set_xlabel("Time step")
    axis.set_ylabel("Original scale")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
