#!/usr/bin/env python3
"""Create a label-verified MSE comparison for ETTm1 96 -> 96 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoformer_reproduction.evaluation import (
    TrainingScaler,
    evaluate_models,
    load_baseline_result,
    load_official_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autoformer-run", type=Path, required=True)
    parser.add_argument("--seasonal-naive", type=Path, required=True)
    parser.add_argument("--dlinear", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-end", type=int, default=34560)
    parser.add_argument("--test-start", type=int, default=46080)
    parser.add_argument("--input-length", type=int, default=96)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--variable", default="OT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaler = TrainingScaler.from_csv(args.csv, train_end=args.train_end)
    official = load_official_result(args.autoformer_run)
    seasonal_naive, seasonal_columns = load_baseline_result(args.seasonal_naive)
    dlinear, dlinear_columns = load_baseline_result(args.dlinear)
    if seasonal_columns != scaler.columns or dlinear_columns != scaler.columns:
        raise ValueError("baseline column order does not match the ETTm1 CSV")
    if args.variable not in scaler.columns:
        raise ValueError(f"unknown plotting variable: {args.variable}")

    models = evaluate_models(
        official,
        {"Seasonal Naive": seasonal_naive, "DLinear": dlinear},
        scaler,
        scaler.columns,
    )
    if not 0 <= args.window_index < official.target.shape[0]:
        raise ValueError(f"window index is outside [0, {official.target.shape[0]})")

    payload = {
        "protocol": {
            "dataset": str(args.csv),
            "train_end": args.train_end,
            "test_start": args.test_start,
            "input_length": args.input_length,
            "prediction_length": int(official.target.shape[1]),
            "columns": scaler.columns,
        },
        "alignment": {
            "passed": True,
            "shape": list(official.target.shape),
            "compared_against": ["Seasonal Naive", "DLinear"],
        },
        "models": models,
        "limitations": [
            "This is an initial single-run comparison; DLinear uses seed 2026 and Autoformer uses the authors' fixed seed 2021.",
            "A plotted test window is qualitative; global test-set metrics determine the table.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = args.output_dir / "ettm1_l96_h96_initial_metrics.json"
    metrics_markdown = args.output_dir / "ettm1_l96_h96_initial_metrics.md"
    plot_path = args.output_dir / "ettm1_l96_h96_initial_prediction.png"
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_markdown.write_text(_metrics_markdown(payload, scaler.columns), encoding="utf-8")
    _plot_window(
        args.csv,
        scaler,
        official,
        seasonal_naive,
        dlinear,
        args.test_start,
        args.input_length,
        args.window_index,
        args.variable,
        plot_path,
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
    official,
    seasonal_naive,
    dlinear,
    test_start: int,
    input_length: int,
    window_index: int,
    variable: str,
    output_path: Path,
) -> None:
    frame = pd.read_csv(csv_path)
    values = frame[scaler.columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    variable_index = scaler.columns.index(variable)
    target_start = test_start + window_index
    history = values[target_start - input_length : target_start, variable_index]
    actual = seasonal_naive.target[window_index, :, variable_index]
    autoformer_prediction = scaler.denormalize(official.prediction)[window_index, :, variable_index]
    seasonal_prediction = seasonal_naive.prediction[window_index, :, variable_index]
    dlinear_prediction = dlinear.prediction[window_index, :, variable_index]
    horizon = len(actual)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(np.arange(input_length), history, label="Observed history", color="black")
    future_x = np.arange(input_length, input_length + horizon)
    axis.plot(future_x, actual, label="Ground truth", color="tab:red")
    axis.plot(future_x, autoformer_prediction, label="Autoformer", color="tab:blue")
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
