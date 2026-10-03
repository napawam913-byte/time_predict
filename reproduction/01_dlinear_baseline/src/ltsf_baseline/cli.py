"""Command-line entry point for reproducible ETTm1 baseline runs."""

import argparse
import json
from pathlib import Path

from ltsf_baseline.runner import run_experiment


def load_config(config_path: str | Path) -> dict[str, object]:
    """Load JSON and resolve a relative dataset path from the config file location."""

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config JSON root must be an object")
    dataset_path = config.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("config dataset_path must be a non-empty string")
    resolved_dataset = Path(dataset_path)
    if not resolved_dataset.is_absolute():
        resolved_dataset = config_path.parent / resolved_dataset
    config["dataset_path"] = str(resolved_dataset.resolve())
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a leakage-safe ETTm1 baseline on CPU.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment configuration.")
    parser.add_argument("--run-name", required=True, help="New immutable name under runs/.")
    parser.add_argument("--project-root", required=True, help="Project directory that contains runs/.")
    args = parser.parse_args(argv)

    metrics = run_experiment(
        load_config(args.config),
        project_root=Path(args.project_root).resolve(),
        run_name=args.run_name,
    )
    print(json.dumps({key: metrics[key] for key in ("mse", "mae", "rmse")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
