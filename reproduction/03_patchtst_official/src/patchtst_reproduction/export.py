"""Export PatchTST test predictions and labels without modifying official files."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ltsf_evaluation import ForecastResult


@dataclass(frozen=True)
class OfficialArtifacts:
    """The unique official checkpoint/result pair belonging to one run setting."""

    checkpoint: Path
    result_dir: Path
    setting: str


def discover_official_artifacts(official_run_dir: str | Path) -> OfficialArtifacts:
    """Find exactly one matching official checkpoint and result directory."""

    official_run_dir = Path(official_run_dir)
    checkpoint_paths = sorted(official_run_dir.glob("checkpoints/*/checkpoint.pth"))
    if len(checkpoint_paths) != 1:
        raise ValueError(
            "official run must contain exactly one checkpoint at "
            f"checkpoints/<setting>/checkpoint.pth; found {len(checkpoint_paths)}"
        )
    result_dirs = sorted(
        path
        for path in (official_run_dir / "results").glob("*")
        if path.is_dir() and (path / "pred.npy").is_file()
    )
    if len(result_dirs) != 1:
        raise ValueError(
            "official run must contain exactly one result directory with pred.npy; "
            f"found {len(result_dirs)}"
        )
    checkpoint = checkpoint_paths[0]
    result_dir = result_dirs[0]
    setting = checkpoint.parent.name
    if result_dir.name != setting:
        raise ValueError(
            "official checkpoint setting does not match the result directory: "
            f"{setting!r} != {result_dir.name!r}"
        )
    return OfficialArtifacts(checkpoint, result_dir, setting)


def write_normalized_archive(
    archive_path: str | Path,
    prediction: np.ndarray,
    target: np.ndarray,
    columns: list[str],
) -> Path:
    """Write the complete normalized artifact required for fair comparison."""

    result = ForecastResult(prediction, target, "normalized")
    if len(columns) != result.target.shape[-1]:
        raise ValueError("column count does not match prediction channels")
    if len(set(columns)) != len(columns) or any(not column for column in columns):
        raise ValueError("columns must be unique non-empty names")
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive_path,
        prediction=result.prediction,
        target=result.target,
        columns=np.asarray(columns),
        scale=np.asarray("normalized"),
    )
    return archive_path


def export_from_official_run(
    *,
    upstream_supervised_dir: str | Path,
    official_run_dir: str | Path,
    config_path: str | Path,
    archive_path: str | Path,
) -> Path:
    """Run the official PatchTST test pathway and persist both normalized tensors."""

    artifacts = discover_official_artifacts(official_run_dir)
    args = _load_args(config_path)
    upstream_supervised_dir = Path(upstream_supervised_dir)
    if not (upstream_supervised_dir / "exp" / "exp_main.py").is_file():
        raise FileNotFoundError(f"PatchTST supervised source does not exist: {upstream_supervised_dir}")
    if args.model != "PatchTST":
        raise ValueError(f"exporter only supports PatchTST, got {args.model!r}")

    exp_main = _load_official_exp_main(upstream_supervised_dir)
    experiment = exp_main(args)
    _load_checkpoint(experiment, artifacts.checkpoint)
    test_loader = complete_test_loader(experiment)
    prediction, target = _collect_test_windows(experiment, test_loader)
    return write_normalized_archive(archive_path, prediction, target, _columns_for_args(args))


def _load_args(config_path: str | Path) -> Namespace:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"official run configuration does not exist: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise ValueError("official run configuration must contain an args object")
    return Namespace(**payload["args"])


def _load_official_exp_main(upstream_supervised_dir: Path) -> type[Any]:
    source = str(upstream_supervised_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from exp.exp_main import Exp_Main

    return Exp_Main


def _load_checkpoint(experiment: Any, checkpoint_path: Path) -> None:
    import torch

    state = torch.load(checkpoint_path, map_location=experiment.device)
    experiment.model.load_state_dict(state)


def complete_test_loader(experiment: Any) -> Any:
    """Build an official-dataset loader that preserves its final partial batch.

    The immutable upstream ``data_provider(..., "test")`` uses
    ``drop_last=True``. That silently omits the final 33 ETTm1 test windows at
    batch size 128, so this comparison-only exporter recreates the same test
    dataset with ``drop_last=False``. Model weights and the official source are
    not changed.
    """

    from torch.utils.data import DataLoader

    test_data, _ = experiment._get_data(flag="test")
    return DataLoader(
        test_data,
        batch_size=experiment.args.batch_size,
        shuffle=False,
        num_workers=experiment.args.num_workers,
        drop_last=False,
    )


def _collect_test_windows(experiment: Any, test_loader: Any) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the official ``Exp_Main.test`` tensor slice for PatchTST."""

    import torch

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    experiment.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(experiment.device)
            batch_y = batch_y.float().to(experiment.device)
            outputs = experiment.model(batch_x)
            feature_start = -1 if experiment.args.features == "MS" else 0
            predictions.append(
                outputs[:, -experiment.args.pred_len :, feature_start:].detach().cpu().numpy()
            )
            targets.append(
                batch_y[:, -experiment.args.pred_len :, feature_start:].detach().cpu().numpy()
            )
    if not predictions:
        raise ValueError("official test loader produced no prediction windows")
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def _columns_for_args(args: Namespace) -> list[str]:
    dataset_path = Path(args.root_path) / args.data_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset CSV does not exist: {dataset_path}")
    frame = pd.read_csv(dataset_path, nrows=1)
    if "date" not in frame.columns:
        raise ValueError("ETTm1 CSV requires a date column")
    if args.features in {"M", "MS"}:
        return [str(column) for column in frame.columns if column != "date"]
    return [str(args.target)]
