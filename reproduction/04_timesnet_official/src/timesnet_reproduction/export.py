"""Package exactly one set of author-produced TimesNet test arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ltsf_evaluation import ForecastResult


EXPECTED_ETTM1_SHAPE = (11_425, 96, 7)


@dataclass(frozen=True)
class OfficialArtifacts:
    """The one checkpoint/result pair emitted by an official TimesNet run."""

    checkpoint: Path
    result_dir: Path
    setting: str


def discover_official_artifacts(official_run_dir: str | Path) -> OfficialArtifacts:
    """Find one setting with both author prediction and label arrays.

    This function only discovers files written by ``run.py``.  It does not
    import the model, reload a checkpoint, or issue another forward pass.
    """

    official_run_dir = Path(official_run_dir)
    checkpoints = sorted(official_run_dir.glob("checkpoints/*/checkpoint.pth"))
    if len(checkpoints) != 1:
        raise ValueError(
            "official run must contain exactly one checkpoint at "
            f"checkpoints/<setting>/checkpoint.pth; found {len(checkpoints)}"
        )
    result_dirs = sorted(
        path
        for path in (official_run_dir / "results").glob("*")
        if path.is_dir() and (path / "pred.npy").is_file() and (path / "true.npy").is_file()
    )
    if len(result_dirs) != 1:
        raise ValueError(
            "official run must contain exactly one result directory with pred.npy and true.npy; "
            f"found {len(result_dirs)}"
        )
    checkpoint = checkpoints[0]
    result_dir = result_dirs[0]
    setting = checkpoint.parent.name
    if result_dir.name != setting:
        raise ValueError(
            "official checkpoint setting does not match the result directory: "
            f"{setting!r} != {result_dir.name!r}"
        )
    return OfficialArtifacts(checkpoint=checkpoint, result_dir=result_dir, setting=setting)


def write_normalized_archive(
    archive_path: str | Path,
    prediction: np.ndarray,
    target: np.ndarray,
    columns: list[str],
) -> Path:
    """Write arrays in the project-wide explicit normalized-scale contract."""

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
    official_run_dir: str | Path,
    dataset_csv: str | Path,
    archive_path: str | Path,
) -> Path:
    """Package untouched official ETTm1 arrays after strict completeness checks."""

    artifacts = discover_official_artifacts(official_run_dir)
    prediction = np.load(artifacts.result_dir / "pred.npy", allow_pickle=False)
    target = np.load(artifacts.result_dir / "true.npy", allow_pickle=False)
    _require_complete_ettm1_shape(prediction, "pred.npy")
    _require_complete_ettm1_shape(target, "true.npy")
    return write_normalized_archive(archive_path, prediction, target, _ettm1_columns(dataset_csv))


def _require_complete_ettm1_shape(values: np.ndarray, name: str) -> None:
    if values.shape != EXPECTED_ETTM1_SHAPE:
        raise ValueError(
            f"{name} has shape {values.shape}; expected ETTm1 output shape "
            f"{EXPECTED_ETTM1_SHAPE}"
        )


def _ettm1_columns(dataset_csv: str | Path) -> list[str]:
    dataset_csv = Path(dataset_csv)
    if not dataset_csv.is_file():
        raise FileNotFoundError(f"ETTm1 CSV does not exist: {dataset_csv}")
    frame = pd.read_csv(dataset_csv, nrows=1)
    if "date" not in frame.columns:
        raise ValueError("ETTm1 CSV requires a date column")
    columns = [str(column) for column in frame.columns if column != "date"]
    if len(columns) != EXPECTED_ETTM1_SHAPE[-1]:
        raise ValueError(
            f"ETTm1 CSV must contain {EXPECTED_ETTM1_SHAPE[-1]} value columns, found {len(columns)}"
        )
    return columns


__all__ = [
    "EXPECTED_ETTM1_SHAPE",
    "OfficialArtifacts",
    "discover_official_artifacts",
    "export_from_official_run",
    "write_normalized_archive",
]
