"""Contracts for packaging untouched TimesNet author result arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timesnet_reproduction.export import (
    EXPECTED_ETTM1_SHAPE,
    discover_official_artifacts,
    export_from_official_run,
    write_normalized_archive,
)


def _write_ettm1_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "date": ["2016-07-01 00:00:00"],
            "HUFL": [1.0],
            "HULL": [2.0],
            "MUFL": [3.0],
            "MULL": [4.0],
            "LUFL": [5.0],
            "LULL": [6.0],
            "OT": [7.0],
        }
    )
    frame.to_csv(path, index=False)
    return path


def _write_author_artifacts(
    root: Path,
    *,
    checkpoint_setting: str = "setting",
    result_setting: str | None = None,
    prediction: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> None:
    result_setting = result_setting or checkpoint_setting
    checkpoint = root / "checkpoints" / checkpoint_setting / "checkpoint.pth"
    result_dir = root / "results" / result_setting
    checkpoint.parent.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    checkpoint.touch()
    if prediction is not None:
        np.save(result_dir / "pred.npy", prediction)
    if target is not None:
        np.save(result_dir / "true.npy", target)


def test_export_preserves_complete_author_arrays_and_declares_normalized_scale(tmp_path: Path) -> None:
    """The export reads the official arrays; it does not execute model inference."""

    prediction = np.zeros(EXPECTED_ETTM1_SHAPE, dtype=np.float32)
    target = np.ones(EXPECTED_ETTM1_SHAPE, dtype=np.float32)
    _write_author_artifacts(tmp_path, prediction=prediction, target=target)

    archive_path = export_from_official_run(
        official_run_dir=tmp_path,
        dataset_csv=_write_ettm1_csv(tmp_path / "ETTm1.csv"),
        archive_path=tmp_path / "predictions.npz",
    )

    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"prediction", "target", "columns", "scale"}
        assert archive["prediction"].shape == EXPECTED_ETTM1_SHAPE
        assert archive["target"].shape == EXPECTED_ETTM1_SHAPE
        assert archive["columns"].tolist() == [
            "HUFL",
            "HULL",
            "MUFL",
            "MULL",
            "LUFL",
            "LULL",
            "OT",
        ]
        assert archive["scale"].item() == "normalized"
        assert float(archive["target"].mean()) == 1.0


def test_discovery_rejects_missing_ambiguous_or_setting_mismatched_author_artifacts(
    tmp_path: Path,
) -> None:
    """A stale or partial run cannot be silently selected for comparison."""

    with pytest.raises(ValueError, match="exactly one checkpoint"):
        discover_official_artifacts(tmp_path)

    _write_author_artifacts(
        tmp_path,
        prediction=np.zeros((1, 1, 1)),
        target=np.zeros((1, 1, 1)),
    )
    artifacts = discover_official_artifacts(tmp_path)
    assert artifacts.setting == "setting"

    _write_author_artifacts(
        tmp_path,
        checkpoint_setting="second",
        prediction=np.zeros((1, 1, 1)),
        target=np.zeros((1, 1, 1)),
    )
    with pytest.raises(ValueError, match="exactly one checkpoint"):
        discover_official_artifacts(tmp_path)

    mismatch = tmp_path / "mismatch"
    _write_author_artifacts(
        mismatch,
        checkpoint_setting="checkpoint-setting",
        result_setting="result-setting",
        prediction=np.zeros((1, 1, 1)),
        target=np.zeros((1, 1, 1)),
    )
    with pytest.raises(ValueError, match="does not match"):
        discover_official_artifacts(mismatch)


def test_export_rejects_incomplete_or_wrong_shape_author_outputs(tmp_path: Path) -> None:
    """All 11,425 official test windows are required before fair evaluation."""

    csv_path = _write_ettm1_csv(tmp_path / "ETTm1.csv")
    _write_author_artifacts(
        tmp_path / "incomplete",
        prediction=np.zeros((11_424, 96, 7)),
        target=np.zeros((11_424, 96, 7)),
    )
    with pytest.raises(ValueError, match="expected ETTm1 output shape"):
        export_from_official_run(
            official_run_dir=tmp_path / "incomplete",
            dataset_csv=csv_path,
            archive_path=tmp_path / "incomplete.npz",
        )

    _write_author_artifacts(
        tmp_path / "missing-target",
        prediction=np.zeros(EXPECTED_ETTM1_SHAPE, dtype=np.float32),
        target=None,
    )
    with pytest.raises(ValueError, match="exactly one result"):
        export_from_official_run(
            official_run_dir=tmp_path / "missing-target",
            dataset_csv=csv_path,
            archive_path=tmp_path / "missing-target.npz",
        )


def test_archive_writer_rejects_nonfinite_or_invalid_column_contract(tmp_path: Path) -> None:
    """The shared evaluator receives finite rank-three arrays and declared columns."""

    with pytest.raises(ValueError, match="finite"):
        write_normalized_archive(
            tmp_path / "bad.npz",
            np.array([[[float("nan")]]]),
            np.zeros((1, 1, 1)),
            ["HUFL"],
        )
    with pytest.raises(ValueError, match="column"):
        write_normalized_archive(
            tmp_path / "bad-columns.npz",
            np.zeros((1, 1, 2)),
            np.zeros((1, 1, 2)),
            ["HUFL"],
        )
