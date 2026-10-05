"""Contracts for exporting aligned PatchTST predictions without upstream edits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from patchtst_reproduction import export
from patchtst_reproduction.export import discover_official_artifacts, write_normalized_archive


def _make_official_artifacts(root: Path, setting: str = "one") -> None:
    checkpoint = root / "checkpoints" / setting / "checkpoint.pth"
    result = root / "results" / setting / "pred.npy"
    checkpoint.parent.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    checkpoint.touch()
    np.save(result, np.zeros((1, 2, 2)))


def test_export_writes_a_declared_normalized_archive_with_prediction_target_and_columns(
    tmp_path: Path,
) -> None:
    """Comparison code receives every tensor needed to reject label misalignment."""

    archive_path = write_normalized_archive(
        tmp_path / "predictions.npz",
        prediction=np.zeros((2, 3, 2)),
        target=np.ones((2, 3, 2)),
        columns=["HUFL", "OT"],
    )

    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"prediction", "target", "columns", "scale"}
        assert archive["prediction"].shape == (2, 3, 2)
        assert archive["target"].shape == (2, 3, 2)
        assert archive["columns"].tolist() == ["HUFL", "OT"]
        assert archive["scale"].item() == "normalized"


@pytest.mark.parametrize(
    ("prediction", "target", "columns", "message"),
    [
        (np.zeros((2, 3)), np.zeros((2, 3)), ["HUFL", "OT"], "shape"),
        (np.zeros((2, 3, 2)), np.zeros((2, 2, 2)), ["HUFL", "OT"], "match"),
        (np.zeros((2, 3, 2)), np.zeros((2, 3, 2)), ["HUFL"], "column"),
    ],
)
def test_export_rejects_arrays_outside_the_window_horizon_channel_contract(
    tmp_path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    columns: list[str],
    message: str,
) -> None:
    """An archive cannot be written with a hidden shape or channel mismatch."""

    with pytest.raises(ValueError, match=message):
        write_normalized_archive(tmp_path / "predictions.npz", prediction, target, columns)


def test_discovery_rejects_absent_or_ambiguous_checkpoint_and_result_directories(tmp_path: Path) -> None:
    """The exporter must not select an arbitrary author output from an old run."""

    with pytest.raises(ValueError, match="checkpoint"):
        discover_official_artifacts(tmp_path)

    _make_official_artifacts(tmp_path, "first")
    artifacts = discover_official_artifacts(tmp_path)
    assert artifacts.setting == "first"
    assert artifacts.checkpoint.name == "checkpoint.pth"

    _make_official_artifacts(tmp_path, "second")
    with pytest.raises(ValueError, match="checkpoint"):
        discover_official_artifacts(tmp_path)

    ambiguous_result_root = tmp_path / "only-result"
    _make_official_artifacts(ambiguous_result_root, "one")
    second_result = ambiguous_result_root / "results" / "two" / "pred.npy"
    second_result.parent.mkdir(parents=True)
    np.save(second_result, np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="result"):
        discover_official_artifacts(ambiguous_result_root)


def test_complete_test_loader_keeps_the_last_incomplete_official_batch() -> None:
    """Fair evaluation must retain all windows, unlike upstream's test loader."""

    dataset = TensorDataset(torch.arange(5))

    class Experiment:
        args = type("Args", (), {"batch_size": 4, "num_workers": 0})()

        @staticmethod
        def _get_data(flag: str):
            assert flag == "test"
            return dataset, DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)

    loader = export.complete_test_loader(Experiment())

    assert loader.drop_last is False
    assert torch.cat([batch[0] for batch in loader]).tolist() == [0, 1, 2, 3, 4]
