"""Contracts for read-only inspection of author-selected TimesNet periods."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from timesnet_reproduction.observation import observe_periods


def _write_fake_author_module(root: Path) -> None:
    models = root / "models"
    models.mkdir(parents=True)
    (models / "TimesNet.py").write_text(
        "import torch\n"
        "def FFT_for_Period(x, k=2):\n"
        "    assert x.ndim == 3\n"
        "    return [24, 16][:k], torch.ones((x.shape[0], k))\n",
        encoding="utf-8",
    )


def test_observation_calls_author_fft_and_reports_padded_period_grids(tmp_path: Path) -> None:
    """The observer reports the model's period choice without owning an FFT implementation."""

    _write_fake_author_module(tmp_path)

    observation = observe_periods(tmp_path, torch.randn(2, 192, 64), top_k=2)

    assert observation.periods == (24, 16)
    assert observation.weight_shape == (2, 2)
    assert observation.total_length == 192
    assert observation.grids == ((8, 24), (12, 16))


@pytest.mark.parametrize("top_k", [0, -1])
def test_observation_rejects_non_positive_top_k(tmp_path: Path, top_k: int) -> None:
    """Invalid observation settings must not enter an author forward path."""

    _write_fake_author_module(tmp_path)

    with pytest.raises(ValueError, match="top_k"):
        observe_periods(tmp_path, torch.randn(2, 192, 64), top_k=top_k)


def test_observation_rejects_non_finite_or_non_sequence_values(tmp_path: Path) -> None:
    """The diagnostic is only defined on finite B×T×C TimesBlock inputs."""

    _write_fake_author_module(tmp_path)

    with pytest.raises(ValueError, match="three dimensions"):
        observe_periods(tmp_path, torch.randn(2, 192), top_k=1)
    with pytest.raises(ValueError, match="finite"):
        observe_periods(tmp_path, torch.full((2, 192, 64), float("nan")), top_k=1)
