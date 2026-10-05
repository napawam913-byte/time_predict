"""Read-only observation helpers for the author TimesNet period selection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from typing import Callable

import torch


@dataclass(frozen=True)
class PeriodObservation:
    """Selected author periods and the 2D grids their TimesBlock will form."""

    periods: tuple[int, ...]
    weight_shape: tuple[int, ...]
    total_length: int
    grids: tuple[tuple[int, int], ...]


def observe_periods(
    upstream_root: str | Path,
    values: torch.Tensor,
    top_k: int,
) -> PeriodObservation:
    """Invoke the immutable author's FFT helper and report its reshape geometry.

    ``values`` represents one TimesBlock input and therefore has shape
    ``[batch, seq_len + pred_len, d_model]``. It is not fed back into training
    or inference; this function exists solely for a learning-oriented smoke
    observation.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 3:
        raise ValueError("values must have three dimensions [batch, time, channels]")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if top_k > values.shape[1] // 2:
        raise ValueError("top_k exceeds the available positive FFT bins")

    fft_for_period = _load_author_fft(Path(upstream_root))
    raw_periods, weights = fft_for_period(values, top_k)
    periods = tuple(int(period) for period in raw_periods)
    if len(periods) != top_k or any(period <= 0 for period in periods):
        raise ValueError("author FFT_for_Period returned invalid periods")

    total_length = int(values.shape[1])
    grids = tuple(((total_length + period - 1) // period, period) for period in periods)
    return PeriodObservation(
        periods=periods,
        weight_shape=tuple(int(size) for size in weights.shape),
        total_length=total_length,
        grids=grids,
    )


def _load_author_fft(upstream_root: Path) -> Callable[[torch.Tensor, int], tuple[object, torch.Tensor]]:
    model_path = upstream_root / "models" / "TimesNet.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"author TimesNet source does not exist: {model_path}")

    module_name = "_timesnet_author_" + sha256(str(model_path.resolve()).encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load author TimesNet source: {model_path}")

    source_path = str(upstream_root.resolve())
    inserted = source_path not in sys.path
    if inserted:
        sys.path.insert(0, source_path)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(source_path)

    fft_for_period = getattr(module, "FFT_for_Period", None)
    if not callable(fft_for_period):
        raise AttributeError("author TimesNet source does not define FFT_for_Period")
    return fft_for_period
