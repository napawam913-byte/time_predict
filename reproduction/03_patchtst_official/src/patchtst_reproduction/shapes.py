"""Small, dependency-light checks for PatchTST's end-padded patch layout."""

from __future__ import annotations

import numpy as np


def padded_patch_count(sequence_length: int, *, patch_len: int, stride: int) -> int:
    """Return the official end-padded PatchTST token count."""

    if sequence_length < patch_len:
        raise ValueError("sequence length must be at least the patch length")
    if patch_len <= 0 or stride <= 0:
        raise ValueError("patch length and stride must be positive")
    return (sequence_length - patch_len) // stride + 2


def end_padded_patches(values: np.ndarray, *, patch_len: int, stride: int) -> np.ndarray:
    """Mirror ``ReplicationPad1d(...).unfold(...).permute(0,1,3,2)`` with NumPy."""

    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"values must have shape [batch, channels, length], got {values.shape}")
    expected_count = padded_patch_count(values.shape[-1], patch_len=patch_len, stride=stride)
    padded = np.concatenate((values, np.repeat(values[..., -1:], stride, axis=-1)), axis=-1)
    starts = range(0, padded.shape[-1] - patch_len + 1, stride)
    patches = np.stack([padded[..., start : start + patch_len] for start in starts], axis=-1)
    if patches.shape[-1] != expected_count:
        raise AssertionError("end-padded patch count diverges from the PatchTST contract")
    return patches


def forecast_layout(backbone_output: np.ndarray) -> np.ndarray:
    """Restore official PatchTST backbone output from [B, C, H] to [B, H, C]."""

    backbone_output = np.asarray(backbone_output)
    if backbone_output.ndim != 3:
        raise ValueError(
            "backbone output must have shape [batch, channels, horizon], "
            f"got {backbone_output.shape}"
        )
    return np.transpose(backbone_output, (0, 2, 1))
