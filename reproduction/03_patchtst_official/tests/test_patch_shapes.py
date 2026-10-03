"""Tensor-shape checks for the PatchTST patching mental model."""

from __future__ import annotations

import numpy as np

from patchtst_reproduction.shapes import end_padded_patches, forecast_layout, padded_patch_count


def test_end_padding_turns_b_c_96_into_b_c_16_12_without_averaging_values() -> None:
    """The final patch preserves its raw tail and appends repeated final observations."""

    values = np.arange(2 * 7 * 96, dtype=np.float64).reshape(2, 7, 96)
    patches = end_padded_patches(values, patch_len=16, stride=8)

    assert padded_patch_count(96, patch_len=16, stride=8) == 12
    assert patches.shape == (2, 7, 16, 12)
    np.testing.assert_array_equal(patches[0, 0, :, 0], np.arange(16, dtype=np.float64))
    np.testing.assert_array_equal(
        patches[0, 0, :, -1], np.array([*range(88, 96), *([95] * 8)], dtype=np.float64)
    )


def test_forecast_head_layout_returns_batch_horizon_channel_order() -> None:
    """The official backbone's B×C×H head output is restored to B×H×C."""

    backbone_output = np.zeros((3, 7, 96), dtype=np.float64)
    forecast = forecast_layout(backbone_output)

    assert forecast.shape == (3, 96, 7)
