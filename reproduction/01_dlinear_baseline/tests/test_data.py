import numpy as np
import pandas as pd
import pytest

from ltsf_baseline.data import SplitBoundaries, prepare_ettm1


def test_scaler_uses_only_training_rows(synthetic_ettm1_csv):
    boundaries = SplitBoundaries(train_end=6, validation_end=9, test_end=12)
    prepared = prepare_ettm1(synthetic_ettm1_csv, 2, 2, boundaries)

    np.testing.assert_allclose(prepared.scaler.mean_, [2.5, 25.0])
    np.testing.assert_allclose(
        prepared.scaler.scale_, [np.sqrt(35 / 12), np.sqrt(3500 / 12)]
    )


def test_validation_and_test_labels_stay_inside_their_target_ranges(synthetic_ettm1_csv):
    boundaries = SplitBoundaries(train_end=6, validation_end=9, test_end=12)
    prepared = prepare_ettm1(synthetic_ettm1_csv, 2, 2, boundaries)

    assert prepared.validation.target_starts.tolist() == [6, 7]
    assert prepared.test.target_starts.tolist() == [9, 10]
    history, target = prepared.test[0]
    assert history.shape == (2, 2)
    assert target.shape == (2, 2)


def test_rejects_non_monotonic_timestamps(tmp_path):
    path = tmp_path / "bad.csv"
    frame = pd.DataFrame(
        {
            "date": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:30:00",
                "2024-01-01 00:15:00",
                "2024-01-01 00:45:00",
            ],
            "a": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_ettm1(path, 1, 1, SplitBoundaries(2, 3, 4))


def test_rejects_csv_without_date_column(tmp_path):
    path = tmp_path / "missing_date.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="date"):
        prepare_ettm1(path, 1, 1, SplitBoundaries(2, 3, 4))


def test_constant_training_channel_scales_without_nan(synthetic_ettm1_csv):
    frame = pd.read_csv(synthetic_ettm1_csv)
    frame["b"] = 5.0
    frame.to_csv(synthetic_ettm1_csv, index=False)

    prepared = prepare_ettm1(
        synthetic_ettm1_csv, 2, 2, SplitBoundaries(6, 9, 12)
    )

    assert np.isfinite(prepared.train.values).all()
