"""Compatibility layer checks for unmodified 2021 Autoformer source."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = EXPERIMENT_ROOT / ".venv/bin/python"
COMPAT_PATH = EXPERIMENT_ROOT / "compat"


def test_compat_layer_restores_only_the_removed_apis_used_by_upstream() -> None:
    """The old Pandas drop axis and NumPy Inf names work without editing upstream."""

    environment = os.environ | {"PYTHONPATH": str(COMPAT_PATH)}
    result = subprocess.run(
        [
            str(PYTHON),
            "-c",
            "import numpy as np; import pandas as pd; "
            "assert np.Inf == np.inf; "
            "assert list(pd.DataFrame({'date':[1], 'OT':[2]}).drop(['date'], 1)) == ['OT']",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
