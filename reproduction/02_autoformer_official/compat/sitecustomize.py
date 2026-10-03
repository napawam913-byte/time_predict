"""Narrow runtime shims for APIs removed after Autoformer's published stack.

Loaded automatically through PYTHONPATH.  This file is outside the pinned
author checkout and intentionally restores only the two removed names used by
that checkout under NumPy 2 / Pandas 2.
"""

import numpy as np
import pandas as pd


if not hasattr(np, "Inf"):
    np.Inf = np.inf


_dataframe_drop = pd.DataFrame.drop


def _drop_accepting_legacy_axis(self, labels=None, *args, **kwargs):
    """Accept the historical ``drop(labels, axis)`` positional form."""

    if len(args) > 1:
        raise TypeError("drop accepts at most one legacy positional axis argument")
    if args:
        if "axis" in kwargs:
            raise TypeError("drop received axis both positionally and by keyword")
        kwargs["axis"] = args[0]
    return _dataframe_drop(self, labels=labels, **kwargs)


pd.DataFrame.drop = _drop_accepting_legacy_axis
