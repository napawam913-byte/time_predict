"""Compatibility shim loaded outside the immutable 2023 TSLib source tree.

The author data loader uses ``DataFrame.drop(labels, axis)`` with ``axis`` as
a positional argument. pandas 2 keeps the operation but makes ``axis``
keyword-only. This wrapper restores that call form without modifying upstream.
"""

from __future__ import annotations

import pandas as pd


_author_drop = pd.DataFrame.drop


def _drop_with_legacy_axis(self: pd.DataFrame, labels=None, axis=0, *args, **kwargs):
    return _author_drop(self, labels=labels, axis=axis, *args, **kwargs)


pd.DataFrame.drop = _drop_with_legacy_axis
