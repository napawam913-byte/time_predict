#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
experiment_root="$(cd "$script_dir/.." && pwd -P)"
project_root="$(cd "$experiment_root/../.." && pwd -P)"
upstream_dir="$experiment_root/upstream/Time-Series-Library"
python_executable="${TIMESNET_PYTHON:-$experiment_root/.venv/bin/python}"

if [[ ! -x "$python_executable" ]]; then
  printf 'TimesNet environment is missing: %s\n' "$python_executable" >&2
  exit 1
fi

bash "$script_dir/verify_upstream.sh"
# Python automatically loads compat/sitecustomize.py from this PYTHONPATH.
TIMESNET_UPSTREAM_DIR="$upstream_dir" \
PYTHONPATH="$experiment_root/compat:$upstream_dir:$experiment_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_executable" - <<'PY'
from argparse import Namespace
import os
from pathlib import Path

import torch

from models.TimesNet import Model
from timesnet_reproduction.observation import observe_periods

configs = Namespace(
    task_name="long_term_forecast",
    seq_len=96,
    label_len=48,
    pred_len=96,
    e_layers=2,
    d_model=64,
    d_ff=64,
    top_k=5,
    num_kernels=6,
    enc_in=7,
    c_out=7,
    embed="timeF",
    freq="h",
    dropout=0.1,
)
model = Model(configs).eval()
batch_size = 2
with torch.no_grad():
    forecast = model(
        torch.randn(batch_size, 96, 7),
        torch.randn(batch_size, 96, 4),
        torch.zeros(batch_size, 144, 7),
        torch.randn(batch_size, 144, 4),
    )
observation = observe_periods(
    Path(os.environ["TIMESNET_UPSTREAM_DIR"]), torch.randn(batch_size, 192, 64), top_k=5
)
print(f"forecast_shape={tuple(forecast.shape)}")
print(f"selected_periods={observation.periods}")
print(f"period_grids={observation.grids}")
PY
