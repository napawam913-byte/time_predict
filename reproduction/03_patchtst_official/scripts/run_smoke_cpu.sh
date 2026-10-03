#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
experiment_root="$(cd "$script_dir/.." && pwd -P)"
upstream_dir="$experiment_root/upstream/PatchTST/PatchTST_supervised"
python_executable="${PATCHTST_PYTHON:-$experiment_root/.venv/bin/python}"

if [[ ! -x "$python_executable" ]]; then
  printf 'PatchTST environment is missing: %s\n' "$python_executable" >&2
  exit 1
fi
bash "$script_dir/verify_upstream.sh"

PYTHONPATH="$upstream_dir" "$python_executable" - "$upstream_dir" <<'PY'
from argparse import Namespace
from pathlib import Path
import sys

import torch

source = str(Path(sys.argv[1]).resolve())
if source not in sys.path:
    sys.path.insert(0, source)
from models.PatchTST import Model

args = Namespace(
    enc_in=7, seq_len=96, pred_len=96, e_layers=3, n_heads=16, d_model=128,
    d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0.0, individual=0,
    patch_len=16, stride=8, padding_patch="end", revin=1, affine=0,
    subtract_last=0, decomposition=0, kernel_size=25,
)
model = Model(args).eval()
inputs = torch.randn(2, 96, 7)
with torch.no_grad():
    outputs = model(inputs)
assert tuple(outputs.shape) == (2, 96, 7), tuple(outputs.shape)
print("input_shape=(2, 96, 7)")
print("patch_layout=(2, 7, 16, 12)")
print("forecast_shape=(2, 96, 7)")
PY
