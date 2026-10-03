#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
EXPERIMENT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
VENV_PATH="$EXPERIMENT_ROOT/.venv"
MPL_CACHE_DIR="$EXPERIMENT_ROOT/.mplconfig"
PIP_CACHE_DIR="$EXPERIMENT_ROOT/.pip-cache"

mkdir -p "$MPL_CACHE_DIR" "$PIP_CACHE_DIR"
export MPLCONFIGDIR="$MPL_CACHE_DIR"
export PIP_CACHE_DIR

if [[ ! -d "$VENV_PATH" ]]; then
  python3 -m venv "$VENV_PATH"
fi

PYTHON="$VENV_PATH/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --requirement "$EXPERIMENT_ROOT/requirements.compat.txt"
"$PYTHON" - <<'PY'
import matplotlib
import numpy
import pandas
import sklearn
import torch
from reformer_pytorch import LSHSelfAttention

print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}, pandas={pandas.__version__}")
print(f"scikit-learn={sklearn.__version__}, matplotlib={matplotlib.__version__}")
print(f"reformer_pytorch={LSHSelfAttention.__module__}")
PY

mkdir -p "$EXPERIMENT_ROOT/provenance"
{
  "$PYTHON" --version
  "$PYTHON" -m pip freeze
} > "$EXPERIMENT_ROOT/provenance/environment.freeze.txt"
