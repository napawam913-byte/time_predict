#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <pytorch-wheel-index-url>" >&2
  echo "example: $0 https://download.pytorch.org/whl/cu121" >&2
  exit 2
fi

TORCH_INDEX_URL=$1
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
"$PYTHON" -m pip install --index-url "$TORCH_INDEX_URL" torch==2.5.1
"$PYTHON" -m pip install --requirement "$EXPERIMENT_ROOT/requirements.common.txt"
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU; select a compatible wheel and check nvidia-smi.")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, device={torch.cuda.get_device_name(0)}")
PY

mkdir -p "$EXPERIMENT_ROOT/provenance"
{
  "$PYTHON" --version
  "$PYTHON" -m pip freeze
} > "$EXPERIMENT_ROOT/provenance/environment.freeze.txt"
