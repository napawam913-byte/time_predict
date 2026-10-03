#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/run_common.sh"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run-id>" >&2
  exit 2
fi

"$python_executable" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("A CUDA GPU is required for this launcher.")
print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
PY

run_author_experiment formal "$1" default
