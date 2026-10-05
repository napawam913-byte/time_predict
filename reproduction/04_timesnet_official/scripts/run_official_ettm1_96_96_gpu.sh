#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <unique-run-id>\n' "${0##*/}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
experiment_root="$(cd "$script_dir/.." && pwd -P)"
python_executable="$experiment_root/.venv/bin/python"

if [[ ! -x "$python_executable" ]]; then
  printf 'TimesNet GPU environment is missing: %s\n' "$python_executable" >&2
  exit 1
fi

"$python_executable" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the TimesNet GPU launcher")
print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
PY

# run_author_experiment invokes unchanged author run.py and then packages its arrays.
source "$script_dir/run_common.sh"
run_author_experiment "$1"
