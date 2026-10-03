#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
reproduction_root="$(cd "$script_dir/.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="$reproduction_root/.venv"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r "$reproduction_root/requirements.common.txt"
"$venv_dir/bin/python" -m pip install "torch==2.5.1" --index-url "$torch_index_url"

printf 'Created PatchTST CPU environment: %s\n' "$venv_dir"
