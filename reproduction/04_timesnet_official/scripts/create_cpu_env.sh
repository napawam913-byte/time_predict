#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
experiment_root="$(cd "$script_dir/.." && pwd -P)"
python_bootstrap="${TIMESNET_PYTHON:-python3}"
python_executable="$experiment_root/.venv/bin/python"

if ! command -v "$python_bootstrap" >/dev/null 2>&1; then
  printf 'TimesNet Python interpreter is unavailable: %s\n' "$python_bootstrap" >&2
  exit 1
fi

if [[ "$("$python_bootstrap" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  printf 'TimesNet wrapper requires Python 3.12; set TIMESNET_PYTHON to a Python 3.12 interpreter.\n' >&2
  exit 1
fi

if [[ ! -x "$python_executable" ]]; then
  "$python_bootstrap" -m venv "$experiment_root/.venv"
fi

"$python_executable" -m pip install --upgrade pip
"$python_executable" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
"$python_executable" -m pip install -r "$experiment_root/requirements.common.txt"

printf 'Created TimesNet CPU environment: %s\n' "$experiment_root/.venv"
