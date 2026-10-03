#!/usr/bin/env bash

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  project_root=$(cd "$PROJECT_ROOT" && pwd -P)
  experiment_root="$project_root/reproduction/02_autoformer_official"
else
  experiment_root=$(cd "$script_directory/.." && pwd -P)
  project_root=$(cd "$experiment_root/../.." && pwd -P)
fi
upstream_dir="$experiment_root/upstream/Autoformer"
compat_dir="$experiment_root/compat"
python_executable="$experiment_root/.venv/bin/python"

validate_run_id() {
  local run_id=$1
  if [[ -z "$run_id" || "$run_id" == */* || "$run_id" == .* ]]; then
    echo "run id must be one directory name without a path: $run_id" >&2
    return 1
  fi
}

run_author_experiment() {
  local kind=$1
  local run_id=$2
  local train_epochs=$3
  local run_dir="$experiment_root/runs/$run_id"

  validate_run_id "$run_id"
  if [[ -e "$run_dir" ]]; then
    echo "run directory already exists and will not be overwritten: $run_dir" >&2
    return 1
  fi
  bash "$script_directory/verify_upstream.sh"
  if [[ ! -x "$python_executable" ]]; then
    echo "Autoformer CPU environment is missing: $python_executable" >&2
    return 1
  fi

  mkdir -p "$run_dir"
  local mpl_cache_dir="$run_dir/matplotlib-cache"
  local pip_cache_dir="$run_dir/pip-cache"
  mkdir -p "$mpl_cache_dir" "$pip_cache_dir"
  local -a args=(
    --is_training 1
    --root_path "$project_root/DataSet/ETTm1"
    --data_path ETTm1.csv
    --model_id ETTm1_96_96
    --model Autoformer
    --data ETTm1
    --features M
    --seq_len 96
    --label_len 48
    --pred_len 96
    --e_layers 2
    --d_layers 1
    --factor 3
    --enc_in 7
    --dec_in 7
    --c_out 7
    --des Exp
    --freq t
    --itr 1
  )
  if [[ "$train_epochs" != "default" ]]; then
    args+=(--train_epochs "$train_epochs")
  fi

  printf '%q ' "$python_executable" "$upstream_dir/run.py" "${args[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  {
    "$python_executable" --version
    PIP_CACHE_DIR="$pip_cache_dir" "$python_executable" -m pip freeze
  } > "$run_dir/environment.txt"
  {
    uname -a
    lscpu 2>/dev/null || true
  } > "$run_dir/system.txt"
  "$python_executable" - "$run_dir/run.json" "$kind" "$run_id" "$upstream_dir" "$train_epochs" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

path = Path(sys.argv[1])
commit = subprocess.check_output(["git", "-C", sys.argv[4], "rev-parse", "HEAD"], text=True).strip()
path.write_text(
    json.dumps(
        {
            "kind": sys.argv[2],
            "run_id": sys.argv[3],
            "upstream_commit": commit,
            "train_epochs_override": None if sys.argv[5] == "default" else int(sys.argv[5]),
            "author_seed": 2021,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

  (
    cd "$run_dir"
    MPLCONFIGDIR="$mpl_cache_dir" PYTHONPATH="$compat_dir:$upstream_dir" "$python_executable" "$upstream_dir/run.py" "${args[@]}" \
      >stdout.log 2>stderr.log
  )
}
