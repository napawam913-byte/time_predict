#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
experiment_root="$(cd "$script_dir/.." && pwd -P)"
project_root="$(cd "$experiment_root/../.." && pwd -P)"
upstream_dir="$experiment_root/upstream/Time-Series-Library"
compat_dir="$experiment_root/compat"
python_executable="$experiment_root/.venv/bin/python"
readonly author_commit="2665a3143dae12d1cbcc31ddd396bbff48773bce"

validate_run_id() {
  local run_id=$1
  if [[ -z "$run_id" || "$run_id" == */* || "$run_id" == .* || "$run_id" == ".." ]]; then
    printf 'run id must be one visible directory name without a path: %s\n' "$run_id" >&2
    return 1
  fi
}

write_run_config() {
  local config_path=$1
  local run_id=$2
  local official_run_dir=$3
  "$python_executable" - "$config_path" "$run_id" "$project_root" "$official_run_dir" "$author_commit" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
run_id = sys.argv[2]
project_root = Path(sys.argv[3])
official_run_dir = Path(sys.argv[4])
author_commit = sys.argv[5]
args = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "root_path": str(project_root / "DataSet" / "ETTm1"),
    "data_path": "ETTm1.csv",
    "model_id": "ETTm1_96_96",
    "model": "TimesNet",
    "data": "ETTm1",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "e_layers": 2,
    "d_layers": 1,
    "factor": 3,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 64,
    "d_ff": 64,
    "top_k": 5,
    "des": "Exp",
    "itr": 1,
    "batch_size": 32,
    "train_epochs": 10,
    "patience": 3,
    "learning_rate": 0.0001,
    "lradj": "type1",
    "num_workers": 10,
    "checkpoints": str(official_run_dir / "checkpoints"),
    "gpu": 0,
}
path.write_text(
    json.dumps(
        {
            "run_id": run_id,
            "author_repository": "https://github.com/thuml/Time-Series-Library.git",
            "author_commit": author_commit,
            "author_seed": 2021,
            "args": args,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_author_experiment() {
  local run_id=$1
  local run_dir="$experiment_root/runs/$run_id"
  local official_run_dir="$run_dir/official"
  local config_path="$run_dir/config.json"
  local dataset_csv="$project_root/DataSet/ETTm1/ETTm1.csv"

  validate_run_id "$run_id"
  if [[ -e "$run_dir" ]]; then
    printf 'run directory already exists and will not be overwritten: %s\n' "$run_dir" >&2
    return 1
  fi
  if [[ ! -x "$python_executable" ]]; then
    printf 'TimesNet environment is missing: %s\n' "$python_executable" >&2
    return 1
  fi
  if [[ ! -f "$dataset_csv" ]]; then
    printf 'ETTm1 CSV is missing: %s\n' "$dataset_csv" >&2
    return 1
  fi

  # Check upstream identity and cleanliness before creating any result directory.
  bash "$script_dir/verify_upstream.sh"
  mkdir -p "$official_run_dir"
  write_run_config "$config_path" "$run_id" "$official_run_dir"

  local -a args=(
    --task_name long_term_forecast
    --is_training 1
    --root_path "$project_root/DataSet/ETTm1"
    --data_path ETTm1.csv
    --model_id ETTm1_96_96
    --model TimesNet
    --data ETTm1
    --features M
    --target OT
    --freq h
    --seq_len 96
    --label_len 48
    --pred_len 96
    --e_layers 2
    --d_layers 1
    --factor 3
    --enc_in 7
    --dec_in 7
    --c_out 7
    --d_model 64
    --d_ff 64
    --top_k 5
    --des Exp
    --itr 1
    --batch_size 32
    --train_epochs 10
    --patience 3
    --learning_rate 0.0001
    --lradj type1
    --num_workers 10
    --checkpoints "$official_run_dir/checkpoints"
    --gpu 0
  )

  printf '%q ' "$python_executable" "$upstream_dir/run.py" "${args[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "$python_executable" --version > "$run_dir/python-version.txt"
  "$python_executable" -m pip freeze > "$run_dir/environment.txt"

  (
    cd "$official_run_dir"
    PYTHONPATH="$compat_dir:$upstream_dir${PYTHONPATH:+:$PYTHONPATH}" \
      "$python_executable" "$upstream_dir/run.py" "${args[@]}" \
      > stdout.log 2> stderr.log
  )

  PYTHONPATH="$project_root/reproduction/common:$experiment_root/src" \
    "$python_executable" "$script_dir/export_ettm1_predictions.py" \
    --official-run-dir "$official_run_dir" \
    --csv "$dataset_csv" \
    --output "$run_dir/predictions.npz"
}
