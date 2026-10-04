#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
experiment_root="$(cd "$script_dir/.." && pwd -P)"
project_root="$(cd "$experiment_root/../.." && pwd -P)"
upstream_dir="$experiment_root/upstream/PatchTST/PatchTST_supervised"
python_executable="$experiment_root/.venv/bin/python"

validate_run_id() {
  local run_id=$1
  if [[ -z "$run_id" || "$run_id" == */* || "$run_id" == .* ]]; then
    printf 'run id must be one directory name without a path: %s\n' "$run_id" >&2
    return 1
  fi
}

write_run_config() {
  local config_path=$1
  local official_run_dir=$2
  "$python_executable" - "$config_path" "$project_root" "$official_run_dir" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
project_root = Path(sys.argv[2])
official_run_dir = Path(sys.argv[3])
args = {
    "random_seed": 2021,
    "is_training": 1,
    "model_id": "ETTm1_96_96",
    "model": "PatchTST",
    "data": "ETTm1",
    "root_path": str(project_root / "DataSet" / "ETTm1"),
    "data_path": "ETTm1.csv",
    "features": "M",
    "target": "OT",
    "freq": "t",
    "checkpoints": str(official_run_dir / "checkpoints"),
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": "end",
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "individual": 0,
    "embed_type": 0,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_layers": 1,
    "d_ff": 256,
    "moving_avg": 25,
    "factor": 1,
    "distil": True,
    "dropout": 0.2,
    "embed": "timeF",
    "activation": "gelu",
    "output_attention": False,
    "do_predict": False,
    "num_workers": 10,
    "itr": 1,
    "train_epochs": 100,
    "batch_size": 128,
    "patience": 20,
    "learning_rate": 0.0001,
    "des": "Exp",
    "loss": "mse",
    "lradj": "TST",
    "pct_start": 0.4,
    "use_amp": False,
    "use_gpu": True,
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0",
    "test_flop": False,
}
path.write_text(json.dumps({"args": args}, indent=2) + "\n", encoding="utf-8")
PY
}

run_author_experiment() {
  local run_id=$1
  local run_dir="$experiment_root/runs/$run_id"
  local official_run_dir="$run_dir/official"
  local config_path="$run_dir/config.json"

  validate_run_id "$run_id"
  if [[ -e "$run_dir" ]]; then
    printf 'run directory already exists and will not be overwritten: %s\n' "$run_dir" >&2
    return 1
  fi
  if [[ ! -x "$python_executable" ]]; then
    printf 'PatchTST environment is missing: %s\n' "$python_executable" >&2
    return 1
  fi
  bash "$script_dir/verify_upstream.sh"
  mkdir -p "$official_run_dir"
  write_run_config "$config_path" "$official_run_dir"

  local -a args=(
    --random_seed 2021
    --is_training 1
    --root_path "$project_root/DataSet/ETTm1"
    --data_path ETTm1.csv
    --model_id ETTm1_96_96
    --model PatchTST
    --data ETTm1
    --features M
    --freq t
    --seq_len 96
    --label_len 48
    --pred_len 96
    --enc_in 7
    --dec_in 7
    --c_out 7
    --e_layers 3
    --n_heads 16
    --d_model 128
    --d_ff 256
    --dropout 0.2
    --fc_dropout 0.2
    --head_dropout 0
    --patch_len 16
    --stride 8
    --padding_patch end
    --revin 1
    --affine 0
    --subtract_last 0
    --decomposition 0
    --des Exp
    --train_epochs 100
    --patience 20
    --lradj TST
    --pct_start 0.4
    --itr 1
    --batch_size 128
    --learning_rate 0.0001
    --checkpoints "$official_run_dir/checkpoints"
  )

  printf '%q ' "$python_executable" "$upstream_dir/run_longExp.py" "${args[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "$python_executable" --version > "$run_dir/python-version.txt"
  "$python_executable" -m pip freeze > "$run_dir/environment.txt"

  (
    cd "$official_run_dir"
    PYTHONPATH="$upstream_dir" "$python_executable" "$upstream_dir/run_longExp.py" "${args[@]}" \
      > stdout.log 2> stderr.log
  )
  PYTHONPATH="$project_root/reproduction/common:$experiment_root/src" \
    "$python_executable" "$script_dir/export_ettm1_predictions.py" \
      --upstream-supervised-dir "$upstream_dir" \
      --official-run-dir "$official_run_dir" \
      --config "$config_path" \
      --output "$run_dir/predictions.npz"
}
