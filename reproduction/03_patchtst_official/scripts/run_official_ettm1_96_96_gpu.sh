#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <unique-run-id>\n' "$0" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# run_author_experiment trains the official source then invokes export_ettm1_predictions.py.
source "$script_dir/run_common.sh"
run_author_experiment "$1"
