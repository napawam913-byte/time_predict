#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/run_common.sh"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run-id>" >&2
  exit 2
fi
run_author_experiment smoke "$1" 1
