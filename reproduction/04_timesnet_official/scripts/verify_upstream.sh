#!/usr/bin/env bash
set -euo pipefail

readonly OFFICIAL_URL="https://github.com/thuml/Time-Series-Library.git"
readonly OFFICIAL_COMMIT="2665a3143dae12d1cbcc31ddd396bbff48773bce"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../../.." && pwd -P)"
upstream_dir="$project_root/reproduction/04_timesnet_official/upstream/Time-Series-Library"

if [[ ! -d "$upstream_dir/.git" ]]; then
  printf 'Official Time-Series-Library source is absent: run fetch_upstream.sh first.\n' >&2
  exit 1
fi

if [[ "$(git -C "$upstream_dir" remote get-url origin)" != "$OFFICIAL_URL" ]]; then
  printf 'Unexpected upstream origin URL.\n' >&2
  exit 1
fi

if [[ "$(git -C "$upstream_dir" rev-parse HEAD)" != "$OFFICIAL_COMMIT" ]]; then
  printf 'Unexpected upstream commit.\n' >&2
  exit 1
fi

if ! git -C "$upstream_dir" diff --quiet; then
  printf 'Official Time-Series-Library working tree has unstaged modifications.\n' >&2
  exit 1
fi

if ! git -C "$upstream_dir" diff --cached --quiet; then
  printf 'Official Time-Series-Library working tree has staged modifications.\n' >&2
  exit 1
fi

if [[ -n "$(git -C "$upstream_dir" status --porcelain)" ]]; then
  printf 'Official Time-Series-Library working tree has untracked or otherwise dirty files.\n' >&2
  exit 1
fi

required_paths=(
  "run.py"
  "models/TimesNet.py"
  "scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh"
)
for relative_path in "${required_paths[@]}"; do
  if [[ ! -f "$upstream_dir/$relative_path" ]]; then
    printf 'Official Time-Series-Library source is missing required file: %s\n' "$relative_path" >&2
    exit 1
  fi
done

printf 'Verified official TimesNet source at commit %s\n' "$OFFICIAL_COMMIT"
