#!/usr/bin/env bash
set -euo pipefail

readonly OFFICIAL_URL="https://github.com/thuml/Time-Series-Library.git"
readonly OFFICIAL_COMMIT="2665a3143dae12d1cbcc31ddd396bbff48773bce"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../../.." && pwd -P)"
upstream_dir="$project_root/reproduction/04_timesnet_official/upstream/Time-Series-Library"

if [[ -e "$upstream_dir" && ! -d "$upstream_dir/.git" ]]; then
  printf 'Refusing to replace non-Git upstream path: %s\n' "$upstream_dir" >&2
  exit 1
fi

if [[ ! -d "$upstream_dir/.git" ]]; then
  mkdir -p "$(dirname "$upstream_dir")"
  git init --quiet "$upstream_dir"
fi

if git -C "$upstream_dir" remote get-url origin >/dev/null 2>&1; then
  git -C "$upstream_dir" remote set-url origin "$OFFICIAL_URL"
else
  git -C "$upstream_dir" remote add origin "$OFFICIAL_URL"
fi
git -C "$upstream_dir" fetch --depth 1 origin "$OFFICIAL_COMMIT"
git -C "$upstream_dir" checkout --detach "$OFFICIAL_COMMIT"
"$script_dir/verify_upstream.sh"

printf 'Pinned official TimesNet source at %s\n' "$upstream_dir"
