#!/usr/bin/env bash
set -euo pipefail

readonly OFFICIAL_URL="https://github.com/yuqinie98/PatchTST.git"
readonly OFFICIAL_COMMIT="204c21efe0b39603ad6e2ca640ef5896646ab1a9"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../../.." && pwd -P)"
upstream_dir="$project_root/reproduction/03_patchtst_official/upstream/PatchTST"

if [[ ! -d "$upstream_dir/.git" ]]; then
  printf 'Official PatchTST source is absent: run fetch_upstream.sh first.\n' >&2
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
  printf 'Official PatchTST working tree has unstaged modifications.\n' >&2
  exit 1
fi

if ! git -C "$upstream_dir" diff --cached --quiet; then
  printf 'Official PatchTST working tree has staged modifications.\n' >&2
  exit 1
fi

if [[ -n "$(git -C "$upstream_dir" status --porcelain)" ]]; then
  printf 'Official PatchTST working tree has untracked or otherwise dirty files.\n' >&2
  exit 1
fi

printf 'Verified official PatchTST source at commit %s\n' "$OFFICIAL_COMMIT"
