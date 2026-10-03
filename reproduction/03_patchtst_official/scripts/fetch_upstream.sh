#!/usr/bin/env bash
set -euo pipefail

readonly OFFICIAL_URL="https://github.com/yuqinie98/PatchTST.git"
readonly OFFICIAL_COMMIT="204c21efe0b39603ad6e2ca640ef5896646ab1a9"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../../.." && pwd -P)"
upstream_dir="$project_root/reproduction/03_patchtst_official/upstream/PatchTST"

if [[ -e "$upstream_dir" && ! -d "$upstream_dir/.git" ]]; then
  printf 'Refusing to replace non-Git upstream path: %s\n' "$upstream_dir" >&2
  exit 1
fi

if [[ ! -d "$upstream_dir/.git" ]]; then
  mkdir -p "$(dirname "$upstream_dir")"
  git clone --no-checkout "$OFFICIAL_URL" "$upstream_dir"
fi

git -C "$upstream_dir" remote set-url origin "$OFFICIAL_URL"
git -C "$upstream_dir" fetch --depth 1 origin "$OFFICIAL_COMMIT"
git -C "$upstream_dir" checkout --detach "$OFFICIAL_COMMIT"
"$script_dir/verify_upstream.sh"

printf 'Pinned official PatchTST source at %s\n' "$upstream_dir"
