#!/usr/bin/env bash
set -euo pipefail

cuda_wheel="${1:-cu121}"
case "$cuda_wheel" in
  cu118|cu121|cu124) ;;
  *)
    printf 'Unsupported CUDA wheel selector %s; use cu118, cu121, or cu124.\n' "$cuda_wheel" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TORCH_INDEX_URL="https://download.pytorch.org/whl/$cuda_wheel" "$script_dir/create_cpu_env.sh"

printf 'Installed PatchTST GPU PyTorch wheel for %s.\n' "$cuda_wheel"
