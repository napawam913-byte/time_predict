#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd -P)
  EXPERIMENT_ROOT="$PROJECT_ROOT/reproduction/02_autoformer_official"
else
  EXPERIMENT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
  PROJECT_ROOT=$(cd "$EXPERIMENT_ROOT/../.." && pwd -P)
fi
UPSTREAM_DIR="$EXPERIMENT_ROOT/upstream/Autoformer"
REPOSITORY_URL="https://github.com/thuml/Autoformer"

if [[ -e "$UPSTREAM_DIR" ]]; then
  if ! git -C "$UPSTREAM_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "refusing to replace existing upstream directory: not a Git worktree: $UPSTREAM_DIR" >&2
    exit 1
  fi
else
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone "$REPOSITORY_URL" "$UPSTREAM_DIR"
fi

REMOTE_URL=$(git -C "$UPSTREAM_DIR" remote get-url origin 2>/dev/null || true)
if [[ "$REMOTE_URL" != "$REPOSITORY_URL" ]]; then
  echo "unexpected upstream origin: $REMOTE_URL" >&2
  exit 1
fi

bash "$SCRIPT_DIR/verify_upstream.sh"
