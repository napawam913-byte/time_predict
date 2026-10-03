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
CSV_PATH="$PROJECT_ROOT/DataSet/ETTm1/ETTm1.csv"
REPOSITORY_URL="https://github.com/thuml/Autoformer"
PROVENANCE_PATH="$EXPERIMENT_ROOT/provenance/upstream.json"

if ! git -C "$UPSTREAM_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "official upstream is missing or not a Git worktree: $UPSTREAM_DIR" >&2
  exit 1
fi
if ! git -C "$UPSTREAM_DIR" diff --quiet || [[ -n "$(git -C "$UPSTREAM_DIR" status --porcelain)" ]]; then
  echo "official upstream is not clean; refusing to run modified source" >&2
  exit 1
fi
REMOTE_URL=$(git -C "$UPSTREAM_DIR" remote get-url origin 2>/dev/null || true)
if [[ "$REMOTE_URL" != "$REPOSITORY_URL" ]]; then
  echo "unexpected upstream origin: $REMOTE_URL" >&2
  exit 1
fi
if [[ ! -f "$CSV_PATH" ]]; then
  echo "ETTm1 CSV is missing: $CSV_PATH" >&2
  exit 1
fi
if [[ ! -f "$UPSTREAM_DIR/requirements.txt" ]]; then
  echo "official requirements file is missing: $UPSTREAM_DIR/requirements.txt" >&2
  exit 1
fi

mkdir -p "$(dirname "$PROVENANCE_PATH")"
python3 - "$PROVENANCE_PATH" "$UPSTREAM_DIR" "$CSV_PATH" "$REPOSITORY_URL" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

output_path = Path(sys.argv[1])
upstream_dir = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
repository_url = sys.argv[4]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

commit = subprocess.check_output(
    ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True
).strip()
record = {
    "repository_url": repository_url,
    "commit": commit,
    "source_clean": True,
    "requirements_sha256": sha256(upstream_dir / "requirements.txt"),
    "dataset_path": str(csv_path),
    "dataset_sha256": sha256(csv_path),
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
}
output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
