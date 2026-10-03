#!/usr/bin/env python3
"""Export normalized PatchTST ETTm1 predictions and labels from one official run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for source in (
    PROJECT_ROOT / "reproduction" / "common",
    PROJECT_ROOT / "reproduction" / "03_patchtst_official" / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from patchtst_reproduction.export import export_from_official_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-supervised-dir", type=Path, required=True)
    parser.add_argument("--official-run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = export_from_official_run(
        upstream_supervised_dir=args.upstream_supervised_dir,
        official_run_dir=args.official_run_dir,
        config_path=args.config,
        archive_path=args.output,
    )
    print(f"Wrote normalized prediction archive: {archive}")


if __name__ == "__main__":
    main()
