"""Contracts for keeping the official PatchTST source immutable and reproducible."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = PROJECT_ROOT / "reproduction" / "03_patchtst_official"
OFFICIAL_URL = "https://github.com/yuqinie98/PatchTST.git"
OFFICIAL_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def test_fetcher_and_provenance_pin_the_same_official_commit() -> None:
    """The fetch route must not silently follow a moving branch head."""

    fetch_script = REPRODUCTION_ROOT / "scripts" / "fetch_upstream.sh"
    provenance_path = REPRODUCTION_ROOT / "provenance" / "upstream.json"

    assert fetch_script.is_file(), "missing reproducible upstream fetch script"
    assert provenance_path.is_file(), "missing recorded upstream provenance"

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["repository"] == OFFICIAL_URL
    assert provenance["commit"] == OFFICIAL_COMMIT
    assert OFFICIAL_URL in fetch_script.read_text(encoding="utf-8")
    assert OFFICIAL_COMMIT in fetch_script.read_text(encoding="utf-8")


def test_verifier_explicitly_rejects_a_dirty_official_tree() -> None:
    """Project compatibility code must live beside, never inside, the upstream clone."""

    verify_script = REPRODUCTION_ROOT / "scripts" / "verify_upstream.sh"
    assert verify_script.is_file(), "missing upstream verifier"

    source = verify_script.read_text(encoding="utf-8")
    assert "diff --quiet" in source
    assert "diff --cached --quiet" in source
    assert "status --porcelain" in source


def test_project_ignore_rules_keep_refetchable_upstream_code_out_of_git() -> None:
    """The official clone remains a refetchable input, not authored project code."""

    ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "reproduction/03_patchtst_official/upstream/" in ignore_rules


def test_cpu_environment_installs_a_cpu_pytorch_wheel_for_the_smoke_script() -> None:
    """The advertised CPU smoke command needs torch even without a CUDA GPU."""

    source = (REPRODUCTION_ROOT / "scripts" / "create_cpu_env.sh").read_text(encoding="utf-8")
    assert 'torch==2.5.1' in source
    assert 'https://download.pytorch.org/whl/cpu' in source
