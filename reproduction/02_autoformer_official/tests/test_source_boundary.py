"""Behavioral checks for the immutable official-source boundary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import json


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script_name: str, project_root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ | {"PROJECT_ROOT": str(project_root)}
    return subprocess.run(
        ["bash", str(SCRIPTS / script_name)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_fetch_refuses_to_replace_a_non_git_upstream_directory(tmp_path: Path) -> None:
    """A mistaken destination must fail without deleting user files."""

    upstream = tmp_path / "reproduction/02_autoformer_official/upstream/Autoformer"
    upstream.mkdir(parents=True)
    marker = upstream / "keep-me.txt"
    marker.write_text("user file\n", encoding="utf-8")

    result = _run("fetch_upstream.sh", tmp_path)

    assert result.returncode != 0
    assert "not a Git worktree" in result.stderr
    assert marker.read_text(encoding="utf-8") == "user file\n"


def test_verify_rejects_a_dirty_official_checkout(tmp_path: Path) -> None:
    """Training must not proceed once the author source has been edited."""

    upstream = tmp_path / "reproduction/02_autoformer_official/upstream/Autoformer"
    upstream.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=upstream, check=True)
    requirements = upstream / "requirements.txt"
    requirements.write_text("numpy==1.0\n", encoding="utf-8")
    subprocess.run(["git", "add", "requirements.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=upstream, check=True, capture_output=True)
    requirements.write_text("numpy==2.0\n", encoding="utf-8")

    csv = tmp_path / "DataSet/ETTm1/ETTm1.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("date,OT\n2021-01-01,1\n", encoding="utf-8")

    result = _run("verify_upstream.sh", tmp_path)

    assert result.returncode != 0
    assert "not clean" in result.stderr


def test_verify_records_a_clean_checkout_and_dataset_hash(tmp_path: Path) -> None:
    """A clean official checkout must yield a machine-readable provenance record."""

    upstream = tmp_path / "reproduction/02_autoformer_official/upstream/Autoformer"
    upstream.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=upstream, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/thuml/Autoformer"],
        cwd=upstream,
        check=True,
    )
    (upstream / "requirements.txt").write_text("numpy==1.0\n", encoding="utf-8")
    subprocess.run(["git", "add", "requirements.txt"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=upstream, check=True, capture_output=True)

    csv = tmp_path / "DataSet/ETTm1/ETTm1.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("date,OT\n2021-01-01,1\n", encoding="utf-8")

    result = _run("verify_upstream.sh", tmp_path)

    assert result.returncode == 0, result.stderr
    provenance = json.loads(
        (tmp_path / "reproduction/02_autoformer_official/provenance/upstream.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["repository_url"] == "https://github.com/thuml/Autoformer"
    assert len(provenance["commit"]) == 40
    assert len(provenance["dataset_sha256"]) == 64
