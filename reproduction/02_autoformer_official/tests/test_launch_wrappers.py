"""Safety contracts shared by the Autoformer CPU launch wrappers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _clean_project_root(tmp_path: Path) -> Path:
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
    return tmp_path


def _run(script_name: str, project_root: Path, run_id: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ | {"PROJECT_ROOT": str(project_root)}
    return subprocess.run(
        ["bash", str(SCRIPTS / script_name), run_id],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_smoke_run_refuses_to_overwrite_an_existing_run_directory(tmp_path: Path) -> None:
    """A run id is immutable even before any Python training can start."""

    project_root = _clean_project_root(tmp_path)
    run_dir = project_root / "reproduction/02_autoformer_official/runs/taken"
    run_dir.mkdir(parents=True)
    marker = run_dir / "keep-me.txt"
    marker.write_text("existing output\n", encoding="utf-8")

    result = _run("run_smoke_cpu.sh", project_root, "taken")

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "existing output\n"


def test_formal_run_rejects_a_path_like_run_id(tmp_path: Path) -> None:
    """Run ids cannot escape the experiment's dedicated runs directory."""

    project_root = _clean_project_root(tmp_path)

    result = _run("run_official_ettm1_96_96_cpu.sh", project_root, "../escape")

    assert result.returncode != 0
    assert "one directory name" in result.stderr


def test_environment_and_training_wrappers_use_a_writable_matplotlib_cache() -> None:
    """Matplotlib must never fall back to the read-only home configuration path."""

    environment_script = (SCRIPTS / "create_cpu_env.sh").read_text(encoding="utf-8")
    run_common = (SCRIPTS / "run_common.sh").read_text(encoding="utf-8")

    assert "MPLCONFIGDIR" in environment_script
    assert "MPLCONFIGDIR" in run_common
    assert "PIP_CACHE_DIR" in environment_script
    assert "PIP_CACHE_DIR" in run_common
