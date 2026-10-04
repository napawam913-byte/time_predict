"""Static contracts for PatchTST's reproducible launch wrappers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "reproduction" / "03_patchtst_official" / "scripts"
REQUIREMENTS = PROJECT_ROOT / "reproduction" / "03_patchtst_official" / "requirements.common.txt"


def _script(name: str) -> str:
    path = SCRIPTS / name
    assert path.is_file(), f"missing launcher: {name}"
    return path.read_text(encoding="utf-8")


def test_common_launcher_fixes_the_fair_ettm1_patchtst_protocol_and_isolates_runs() -> None:
    """The first model comparison cannot silently change its data or patch contract."""

    source = _script("run_common.sh")
    for required in (
        "--random_seed 2021",
        "--features M",
        "--seq_len 96",
        "--pred_len 96",
        "--enc_in 7",
        "--patch_len 16",
        "--stride 8",
        "run directory already exists",
    ):
        assert required in source


def test_gpu_launcher_trains_then_exports_labels_without_editing_upstream() -> None:
    """Author test output alone is insufficient because it omits true.npy."""

    source = _script("run_official_ettm1_96_96_gpu.sh")
    common_source = _script("run_common.sh")
    assert "run_author_experiment" in source
    assert "export_ettm1_predictions.py" in common_source
    assert "predictions.npz" in common_source


def test_cpu_smoke_launcher_checks_patchtst_shapes_without_training() -> None:
    """A local smoke path should only construct and forward the official model once."""

    source = _script("run_smoke_cpu.sh")
    assert "PatchTST" in source
    assert "torch.randn" in source
    assert "96" in source
    assert "7" in source


def test_cpu_smoke_launcher_can_use_an_existing_compatible_python_environment() -> None:
    """A local shape check should not require duplicating a large torch wheel."""

    source = _script("run_smoke_cpu.sh")
    assert "PATCHTST_PYTHON" in source


def test_common_requirements_keep_numpy_below_2_for_the_immutable_upstream() -> None:
    """The pinned official source calls ``np.Inf``, removed by NumPy 2.0."""

    requirements = REQUIREMENTS.read_text(encoding="utf-8")
    assert "numpy>=1.24,<2" in requirements
