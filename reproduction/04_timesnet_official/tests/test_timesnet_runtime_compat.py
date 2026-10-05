"""Contracts for running the immutable 2023 source on supported modern Python."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = PROJECT_ROOT / "reproduction" / "04_timesnet_official"


def test_common_requirements_pin_compatible_numerical_runtime() -> None:
    """NumPy 2 removes np.Inf while the immutable source still uses it."""

    requirements = (REPRODUCTION_ROOT / "requirements.common.txt").read_text(encoding="utf-8")

    assert "torch==2.5.1" in requirements
    assert "numpy==1.26.4" in requirements
    assert "pandas==2.3.3" in requirements
    assert "scipy==1.14.1" in requirements
    assert "sktime==1.2.0" in requirements


def test_environment_scripts_offer_cpu_gpu_and_external_compatibility_paths() -> None:
    """The wrapper must make legacy APIs work without editing author files."""

    cpu = (REPRODUCTION_ROOT / "scripts" / "create_cpu_env.sh").read_text(encoding="utf-8")
    gpu = (REPRODUCTION_ROOT / "scripts" / "create_gpu_env.sh").read_text(encoding="utf-8")
    smoke = (REPRODUCTION_ROOT / "scripts" / "run_smoke_cpu.sh").read_text(encoding="utf-8")
    compat = (REPRODUCTION_ROOT / "compat" / "sitecustomize.py").read_text(encoding="utf-8")

    assert "TIMESNET_PYTHON" in cpu
    assert "download.pytorch.org/whl/cpu" in cpu
    assert "cu121" in gpu
    assert "sitecustomize" in smoke
    assert "TIMESNET_UPSTREAM_DIR" in smoke
    assert 'Path(os.environ["TIMESNET_UPSTREAM_DIR"])' in smoke
    assert "DataFrame.drop" in compat
