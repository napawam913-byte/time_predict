"""Static contracts for the isolated author-code GPU launcher."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = PROJECT_ROOT / "reproduction" / "04_timesnet_official"
RUN_COMMON = REPRODUCTION_ROOT / "scripts" / "run_common.sh"
GPU_LAUNCHER = REPRODUCTION_ROOT / "scripts" / "run_official_ettm1_96_96_gpu.sh"
README = REPRODUCTION_ROOT / "README.md"


def test_launcher_fixes_author_ettm1_96_to_96_protocol_and_isolates_runs() -> None:
    """The wrapper must make the exact first-round protocol inspectable."""

    source = RUN_COMMON.read_text(encoding="utf-8")
    for argument in (
        "--task_name long_term_forecast",
        "--model TimesNet",
        "--seq_len 96",
        "--pred_len 96",
        "--d_model 64",
        "--d_ff 64",
        "--top_k 5",
        "--freq h",
        "--batch_size 32",
        "--train_epochs 10",
    ):
        assert argument in source
    assert "run directory already exists and will not be overwritten" in source
    assert source.index('bash "$script_dir/verify_upstream.sh"') < source.index(
        'mkdir -p "$official_run_dir"'
    )
    assert 'cd "$official_run_dir"' in source
    assert "config.json" in source
    assert "command.txt" in source


def test_gpu_launcher_checks_cuda_then_runs_author_code_and_exports_labels() -> None:
    """Training is allowed only with CUDA and export happens after the author command."""

    common_source = RUN_COMMON.read_text(encoding="utf-8")
    gpu_source = GPU_LAUNCHER.read_text(encoding="utf-8")

    assert "torch.cuda.is_available()" in gpu_source
    assert "run_author_experiment" in gpu_source
    assert "export_ettm1_predictions.py" in common_source
    assert common_source.index("run.py") < common_source.index("export_ettm1_predictions.py")
    assert "PYTHONPATH=\"$compat_dir:$upstream_dir" in common_source


def test_operator_readme_documents_the_cloud_sequence_and_frequency_boundary() -> None:
    """A future cloud user should not mistake the author's default freq for data cadence."""

    source = README.read_text(encoding="utf-8")
    for command in (
        "git pull origin main",
        "fetch_upstream.sh",
        "create_gpu_env.sh cu121",
        "run_official_ettm1_96_96_gpu.sh",
        "tail -f",
    ):
        assert command in source
    assert "freq=h" in source
    assert "15 分钟" in source
