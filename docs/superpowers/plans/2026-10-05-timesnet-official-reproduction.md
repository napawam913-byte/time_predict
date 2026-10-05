# TimesNet Official Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Reproduce the authors' paper-era TimesNet implementation on ETTm1 L=96 to H=96, then compare its full label-aligned test predictions with the existing baselines.

**Architecture:** Keep the 2023-03-31 author TSLib commit as an immutable upstream clone. Version-control only a thin shell for source verification, environment creation, run isolation, archive export, period observation, and shared evaluation; never copy or modify models/TimesNet.py or the author training classes. The author run.py creates checkpoints and pred.npy/true.npy; the shell validates and packages those unchanged normalized arrays into the existing shared evaluation contract.

**Tech Stack:** Bash, Python 3.12-compatible virtual environment, PyTorch 2.5.1, NumPy 1.26.4, pandas 2.3.3 plus a narrowly scoped external legacy-API shim, pytest, matplotlib, the existing reproduction/common/ltsf_evaluation package, and the author TSLib code at commit 2665a3143dae12d1cbcc31ddd396bbff48773bce.

**Spec:** docs/superpowers/specs/2026-10-05-timesnet-official-reproduction-design.md

## Global Constraints

- Upstream is exactly https://github.com/thuml/Time-Series-Library.git at 2665a3143dae12d1cbcc31ddd396bbff48773bce (2023-03-31); keep it under reproduction/04_timesnet_official/upstream/Time-Series-Library/, ignored by Git and unmodified.
- Use the author ETTm1 long-forecast configuration: model TimesNet, task long_term_forecast, M features, seq_len=96, label_len=48, pred_len=96, e_layers=2, d_layers=1, d_model=64, d_ff=64, top_k=5, factor=3, author seed 2021, and the author default freq=h made explicit for reproducibility.
- Use DataSet/ETTm1/ETTm1.csv; retain the chronological author ETTm1 split and train-only StandardScaler normalization; do not introduce a new split or scaler.
- All comparison archives use normalized prediction, target, columns, and scale; TimesNet must contain exactly (11425, 96, 7) windows before comparison.
- Reuse reproduction/common/ltsf_evaluation without changing its existing public semantics or altering prior model results.
- No authored model architecture, FFT, 2D reshape, convolution, training loop, or checkpoint behavior; project code may only launch, observe, validate, package, and compare author code.
- Never overwrite a run directory; all result/history entries are append-only.

## Review Focus

- Stale or wrong upstream: Task 1 tests the fetcher, provenance, exact TSLib URL/commit, and dirty/wrong clone rejection.
- Old-source runtime compatibility: Task 2 pins NumPy <2, uses an external pandas positional-axis compatibility shim, and smokes the real author model without modifying author files.
- Partial or mismatched author outputs: Task 3 rejects zero, multiple, setting-mismatched, non-finite, or incomplete pred.npy/true.npy artifacts.
- ETTm1 test-tail preservation: Task 3 verifies all 11,425 single-batch author windows are packaged.
- Misleading cross-model metric: Task 5 rejects column-order, shape, and elementwise-label mismatch before rendering metrics.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| reproduction/04_timesnet_official/provenance/upstream.json | Immutable author repository URL, commit, source path, and official ETTm1 script path. |
| reproduction/04_timesnet_official/requirements.common.txt | Shell-owned compatibility pins; never edit upstream requirements. |
| reproduction/04_timesnet_official/scripts/*.sh | Reproducible source, environment, smoke, and training entry points. |
| reproduction/04_timesnet_official/src/timesnet_reproduction/export.py | Strict discovery and NPZ packaging of author pred.npy and true.npy. |
| reproduction/04_timesnet_official/src/timesnet_reproduction/observation.py | Read-only use of author FFT_for_Period to record selected periods and official reshape geometry. |
| reproduction/04_timesnet_official/src/timesnet_reproduction/evaluation.py | TimesNet archive loader and adapter over shared LSTF evaluation. |
| reproduction/04_timesnet_official/scripts/*.py | Export and label-verified comparison CLI boundaries. |
| reproduction/04_timesnet_official/tests/*.py | TDD contracts for source boundary, launcher, observation, export, and comparison. |
| reproduction/04_timesnet_official/README.md | User commands, boundaries, release checklist, and official-config differences. |
| .gitignore | Keeps upstream, environments, and runs untracked. |
| experiment/results.md, handoff.md, brainstorm.md | Append-only records after a verified cloud result. |

### Task 1: Immutable author-source boundary

**Files:**
- Create: reproduction/04_timesnet_official/provenance/upstream.json
- Create: reproduction/04_timesnet_official/provenance/README.md
- Create: reproduction/04_timesnet_official/scripts/fetch_upstream.sh
- Create: reproduction/04_timesnet_official/scripts/verify_upstream.sh
- Create: reproduction/04_timesnet_official/tests/test_timesnet_source_boundary.py
- Modify: .gitignore

**Interfaces:**
- Consumes: URL https://github.com/thuml/Time-Series-Library.git, commit 2665a3143dae12d1cbcc31ddd396bbff48773bce, and paths run.py, models/TimesNet.py, scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh.
- Produces: a refetchable verified upstream root at upstream/Time-Series-Library; later launchers call bash scripts/verify_upstream.sh before reading it.

- [ ] **Step 1: Write the failing source-boundary test**

    def test_fetcher_and_provenance_pin_paper_era_tslib_commit() -> None:
        provenance = json.loads(PROVENANCE.read_text())
        assert provenance["repository"] == "https://github.com/thuml/Time-Series-Library.git"
        assert provenance["commit"] == "2665a3143dae12d1cbcc31ddd396bbff48773bce"
        assert "TimesNet_ETTm1.sh" in provenance["official_ettm1_script"]

    def test_verifier_rejects_dirty_or_wrong_author_tree() -> None:
        source = VERIFY_SCRIPT.read_text()
        assert "rev-parse HEAD" in source
        assert "remote get-url origin" in source
        assert "status --porcelain" in source

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=reproduction/common:reproduction/02_autoformer_official/src:reproduction/03_patchtst_official/src pytest -q reproduction/04_timesnet_official/tests/test_timesnet_source_boundary.py

Expected: FAIL because the TimesNet reproduction files do not exist.

- [ ] **Step 3: Implement provenance, fetcher, verifier, and ignore rule**

fetch_upstream.sh clones only when absent, sets origin to the exact URL, checks out detached exact commit, and calls the verifier. verify_upstream.sh rejects a missing source, a different origin, any HEAD other than the pin, untracked/staged/unstaged changes, or a source missing the three declared files. Add reproduction/04_timesnet_official/upstream/, .venv/, and runs/ to .gitignore.

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

    git add .gitignore reproduction/04_timesnet_official/provenance reproduction/04_timesnet_official/scripts/fetch_upstream.sh reproduction/04_timesnet_official/scripts/verify_upstream.sh reproduction/04_timesnet_official/tests/test_timesnet_source_boundary.py
    git commit -m "feat: pin TimesNet author source"

### Task 2: Compatible environment and read-only period observation

**Files:**
- Create: reproduction/04_timesnet_official/requirements.common.txt
- Create: reproduction/04_timesnet_official/scripts/create_cpu_env.sh
- Create: reproduction/04_timesnet_official/scripts/create_gpu_env.sh
- Create: reproduction/04_timesnet_official/scripts/run_smoke_cpu.sh
- Create: reproduction/04_timesnet_official/compat/sitecustomize.py
- Create: reproduction/04_timesnet_official/src/timesnet_reproduction/__init__.py
- Create: reproduction/04_timesnet_official/src/timesnet_reproduction/observation.py
- Create: reproduction/04_timesnet_official/tests/test_timesnet_runtime_compat.py
- Create: reproduction/04_timesnet_official/tests/test_timesnet_period_observation.py

**Interfaces:**
- Consumes: verified upstream models.TimesNet.Model and models.TimesNet.FFT_for_Period; TIMESNET_PYTHON is an optional executable override.
- Produces: observe_periods(upstream_root: Path, values: torch.Tensor, top_k: int) -> PeriodObservation. PeriodObservation records selected periods, FFT weights shape, total length, and each (cycles, period) grid. It is diagnostic only and never changes model input/output.

- [ ] **Step 1: Write failing environment and observation tests**

    def test_common_requirements_preserve_old_source_compatibility() -> None:
        requirements = REQUIREMENTS.read_text()
        assert "torch==2.5.1" in requirements
        assert "numpy==1.26.4" in requirements
        assert "pandas==2.3.3" in requirements
        assert "scipy==1.14.1" in requirements

    def test_observation_uses_author_fft_and_reports_period_grid(monkeypatch, tmp_path) -> None:
        observation = observe_periods(tmp_path, torch.randn(2, 192, 64), top_k=2)
        assert len(observation.periods) == 2
        assert all(cycles * period >= 192 for cycles, period in observation.grids)

- [ ] **Step 2: Run tests to verify they fail**

Run: PYTHONPATH=reproduction/04_timesnet_official/src pytest -q reproduction/04_timesnet_official/tests/test_timesnet_runtime_compat.py reproduction/04_timesnet_official/tests/test_timesnet_period_observation.py

Expected: FAIL because requirements and observation API do not exist.

- [ ] **Step 3: Implement environment scripts and observation API**

Pin the wrapper runtime to torch==2.5.1, numpy==1.26.4, pandas==2.3.3, scikit-learn==1.7.2, matplotlib==3.10.8, einops==0.8.2, scipy==1.14.1, sktime==1.2.0, reformer-pytorch==1.4.4, patool==1.12, and tqdm==4.67.1; use locally available Python 3.12. Because the 2023 author loader calls the pandas-1.x positional `DataFrame.drop(..., 1)` API, place a minimal external `compat/sitecustomize.py` shim first on `PYTHONPATH`; do not modify the author source. CPU installs torch from the CPU index; GPU takes cu121 and installs matching torch. run_smoke_cpu.sh must use verified author Model for one B×96×7 forecast and print forecast_shape=(B, 96, 7), then call observe_periods on author TimesBlock-length data to print selected period/grid metadata.

observe_periods dynamically imports upstream FFT_for_Period; it must not call torch.fft itself. It calculates each grid as the same padded (ceil(total_length / period), period) layout used by author TimesBlock.forward and validates top_k > 0, finite rank-3 input, and positive periods.

- [ ] **Step 4: Run focused tests and real CPU smoke**

Run: PYTHONPATH=reproduction/04_timesnet_official/src pytest -q reproduction/04_timesnet_official/tests/test_timesnet_runtime_compat.py reproduction/04_timesnet_official/tests/test_timesnet_period_observation.py

Then run:
    bash reproduction/04_timesnet_official/scripts/fetch_upstream.sh
    bash reproduction/04_timesnet_official/scripts/create_cpu_env.sh
    bash reproduction/04_timesnet_official/scripts/run_smoke_cpu.sh

Expected: tests PASS; smoke prints a B×96×7 forecast plus author-FFT period/grid observation and does not edit upstream files.

- [ ] **Step 5: Commit**

    git add reproduction/04_timesnet_official/requirements.common.txt reproduction/04_timesnet_official/compat/sitecustomize.py reproduction/04_timesnet_official/scripts/create_cpu_env.sh reproduction/04_timesnet_official/scripts/create_gpu_env.sh reproduction/04_timesnet_official/scripts/run_smoke_cpu.sh reproduction/04_timesnet_official/src/timesnet_reproduction reproduction/04_timesnet_official/tests/test_timesnet_runtime_compat.py reproduction/04_timesnet_official/tests/test_timesnet_period_observation.py
    git commit -m "feat: add TimesNet smoke environment"

### Task 3: Strict packaging of untouched author predictions

**Files:**
- Create: reproduction/04_timesnet_official/src/timesnet_reproduction/export.py
- Create: reproduction/04_timesnet_official/scripts/export_ettm1_predictions.py
- Create: reproduction/04_timesnet_official/tests/test_timesnet_export_contract.py

**Interfaces:**
- Consumes: one author run containing checkpoints/<setting>/checkpoint.pth and results/<setting>/pred.npy, true.npy; ETTm1 CSV with a date column.
- Produces: discover_official_artifacts(official_run_dir: Path) -> OfficialArtifacts; export_from_official_run(official_run_dir: Path, dataset_csv: Path, archive_path: Path) -> Path; a normalized NPZ with prediction, target, columns, scale.

- [ ] **Step 1: Write failing export and tail-completeness tests**

    def test_export_writes_declared_normalized_archive(tmp_path: Path) -> None:
        archive = export_from_official_run(author_run_with_one_setting(tmp_path, windows=11425), ettm1_csv(tmp_path), tmp_path / "predictions.npz")
        with np.load(archive, allow_pickle=False) as result:
            assert result["prediction"].shape == (11425, 96, 7)
            assert set(result.files) == {"prediction", "target", "columns", "scale"}

    def test_export_rejects_missing_ambiguous_mismatched_or_incomplete_author_results(tmp_path: Path) -> None:
        csv_path = ettm1_csv(tmp_path)
        with pytest.raises(ValueError, match="exactly one|setting|11425"):
            export_from_official_run(tmp_path / "missing", csv_path, tmp_path / "predictions.npz")

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=reproduction/common:reproduction/04_timesnet_official/src pytest -q reproduction/04_timesnet_official/tests/test_timesnet_export_contract.py

Expected: FAIL because the exporter does not exist.

- [ ] **Step 3: Implement artifact discovery and packaging**

Read only author-produced pred.npy and true.npy; do not invoke a second model forward pass. Require exactly one common setting name for checkpoint and result paths, equal finite rank-3 arrays, and actual ETTm1 output shape (11425, 96, 7). Read CSV columns in file order excluding date, call shared ForecastResult validation, and save scale=normalized. The CLI takes --official-run-dir, --csv, and --output only.

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2.

Expected: PASS, including the incomplete-tail rejection.

- [ ] **Step 5: Commit**

    git add reproduction/04_timesnet_official/src/timesnet_reproduction/export.py reproduction/04_timesnet_official/scripts/export_ettm1_predictions.py reproduction/04_timesnet_official/tests/test_timesnet_export_contract.py
    git commit -m "feat: export TimesNet official predictions"

### Task 4: Isolated official GPU launcher

**Files:**
- Create: reproduction/04_timesnet_official/scripts/run_common.sh
- Create: reproduction/04_timesnet_official/scripts/run_official_ettm1_96_96_gpu.sh
- Create: reproduction/04_timesnet_official/tests/test_timesnet_launch_wrappers.py
- Create: reproduction/04_timesnet_official/README.md

**Interfaces:**
- Consumes: a valid run name, verified upstream, TimesNet .venv, project ETTm1 CSV, and export CLI.
- Produces: runs/<run-name>/config.json, command.txt, python-version.txt, environment.txt, official/stdout.log, official/stderr.log, checkpoint/results files, and predictions.npz. Bash function run_author_experiment(run_id) is invoked by GPU entry point.

- [ ] **Step 1: Write failing launcher/static-contract tests**

    def test_launcher_fixes_author_ettm1_96_to_96_protocol_and_isolates_runs() -> None:
        source = RUN_COMMON.read_text()
        for argument in ("--model TimesNet", "--seq_len 96", "--pred_len 96", "--d_model 64", "--d_ff 64", "--top_k 5", "--freq h"):
            assert argument in source
        assert "run directory already exists" in source

    def test_gpu_launcher_runs_author_code_then_exports_full_labels() -> None:
        assert "run_author_experiment" in GPU_LAUNCHER.read_text()
        assert "export_ettm1_predictions.py" in RUN_COMMON.read_text()

- [ ] **Step 2: Run tests to verify they fail**

Run: PYTHONPATH=reproduction/04_timesnet_official/src pytest -q reproduction/04_timesnet_official/tests/test_timesnet_launch_wrappers.py

Expected: FAIL because launchers do not exist.

- [ ] **Step 3: Implement launchers and operator documentation**

run_common.sh rejects unsafe/existing run IDs, verifies source before creating any directory, serializes every author argument to config.json, and executes upstream/Time-Series-Library/run.py from the run's official/ directory so relative results/ and test_results/ stay contained. Pass all protocol arguments explicitly, including --freq h, source default --batch_size 32, --train_epochs 10, --patience 3, --learning_rate 0.0001, --lradj type1, --num_workers 10, --itr 1, --checkpoints <official/checkpoints>, --gpu 0, and omit --use_multi_gpu. Capture stdout/stderr, export only after author run.py exits successfully, and never overwrite a run.

The GPU entry point checks CUDA with its chosen interpreter and calls run_author_experiment. README shows cloud commands: pull, fetch, create cu121 environment, launch with fresh run name, tail official/stdout.log, export only after training completes, then compare. Document that explicit freq=h is identical to the author ETTm1 script's implicit default; it is an official-config choice, not a statement that ETTm1 is hourly.

- [ ] **Step 4: Run tests and non-training validation**

Run: PYTHONPATH=reproduction/04_timesnet_official/src pytest -q reproduction/04_timesnet_official/tests/test_timesnet_launch_wrappers.py

Then run:
    bash reproduction/04_timesnet_official/scripts/verify_upstream.sh

Expected: PASS; verifier prints pinned commit; no GPU training starts.

- [ ] **Step 5: Commit**

    git add reproduction/04_timesnet_official/scripts/run_common.sh reproduction/04_timesnet_official/scripts/run_official_ettm1_96_96_gpu.sh reproduction/04_timesnet_official/tests/test_timesnet_launch_wrappers.py reproduction/04_timesnet_official/README.md
    git commit -m "feat: add TimesNet official GPU launcher"

### Task 5: Label-verified TimesNet comparison and regression suite

**Files:**
- Create: reproduction/04_timesnet_official/src/timesnet_reproduction/evaluation.py
- Create: reproduction/04_timesnet_official/scripts/compare_ettm1_l96_h96.py
- Create: reproduction/04_timesnet_official/tests/test_timesnet_comparison_cli.py
- Modify: reproduction/04_timesnet_official/README.md

**Interfaces:**
- Consumes: load_timesnet_result(run_dir: Path) -> tuple[ForecastResult, list[str]], baseline archives, ETTm1 CSV, shared TrainingScaler/evaluate_models.
- Produces: compare_archives(timesnet_archive: Path, seasonal_archive: Path, dlinear_archive: Path, csv_path: Path) -> dict[str, object]; comparison/ettm1_l96_h96_initial_metrics.json, comparison/ettm1_l96_h96_initial_metrics.md, and comparison/ettm1_l96_h96_initial_prediction.png after all-label pass.

- [ ] **Step 1: Write failing comparison tests**

    def test_comparison_writes_metrics_only_after_strict_label_alignment(tmp_path: Path) -> None:
        payload = compare_archives(timesnet_archive, seasonal_archive, dlinear_archive, csv_path)
        assert payload["alignment"]["passed"] is True

    def test_comparison_rejects_column_shape_and_value_misalignment(tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="column order|labels do not align|shape"):
            compare_archives(misaligned_timesnet_archive, seasonal_archive, dlinear_archive, csv_path)

- [ ] **Step 2: Run tests to verify they fail**

Run: PYTHONPATH=reproduction/common:reproduction/04_timesnet_official/src pytest -q reproduction/04_timesnet_official/tests/test_timesnet_comparison_cli.py

Expected: FAIL because evaluator and CLI do not exist.

- [ ] **Step 3: Implement evaluator adapter and report CLI**

Define compare_archives(timesnet_archive: Path, seasonal_archive: Path, dlinear_archive: Path, csv_path: Path) -> dict[str, object] by reusing load_normalized_archive(require_scale=True), TrainingScaler.from_csv, and shared evaluate_models("TimesNet", timesnet, baselines, scaler, columns). Accept --timesnet-run, --seasonal-naive, --dlinear, --csv, and --output-dir plus existing plotting options. Require TimesNet and baselines to match CSV column order and strict normalized labels before writing JSON, Markdown, or PNG. Render global normalized/original MSE/MAE, per-variable original MSE, and one qualitative original-scale plot. Limitations name single author seed 2021, L=96, 2023 TSLib pin, and freq=h official-script setting.

- [ ] **Step 4: Run all unit/regression tests**

Run:
    PYTHONPATH=reproduction/common:reproduction/02_autoformer_official/src:reproduction/03_patchtst_official/src:reproduction/04_timesnet_official/src \
    reproduction/02_autoformer_official/.venv/bin/python -m pytest -q \
    reproduction/common/tests reproduction/02_autoformer_official/tests reproduction/03_patchtst_official/tests reproduction/04_timesnet_official/tests

Expected: PASS; prior Autoformer/PatchTST tests stay green.

- [ ] **Step 5: Commit**

    git add reproduction/04_timesnet_official/src/timesnet_reproduction/evaluation.py reproduction/04_timesnet_official/scripts/compare_ettm1_l96_h96.py reproduction/04_timesnet_official/tests/test_timesnet_comparison_cli.py reproduction/04_timesnet_official/README.md
    git commit -m "feat: compare aligned TimesNet forecasts"

### Task 6: Cloud run, verification, and research-record update

**Files:**
- Modify: experiment/results.md
- Modify: handoff.md
- Modify: brainstorm.md
- Modify: reproduction/04_timesnet_official/README.md

**Interfaces:**
- Consumes: a completed cloud run with predictions.npz, baseline archives at reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_{seasonal_naive,dlinear}_cloud20261003_r2/predictions.npz, and Task 5 comparison CLI.
- Produces: append-only TimesNet run configuration/metric summary, handoff status, and one falsifiable next hypothesis.

- [ ] **Step 1: Add the result release checklist before cloud launch**

Add this exact checklist to README and execute it manually after the cloud run:

    assert archive_shape == (11425, 96, 7)
    assert comparison_json["alignment"]["passed"] is True
    assert "single-run" in comparison_json["limitations"][0]

Do not invent metric expectations or commit a cloud artifact fixture.

- [ ] **Step 2: Launch fresh L40 run after cloud setup**

Run:
    git pull origin main
    bash reproduction/04_timesnet_official/scripts/fetch_upstream.sh
    bash reproduction/04_timesnet_official/scripts/create_gpu_env.sh cu121
    bash reproduction/04_timesnet_official/scripts/run_official_ettm1_96_96_gpu.sh ettm1_l96_h96_timesnet_seed2021_gpu_YYYYMMDD

Expected: official/stdout.log reports CUDA, fixed author arguments, training, and testing; run gains the complete normalized archive.

- [ ] **Step 3: Verify archive and create the only admissible comparison**

Run:
    PYTHONPATH=reproduction/common:reproduction/04_timesnet_official/src \
    reproduction/04_timesnet_official/.venv/bin/python \
    reproduction/04_timesnet_official/scripts/compare_ettm1_l96_h96.py \
      --timesnet-run reproduction/04_timesnet_official/runs/<fresh-run-name> \
      --seasonal-naive reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_seasonal_naive_cloud20261003_r2/predictions.npz \
      --dlinear reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_dlinear_cloud20261003_r2/predictions.npz \
      --csv DataSet/ETTm1/ETTm1.csv \
      --output-dir reproduction/04_timesnet_official/comparison

Expected: only succeeds with alignment.passed=true and a (11425, 96, 7) archive. If it fails, retain run and diagnose split/scale/artifact boundary before retraining.

- [ ] **Step 4: Append evidence-backed project records**

Append actual run name, author commit, all fixed configuration, environment/GPU, archive shape, MSE/MAE/RMSE, per-variable findings, and limitations to experiment/results.md; append completed/next state to handoff.md; append a testable conclusion or negative result to brainstorm.md. Do not claim a paper-table reproduction or general superiority from one run.

- [ ] **Step 5: Run final verification and commit records**

Run the Task 5 regression command plus git diff --check; inspect comparison Markdown and JSON. Commit only versionable code/docs, never upstream/, .venv/, or runs/:

    git add experiment/results.md handoff.md brainstorm.md reproduction/04_timesnet_official/README.md
    git commit -m "docs: record TimesNet aligned reproduction"

## Plan Self-Review

- **Spec coverage:** Task 1 enforces corrected source; Task 2 provides isolated runtime and non-invasive period observation; Task 3 protects archive completeness; Task 4 launches unchanged author training; Task 5 enforces label alignment and metrics; Task 6 records only verified cloud evidence. Every design section maps to a task.
- **Step scan:** Every code task follows failing test, targeted run, minimal implementation, passing verification, and commit. Cloud record has no invented expected metric.
- **Type consistency:** Task 3 defines OfficialArtifacts, discover_official_artifacts, export_from_official_run; Task 4 calls its CLI; Task 5 defines load_timesnet_result; Task 6 calls Task 5 CLI. Every archive shape is [windows, horizon, channels].
- **Review focus coverage:** Task 1 owns upstream identity; Task 2 runtime/model smoke; Task 3 output ambiguity and tail completeness; Task 5 strict cross-model labels.
- **Proportion:** The plan fixes interfaces, values, tests, and verification commands without reproducing TimesNet or author training-loop source.
