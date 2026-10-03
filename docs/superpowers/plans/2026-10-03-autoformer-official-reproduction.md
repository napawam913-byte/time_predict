# Autoformer Official Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an immutable snapshot of the authors' Autoformer implementation on ETTm1 (`96 -> 96`) and produce a label-verified comparison with the existing Seasonal Naive and DLinear baselines.

**Architecture:** Keep the official repository as an untouched nested Git checkout. Place all environment setup, CPU launch commands, provenance records, metric conversion, comparison, and visualization in a surrounding `02_autoformer_official` experiment directory. The comparison library treats official predictions as normalized data and existing baseline predictions as original-scale data, converts both with one train-only scaler, and rejects any misaligned labels.

**Tech Stack:** Bash, Python 3.12, PyTorch CPU, NumPy, Pandas, scikit-learn, Matplotlib, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-10-03-autoformer-official-reproduction-design.md`

## Global Constraints

- Use the existing `DataSet/ETTm1/ETTm1.csv`; do not download or copy a second CSV.
- Use the authors' `https://github.com/thuml/Autoformer` as an unmodified pinned checkout under `upstream/Autoformer`.
- The formal model command must use `features=M`, 7 input/output variables, `seq_len=96`, `label_len=48`, and `pred_len=96`.
- Preserve every existing `reproduction/01_dlinear_baseline/runs/*` artifact; no command may overwrite it.
- Fit scaler mean/std only from raw rows `[0,34560)`; use population standard deviation (`ddof=0`).
- The main comparison must show normalized and original-scale metrics, plus per-variable metrics.
- No source patch is permitted in `upstream/Autoformer`; source dirtiness is a hard failure.
- The top-level workspace is not a Git repository, so tasks record verification instead of creating commits.

## Review Focus

- Existing input CSV is modified or a different copy is accidentally used: provenance must include its SHA-256 and refuse missing files.
- A repeated run name would overwrite checkpoints or metrics: launch wrappers must fail before invoking Python if `runs/<run-id>` exists.
- Official and baseline labels differ by one forecasting window: comparison must reject shape or normalized-label mismatch before calculating metrics.
- Metric scale is silently mixed: output must label each metric section `normalized` or `original_scale` and test both conversions.
- Official source becomes dirty through a convenience edit: launch wrapper must call `git diff --quiet` and fail before training.

### Task 1: Establish immutable official-source and provenance boundary

**Files:**
- Create: `reproduction/02_autoformer_official/README.md`
- Create: `reproduction/02_autoformer_official/scripts/fetch_upstream.sh`
- Create: `reproduction/02_autoformer_official/scripts/verify_upstream.sh`
- Create: `reproduction/02_autoformer_official/provenance/README.md`

**Interfaces:**
- Consumes: `AUTOFORMER_REPO_URL=https://github.com/thuml/Autoformer`, project root, existing ETTm1 CSV.
- Produces: `upstream/Autoformer/` checkout and `provenance/upstream.json` containing URL, commit SHA, clean status, `requirements.txt` SHA-256, CSV SHA-256, and capture timestamp.

- [ ] **Step 1: Create `fetch_upstream.sh` with a single safe setup path**

It must clone the official URL only when `upstream/Autoformer` is absent, fail if the directory exists but is not a Git worktree, then write the initial provenance JSON. It must not use a destructive refresh or reset command.

- [ ] **Step 2: Create `verify_upstream.sh`**

It must require an existing checkout, require `git diff --quiet`, require `git status --porcelain` to be empty, verify the ETTm1 CSV exists, and regenerate `provenance/upstream.json` with the exact current Git HEAD and SHA-256 hashes.

- [ ] **Step 3: Document the source boundary in `README.md` and `provenance/README.md`**

State that `upstream/Autoformer` is authors' code, that wrappers execute it without editing it, and that an unclean upstream makes a run invalid.

- [ ] **Step 4: Verify source capture**

Run: `bash reproduction/02_autoformer_official/scripts/fetch_upstream.sh`

Expected: a nested Git checkout exists, `provenance/upstream.json` has a non-empty full commit SHA and the CSV SHA-256, and `git -C reproduction/02_autoformer_official/upstream/Autoformer status --porcelain` prints nothing.

### Task 2: Build CPU environment and no-overwrite official launch wrappers

**Files:**
- Create: `reproduction/02_autoformer_official/requirements.compat.txt`
- Create: `reproduction/02_autoformer_official/scripts/create_cpu_env.sh`
- Create: `reproduction/02_autoformer_official/scripts/run_smoke_cpu.sh`
- Create: `reproduction/02_autoformer_official/scripts/run_official_ettm1_96_96_cpu.sh`

**Interfaces:**
- Consumes: verified upstream from Task 1; ETTm1 CSV; a shell argument `run-id` that is one safe directory name.
- Produces: an isolated `.venv/`; `runs/<run-id>/command.txt`, `environment.txt`, `system.txt`, `stdout.log`, `stderr.log`, `checkpoints/`, `results/`, and a `run.json` manifest.

- [ ] **Step 1: Define the compatible runtime dependencies**

Write `requirements.compat.txt` for the actual CPU environment. Pin the observed PyTorch CPU version and the versions needed by the unmodified authors' code; do not put the authors' obsolete Python 3.6/PyTorch 1.9 requirement in the executable environment file.

- [ ] **Step 2: Create `create_cpu_env.sh`**

Create `.venv` only if absent, install `requirements.compat.txt`, and write `provenance/environment.freeze.txt` using that venv's `python --version` and `pip freeze`. Never install packages into the existing DLinear `.venv`.

- [ ] **Step 3: Create `run_smoke_cpu.sh`**

Validate upstream cleanliness and the run id, make a unique run directory, then execute the author `run.py` from that run directory with `PYTHONPATH` pointing at the immutable checkout. Use the formal architecture/data arguments but override only `train_epochs=1` for an explicitly marked smoke run. Capture command, environment, CPU information, stdout/stderr, checkpoint and result files.

- [ ] **Step 4: Create `run_official_ettm1_96_96_cpu.sh`**

Use the author script's `ETTm1_96_96` model arguments exactly: `features=M`, `seq_len=96`, `label_len=48`, `pred_len=96`, two encoder layers, one decoder layer, factor three, and seven input/decoder/output channels. Do not run the author's multi-horizon script. Preserve the author defaults for learning rate, batch size, early stopping, and seed; record their resolved values in `run.json`.

- [ ] **Step 5: Verify smoke execution**

Run: `bash reproduction/02_autoformer_official/scripts/run_smoke_cpu.sh ettm1_l96_h96_smoke_seed2021`

Expected: exit status 0; nonempty checkpoint and `results/*/pred.npy`, `true.npy`, `metrics.npy`; `run.json` declares `kind: smoke`; the source checkout remains clean.

### Task 3: Add independently testable comparison and conversion library

**Files:**
- Create: `reproduction/02_autoformer_official/src/autoformer_reproduction/__init__.py`
- Create: `reproduction/02_autoformer_official/src/autoformer_reproduction/evaluation.py`
- Create: `reproduction/02_autoformer_official/tests/test_evaluation.py`

**Interfaces:**
- Produces: `TrainingScaler.from_csv(csv_path, train_end=34560)`, `load_official_result(run_dir)`, `load_baseline_result(npz_path)`, `assert_aligned_labels(...)`, `metric_summary(prediction, target, columns)`, and `evaluate_models(...)`.
- Input arrays have shape `[windows, prediction_length, channels]`; official arrays are normalized, baseline arrays are original scale.
- `evaluate_models` returns a JSON-serializable dict with `normalized`, `original_scale`, and `per_variable` metric sections for every model.

- [ ] **Step 1: Write failing tests for scaler and metric contracts**

Use synthetic 3-channel data to assert exact train-only mean/std conversion, known MSE/MAE/RMSE values, and per-variable metric names. Assert that a baseline original-scale target converts to the same normalized target as an official target.

- [ ] **Step 2: Run the targeted tests and confirm failure**

Run: `PYTHONPATH=reproduction/02_autoformer_official/src reproduction/02_autoformer_official/.venv/bin/python -m pytest reproduction/02_autoformer_official/tests/test_evaluation.py -q`

Expected: FAIL because `autoformer_reproduction.evaluation` does not exist.

- [ ] **Step 3: Implement `evaluation.py`**

Load `results/*/pred.npy` and `true.npy` from the Autoformer run and `prediction.npy`, `target.npy`, `columns.npy` from the existing baseline `.npz`. Reject missing/ambiguous official result directories, bad shapes, mismatched CSV columns, and label arrays that are not numerically equal after standardization. Use `np.mean`, not batch-averaged loss, for all metrics.

- [ ] **Step 4: Add failing alignment tests**

Add tests where sample count, horizon, channel count, and one target value differ. Each must raise a descriptive `ValueError` before any comparison metrics are returned.

- [ ] **Step 5: Implement alignment validation and verify tests pass**

Run: `PYTHONPATH=reproduction/02_autoformer_official/src reproduction/02_autoformer_official/.venv/bin/python -m pytest reproduction/02_autoformer_official/tests/test_evaluation.py -q`

Expected: PASS, including conversion and all mismatch rejections.

### Task 4: Produce the formal Autoformer CPU run and comparable metrics

**Files:**
- Create: `reproduction/02_autoformer_official/scripts/compare_ettm1_l96_h96.py`
- Create: `reproduction/02_autoformer_official/comparison/README.md`
- Modify: `reproduction/02_autoformer_official/README.md`

**Interfaces:**
- Consumes: a Task 2 formal run directory plus `01_dlinear_baseline/runs/ettm1_l96_h96_seasonal_naive_seed2026/predictions.npz` and `ettm1_l96_h96_dlinear_seed2026_weightedval/predictions.npz`.
- Produces: `comparison/ettm1_l96_h96_initial_metrics.json`, `comparison/ettm1_l96_h96_initial_metrics.md`, and `comparison/ettm1_l96_h96_initial_prediction.png`.

- [ ] **Step 1: Write a failing end-to-end fixture test for the comparison CLI**

Use temporary synthetic result directories and baseline archives. Assert the CLI refuses a label mismatch and writes JSON/Markdown only after aligned input succeeds.

- [ ] **Step 2: Implement the comparison CLI**

Call `evaluate_models`; write separate normalized/original-scale totals and per-variable tables. Annotate the output as an initial single-run comparison with Seasonal Naive and DLinear seed 2026 versus Autoformer seed 2021. Do not declare a general winner from this table.

- [ ] **Step 3: Implement the verified qualitative plot**

After label validation, select a deterministic test-window index and one named variable. Plot observed 96-point input, actual 96-point future, and all three original-scale predictions. Include the model/run ids in legend/title; do not use this plot as a metric calculation input.

- [ ] **Step 4: Run the formal author configuration**

Run: `bash reproduction/02_autoformer_official/scripts/run_official_ettm1_96_96_cpu.sh ettm1_l96_h96_autoformer_seed2021_cpu`

Expected: exit status 0, immutable source verification passes, and run artifacts include the author result arrays and resolved command/configuration.

- [ ] **Step 5: Run end-to-end comparison and verify generated artifacts**

Run: `PYTHONPATH=reproduction/02_autoformer_official/src reproduction/02_autoformer_official/.venv/bin/python reproduction/02_autoformer_official/scripts/compare_ettm1_l96_h96.py --autoformer-run reproduction/02_autoformer_official/runs/ettm1_l96_h96_autoformer_seed2021_cpu --seasonal-naive reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_seasonal_naive_seed2026/predictions.npz --dlinear reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_dlinear_seed2026_weightedval/predictions.npz --csv DataSet/ETTm1/ETTm1.csv --output-dir reproduction/02_autoformer_official/comparison`

Expected: JSON, Markdown table, and PNG exist; the Markdown explicitly says labels passed alignment and lists scale/seed limitations.

### Task 5: Verify all deliverables and write the learning conclusion

**Files:**
- Create: `reproduction/02_autoformer_official/comparison/report.md`
- Modify: `reproduction/02_autoformer_official/README.md`

**Interfaces:**
- Consumes: formal run metadata and Task 4 comparison artifacts.
- Produces: an exact reproduction command sequence and a concise, evidence-bounded explanation of the observed difference between the three models.

- [ ] **Step 1: Run the complete test suite**

Run: `PYTHONPATH=reproduction/02_autoformer_official/src reproduction/02_autoformer_official/.venv/bin/python -m pytest reproduction/02_autoformer_official/tests -q`

Expected: PASS.

- [ ] **Step 2: Re-verify immutable source and result provenance**

Run: `bash reproduction/02_autoformer_official/scripts/verify_upstream.sh`

Expected: clean source checkout, stable commit SHA, CSV hash present, and no untracked artifacts inside `upstream/Autoformer`.

- [ ] **Step 3: Write `comparison/report.md`**

State the exact dataset, split, scale, horizon, model source commit, CPU environment, metrics, and observed trend/seasonal behavior. Distinguish measured findings from hypotheses, and state that one run per learned model is insufficient to establish a general conclusion.

- [ ] **Step 4: Update the README with clean rerun instructions**

Document source capture, environment creation, smoke run, formal run, comparison, expected artifacts, and the explicit no-overwrite rule.

- [ ] **Step 5: Final verification**

Run: `rg -n "TODO|TBD" reproduction/02_autoformer_official docs/superpowers/specs/2026-10-03-autoformer-official-reproduction-design.md docs/superpowers/plans/2026-10-03-autoformer-official-reproduction.md`

Expected: exit status 1 and no output, meaning no unresolved placeholders in the new reproduction deliverables.
