# PatchTST 官方实现对齐复现 - 实施计划

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用固定提交的官方 PatchTST 监督学习代码，在 ETTm1 `96 → 96` 上训练并导出可与 Seasonal Naive、DLinear、Autoformer 标签逐元素对齐的结果。

**Architecture:** 把官方仓库原样放在 ignored 的 `upstream/`；项目包装层负责环境、运行、预测/标签导出和公平比较。共享的评估核心显式记录尺度并拒绝未对齐标签，模型特定模块只负责读取自己的产物。

**Tech Stack:** Python 3.12、PyTorch 2.5.1（L40 CUDA 环境）、NumPy、Pandas、scikit-learn、pytest、Bash、官方 PatchTST 提交 `204c21e`。

---

### Task 1: 建立不可变上游边界与可验证环境

**Files:**
- Create: `reproduction/03_patchtst_official/scripts/fetch_upstream.sh`
- Create: `reproduction/03_patchtst_official/scripts/verify_upstream.sh`
- Create: `reproduction/03_patchtst_official/provenance/upstream.json`
- Create: `reproduction/03_patchtst_official/requirements.common.txt`
- Create: `reproduction/03_patchtst_official/scripts/create_cpu_env.sh`
- Create: `reproduction/03_patchtst_official/scripts/create_gpu_env.sh`
- Create: `reproduction/03_patchtst_official/tests/test_source_boundary.py`

**Step 1: Write the failing source-boundary tests.**

Test that the fetch script pins `204c21efe0b39603ad6e2ca640ef5896646ab1a9`, `verify_upstream.sh` rejects a dirty upstream tree, and project-side code cannot be placed under `upstream/PatchTST/`.

**Step 2: Run the test to prove it fails.**

Run: `pytest -q reproduction/03_patchtst_official/tests/test_source_boundary.py`

Expected: failures because scripts and provenance are absent.

**Step 3: Implement the fetch/verify/environment scripts.**

`fetch_upstream.sh` clones or fetches the exact commit without copying project wrappers into it. `verify_upstream.sh` checks URL, HEAD commit and `git diff --quiet`. CPU/GPU helpers create `.venv`, install common packages, and let the caller choose a PyTorch CUDA index such as `cu121`.

**Step 4: Re-run the test.**

Run the same pytest command.

Expected: PASS.

### Task 2: Define the result archive and reusable scale-safe evaluation contract

**Files:**
- Create: `reproduction/common/ltsf_evaluation/__init__.py`
- Create: `reproduction/common/ltsf_evaluation/core.py`
- Create: `reproduction/common/tests/test_ltsf_evaluation.py`
- Modify: `reproduction/02_autoformer_official/src/autoformer_reproduction/evaluation.py`
- Modify: `reproduction/02_autoformer_official/tests/test_evaluation.py`

**Step 1: Write failing tests for a model-neutral evaluator.**

Cover `ForecastResult`, train-only ETTm1 scaling, normalized result archives, raw-scale metrics, a one-value label mismatch, wrong channel order, and a model name supplied as `"PatchTST"` instead of a hard-coded `"Autoformer"`.

**Step 2: Run the new test module.**

Run: `pytest -q reproduction/common/tests/test_ltsf_evaluation.py`

Expected: import failure.

**Step 3: Implement the shared core and keep the Autoformer public helpers compatible.**

Use the existing audited semantics: baseline archives are normalized; raw metrics are obtained only after applying a scaler fitted to rows `[0,34560)`. Keep `load_official_result()` in the Autoformer adapter so existing scripts do not change their CLI.

**Step 4: Run regression tests.**

Run: `PYTHONPATH=reproduction/common:reproduction/02_autoformer_official/src pytest -q reproduction/common/tests reproduction/02_autoformer_official/tests/test_evaluation.py reproduction/02_autoformer_official/tests/test_comparison_cli.py`

Expected: all pass.

### Task 3: Add an official-model exporter without changing upstream source

**Files:**
- Create: `reproduction/03_patchtst_official/src/patchtst_reproduction/__init__.py`
- Create: `reproduction/03_patchtst_official/src/patchtst_reproduction/export.py`
- Create: `reproduction/03_patchtst_official/tests/test_export_contract.py`
- Create: `reproduction/03_patchtst_official/scripts/export_ettm1_predictions.py`

**Step 1: Write failing exporter-contract tests.**

Test that the exporter accepts only `[windows, horizon, channels]`, emits a compressed archive with `prediction`, `target`, `columns`, and a declared normalized scale, and rejects an absent/ambiguous checkpoint or result directory.

**Step 2: Run the test to confirm failure.**

Run: `PYTHONPATH=reproduction/common:reproduction/03_patchtst_official/src pytest -q reproduction/03_patchtst_official/tests/test_export_contract.py`

Expected: import failure.

**Step 3: Implement export against the official test pathway.**

Use the official test dataloader, checkpoint and `outputs[:, -pred_len:, f_dim:]` slice. Collect both model output and `batch_y` target in the same loop. Do not patch `exp/exp_main.py`, whose checked-in implementation omits `true.npy`.

**Step 4: Run exporter tests.**

Run the same test command.

Expected: PASS.

### Task 4: Add deterministic launch wrappers and a CPU shape smoke test

**Files:**
- Create: `reproduction/03_patchtst_official/scripts/run_common.sh`
- Create: `reproduction/03_patchtst_official/scripts/run_smoke_cpu.sh`
- Create: `reproduction/03_patchtst_official/scripts/run_official_ettm1_96_96_gpu.sh`
- Create: `reproduction/03_patchtst_official/tests/test_launch_wrappers.py`
- Create: `reproduction/03_patchtst_official/tests/test_patch_shapes.py`

**Step 1: Write failing wrapper and shape tests.**

Require `seq_len=96`, `pred_len=96`, `features=M`, `enc_in=7`, `patch_len=16`, `stride=8`, author seed 2021, run-name isolation, and a padded patch count of 12. Verify that `B×7×96` becomes `B×7×16×12` before the encoder and `B×96×7` after the forecasting head.

**Step 2: Run tests and observe failure.**

Run: `PYTHONPATH=reproduction/03_patchtst_official/src pytest -q reproduction/03_patchtst_official/tests/test_launch_wrappers.py reproduction/03_patchtst_official/tests/test_patch_shapes.py`

Expected: failures until scripts/modules exist.

**Step 3: Implement wrappers.**

`run_common.sh` resolves project paths and writes each run to a unique ignored directory. The smoke script imports the official model and checks the shape only; the GPU wrapper invokes official supervised training and then the project exporter.

**Step 4: Run the wrapper/shape suite.**

Expected: PASS without downloading ETTm1 or starting a full training run.

### Task 5: Implement label-verified PatchTST comparison and research-facing documentation

**Files:**
- Create: `reproduction/03_patchtst_official/src/patchtst_reproduction/evaluation.py`
- Create: `reproduction/03_patchtst_official/scripts/compare_ettm1_l96_h96.py`
- Create: `reproduction/03_patchtst_official/tests/test_comparison_cli.py`
- Create: `reproduction/03_patchtst_official/README.md`
- Modify: `related_work/04_patchtst_2023/README.md`

**Step 1: Write a failing comparison test.**

Create synthetic normalized PatchTST, Seasonal Naive and DLinear archives. Require one shared target tensor and reject a shifted target, changed column order, or mismatched horizon. Assert generated Markdown names PatchTST and reports normalized/raw global and per-variable metrics.

**Step 2: Run the test to confirm failure.**

Run: `PYTHONPATH=reproduction/common:reproduction/03_patchtst_official/src pytest -q reproduction/03_patchtst_official/tests/test_comparison_cli.py`

Expected: import failure.

**Step 3: Implement the CLI and README.**

The CLI must accept three existing archive paths and `DataSet/ETTm1/ETTm1.csv`, produce an ignored Markdown/figure directory, and fail before metrics if labels differ. The README explains official `L=336` paper settings versus this first fair `L=96` project setting.

**Step 4: Re-run the test.**

Expected: PASS.

### Task 6: Verify locally, run on the L40, and record evidence without overstating it

**Files:**
- Modify: `experiment/results.md`
- Modify: `handoff.md`
- Modify: `brainstorm.md`

**Step 1: Run all automated checks.**

Run: `PYTHONPATH=reproduction/common:reproduction/02_autoformer_official/src:reproduction/03_patchtst_official/src pytest -q reproduction/common/tests reproduction/02_autoformer_official/tests reproduction/03_patchtst_official/tests`

Expected: all tests pass.

**Step 2: Perform the CPU smoke run.**

Run: `bash reproduction/03_patchtst_official/scripts/run_smoke_cpu.sh`

Expected: printed shape evidence showing 12 padded patches and a `B×96×7` forecast.

**Step 3: Run the cloud GPU command.**

On the L40 machine, set up the GPU environment, fetch the pinned upstream source, then run the GPU wrapper with a fresh run name. Preserve old runs; never overwrite an existing run directory.

**Step 4: Compare with the existing aligned baseline archives.**

Pass `...seasonal_naive_cloud20261003_r2/predictions.npz` and `...dlinear_cloud20261003_r2/predictions.npz` to the comparison script. Accept metrics only when target alignment reports success.

**Step 5: Record the result and limitations.**

Append seed, official commit, `L/H/P/S/N`, GPU, epoch/best-checkpoint evidence, parameters, normalized/raw metrics and label alignment to `experiment/results.md`. Update `handoff.md` with exact paths and add the next hypothesis: compare the same models under `L=336`, rather than treating `L=96` as a paper-table replication.

**Step 6: Commit and publish only after verification.**

Run `git status`, review the patch, commit tracked source/docs only (never data, `.venv`, upstream source, checkpoints or run artifacts), and push the public repository.
