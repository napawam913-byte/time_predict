# ETTm1 DLinear 最小复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ETTm1 上建立一条 CPU 可运行、无时间泄漏、可追溯的 Seasonal Naive 与 DLinear 比较管线。

**Architecture:** 复现代码独立位于 `reproduction/01_dlinear_baseline/`。数据模块负责固定时间边界、仅训练段拟合的标准化和滑动窗口；模型和指标模块没有文件 I/O；runner 负责训练、选择最佳验证 checkpoint、反标准化评估以及写入不可覆盖的运行产物。

**Tech Stack:** Python 3.10+, PyTorch CPU, NumPy, pandas, pytest。

**Spec:** `docs/superpowers/specs/2026-09-25-ettm1-dlinear-baseline-design.md`

## Global Constraints

- 代码在 `reproduction/01_dlinear_baseline/`，ETTm1 CSV 只在 `DataSet/ETTm1/ETTm1.csv`。
- 首轮固定多变量预测、`L=96`、`H=96`、种子 `2026`、Seasonal Naive 与共享通道 DLinear。
- 训练、验证、测试按时间排列；训练目标结束点为 34,560，验证结束点为 46,080，测试结束点为 57,600。
- 标准化统计量只使用训练目标区间；指标必须在反标准化的原始尺度计算。
- DLinear 使用核宽 25 的端点复制移动平均、Seasonal/Trend 两条 `Linear(L,H)` 分支。
- 运行目录不可覆盖；论文数字与本地运行结果严格分开。
- 目标项目不是 Git 仓库；没有用户明确指令时不得初始化仓库或执行 commit。

## Review Focus

- 验证/测试样本可读取此前历史，但标签的开始与结束都必须留在自己的目标区间内。
- 测试集极值不能改变 scaler 的均值和标准差，也不能参与训练或早停选择。
- 预测与真值必须以同一个训练 scaler 反标准化后才计算指标。
- `Seasonal Naive` 在 `L=H=96` 时复制整段上一日历史，不进行递归生成。
- 已存在运行名不得静默覆盖 checkpoint、预测数组或指标文件。

## File Structure

```text
reproduction/01_dlinear_baseline/
├── README.md
├── requirements.txt
├── configs/ettm1_l96_h96.json
├── src/ltsf_baseline/
│   ├── __init__.py
│   ├── data.py
│   ├── baselines.py
│   ├── dlinear.py
│   ├── metrics.py
│   ├── runner.py
│   └── cli.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_baselines.py
│   ├── test_dlinear.py
│   ├── test_metrics.py
│   └── test_runner.py
└── runs/.gitkeep
```

## Task 1: 可导入的复现包与固定配置

**Files:**
- Create: `reproduction/01_dlinear_baseline/requirements.txt`
- Create: `reproduction/01_dlinear_baseline/configs/ettm1_l96_h96.json`
- Create: `reproduction/01_dlinear_baseline/src/ltsf_baseline/__init__.py`
- Create: `reproduction/01_dlinear_baseline/tests/test_config_contract.py`
- Create: `reproduction/01_dlinear_baseline/runs/.gitkeep`

**Interfaces:**
- Consumes: no production module.
- Produces: an importable `ltsf_baseline` package and JSON configuration read by `cli.py` in Task 4.

- [x] **Step 1: Write the failing configuration-contract test**

```python
import json
from pathlib import Path


def test_ettm1_first_run_config_is_leakage_safe():
    config_path = Path(__file__).parents[1] / "configs" / "ettm1_l96_h96.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["dataset_path"] == "../../../DataSet/ETTm1/ETTm1.csv"
    assert config["input_length"] == 96
    assert config["prediction_length"] == 96
    assert config["seed"] == 2026
    assert config["model"]["name"] == "dlinear"
    assert config["model"]["individual"] is False
    assert config["training"]["batch_size"] == 32
    assert config["training"]["learning_rate"] == 1e-3
    assert config["training"]["max_epochs"] == 20
    assert config["training"]["patience"] == 3
```

- [x] **Step 2: Run the test to verify that it fails because the file is absent**

Run:

```bash
cd /home/xc_ubantu/homelab/projects/time_predict/reproduction/01_dlinear_baseline
PYTHONPATH=src pytest tests/test_config_contract.py -v
```

Expected: `FAIL` with `FileNotFoundError` for `ettm1_l96_h96.json`.

- [x] **Step 3: Create only the package marker, requirements and fixed JSON configuration**

Use `requirements.txt` with exact minimum dependencies:

```text
numpy>=1.26,<3
pandas>=2.1,<3
torch>=2.2,<3
pytest>=8,<9
```

Use `ettm1_l96_h96.json`:

```json
{
  "dataset_path": "../../../DataSet/ETTm1/ETTm1.csv",
  "input_length": 96,
  "prediction_length": 96,
  "seed": 2026,
  "model": {"name": "dlinear", "individual": false, "moving_average_window": 25},
  "training": {"batch_size": 32, "learning_rate": 0.001, "max_epochs": 20, "patience": 3}
}
```

Create empty `__init__.py` and `runs/.gitkeep`. Do not add model code in this task.

- [x] **Step 4: Re-run the configuration-contract test**

Run the Step 2 command.

Expected: `PASS`.

- [x] **Step 5: Record the no-commit state**

Run:

```bash
git -C /home/xc_ubantu/homelab/projects/time_predict rev-parse --is-inside-work-tree
```

Expected: non-zero because this project is not a Git repository. Preserve that state; do not initialize a repository.

## Task 2: ETTm1 时间切分、标准化与滑动窗口

**Files:**
- Create: `reproduction/01_dlinear_baseline/src/ltsf_baseline/data.py`
- Create: `reproduction/01_dlinear_baseline/tests/conftest.py`
- Create: `reproduction/01_dlinear_baseline/tests/test_data.py`

**Interfaces:**
- Consumes: `numpy.ndarray` shaped `[time, channels]` and a CSV with a strictly increasing `date` column.
- Produces: `SplitBoundaries(train_end, validation_end, test_end)`, `Standardizer.fit/transform/inverse_transform`, `WindowDataset`, and `prepare_ettm1(csv_path, input_length, prediction_length, boundaries=None)`.

- [x] **Step 1: Write tests for training-only standardization and legal labels**

```python
import numpy as np
from ltsf_baseline.data import SplitBoundaries, prepare_ettm1


def test_scaler_uses_only_training_rows(synthetic_ettm1_csv):
    boundaries = SplitBoundaries(train_end=6, validation_end=9, test_end=12)
    prepared = prepare_ettm1(synthetic_ettm1_csv, 2, 2, boundaries)

    np.testing.assert_allclose(prepared.scaler.mean_, [2.5, 25.0])
    np.testing.assert_allclose(prepared.scaler.scale_, [np.sqrt(35 / 12), np.sqrt(3500 / 12)])


def test_validation_and_test_labels_stay_inside_their_target_ranges(synthetic_ettm1_csv):
    boundaries = SplitBoundaries(train_end=6, validation_end=9, test_end=12)
    prepared = prepare_ettm1(synthetic_ettm1_csv, 2, 2, boundaries)

    assert prepared.validation.target_starts.tolist() == [6, 7]
    assert prepared.test.target_starts.tolist() == [9, 10]
    x, y = prepared.test[0]
    assert x.shape == (2, 2)
    assert y.shape == (2, 2)


def test_rejects_non_monotonic_timestamps(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,a\\n2024-01-01 00:15,1\\n2024-01-01 00:00,2\\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_ettm1(path, 1, 1, SplitBoundaries(1, 1, 2))
```

`conftest.py` must create a 12-row, two-channel, 15-minute CSV where channels are `0..11` and `0,10,..110`; it must not use production helpers to derive expected values.

- [x] **Step 2: Run the data tests and verify the expected missing-module failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_data.py -v
```

Expected: `FAIL` with `ModuleNotFoundError: No module named 'ltsf_baseline.data'`.

- [x] **Step 3: Implement the smallest leakage-safe data API**

Implement these concrete behaviors:

```python
@dataclass(frozen=True)
class SplitBoundaries:
    train_end: int
    validation_end: int
    test_end: int


class Standardizer:
    def fit(self, values: np.ndarray) -> "Standardizer":
        """Store featurewise population mean and nonzero standard deviation."""
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return (values - mean_) / scale_."""
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Return values * scale_ + mean_."""


class WindowDataset(Dataset):
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized history [L,C] and following target [H,C]."""


def prepare_ettm1(csv_path: Path, input_length: int, prediction_length: int,
                  boundaries: SplitBoundaries | None = None) -> PreparedETTm1:
    """Validate CSV, fit train-only scaler, and return chronological windows."""
```

Use fixed default boundaries `(34560, 46080, 57600)`. Fit `Standardizer` only on raw rows `[:train_end]`. A `WindowDataset` with target interval `[target_begin, target_end)` must enumerate target start `t` from `target_begin` through `target_end - prediction_length`; emit `x=values[t-L:t]` and `y=values[t:t+H]`. Use train target interval `[L, train_end)`, validation `[train_end, validation_end)`, and test `[validation_end, test_end)`.

Reject missing `date`, no numeric value columns, unparseable/duplicate/non-increasing timestamps, NaN/Inf numeric values, invalid boundary ordering, and split ranges unable to generate one complete window. Replace zero standard deviations with `1.0` so constant channels are valid.

- [x] **Step 4: Re-run all data tests**

Run the Step 2 command.

Expected: `PASS`; especially, test values `1000` and `1100` in later splits must not alter the hand-checked train mean or scale.

- [x] **Step 5: Run a focused test for zero-variance stability**

Add and run:

```python
def test_constant_training_channel_scales_without_nan(synthetic_ettm1_csv):
    frame = pd.read_csv(synthetic_ettm1_csv)
    frame["b"] = 5.0
    frame.to_csv(synthetic_ettm1_csv, index=False)
    prepared = prepare_ettm1(synthetic_ettm1_csv, 2, 2, SplitBoundaries(6, 9, 12))
    assert np.isfinite(prepared.train.values).all()
```

Run:

```bash
PYTHONPATH=src pytest tests/test_data.py::test_constant_training_channel_scales_without_nan -v
```

Expected: `PASS`.

## Task 3: 两个预测基线与原始尺度指标

**Files:**
- Create: `reproduction/01_dlinear_baseline/src/ltsf_baseline/baselines.py`
- Create: `reproduction/01_dlinear_baseline/src/ltsf_baseline/dlinear.py`
- Create: `reproduction/01_dlinear_baseline/src/ltsf_baseline/metrics.py`
- Create: `reproduction/01_dlinear_baseline/tests/test_baselines.py`
- Create: `reproduction/01_dlinear_baseline/tests/test_dlinear.py`
- Create: `reproduction/01_dlinear_baseline/tests/test_metrics.py`

**Interfaces:**
- Consumes: `torch.Tensor` in `[B,L,C]` normalized model space; predictions and labels in that same shape; `Standardizer` from Task 2 for reporting.
- Produces: `seasonal_naive(x, period)`, `MovingAverage`, `DLinear`, and `original_scale_metrics(prediction, target, scaler, column_names)`.

- [x] **Step 1: Write failing tests for Seasonal Naive, decomposition and DLinear shape**

```python
def test_seasonal_naive_repeats_the_last_complete_period():
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    prediction = seasonal_naive(x, prediction_length=4, period=4)
    assert torch.equal(prediction, x)


def test_decomposition_recombines_exactly():
    x = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)
    seasonal, trend = decompose(x, kernel_size=5)
    assert torch.allclose(seasonal + trend, x)


def test_dlinear_maps_batch_time_channel_to_requested_horizon():
    model = DLinear(input_length=8, prediction_length=3, channels=2, individual=False, kernel_size=25)
    assert model(torch.zeros(4, 8, 2)).shape == (4, 3, 2)
```

- [x] **Step 2: Run the model tests and verify missing import failures**

Run:

```bash
PYTHONPATH=src pytest tests/test_baselines.py tests/test_dlinear.py -v
```

Expected: `FAIL` because `seasonal_naive`, `decompose`, and `DLinear` do not exist yet.

- [x] **Step 3: Implement the reference-compatible model behavior**

Implement:

```python
def seasonal_naive(x: torch.Tensor, prediction_length: int, period: int) -> torch.Tensor:
    if x.ndim != 3 or period > x.shape[1] or prediction_length % period != 0:
        raise ValueError("prediction length must be a whole available seasonal period")
    return x[:, -period:, :].repeat(1, prediction_length // period, 1)
```

For DLinear, replicate the first and last time position `(kernel_size-1)//2` times, apply `AvgPool1d(kernel_size, stride=1)`, calculate `seasonal=x-trend`, transpose to `[B,C,L]`, apply two shared `nn.Linear(input_length, prediction_length)` layers, add, and transpose back. Validate input rank, time length and channel count with informative `ValueError`s.

- [x] **Step 4: Re-run model tests**

Run the Step 2 command.

Expected: `PASS`.

- [x] **Step 5: Write a failing original-scale metric test**

```python
def test_metrics_are_calculated_after_inverse_transform():
    scaler = fitted_scaler_with_mean_10_and_scale_2()
    prediction = np.array([[[0.0], [1.0]]])
    target = np.array([[[1.0], [0.0]]])

    metrics = original_scale_metrics(prediction, target, scaler, ["OT"])

    assert metrics["mse"] == 4.0
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert metrics["per_variable_mae"] == {"OT": 2.0}
```

- [x] **Step 6: Run the metric test to verify the missing-module failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_metrics.py -v
```

Expected: `FAIL` because `original_scale_metrics` does not exist.

- [x] **Step 7: Implement original-scale metrics and verify all Task 3 tests**

Use `scaler.inverse_transform` on both arrays, then calculate global mean squared error, absolute error, square-root MSE, and per-channel MAE. Require equal three-dimensional shapes and matching column count.

Run:

```bash
PYTHONPATH=src pytest tests/test_baselines.py tests/test_dlinear.py tests/test_metrics.py -v
```

Expected: all tests `PASS`.

## Task 4: CPU 训练、评估、不可覆盖运行产物与 CLI

**Files:**
- Create: `reproduction/01_dlinear_baseline/src/ltsf_baseline/runner.py`
- Create: `reproduction/01_dlinear_baseline/src/ltsf_baseline/cli.py`
- Create: `reproduction/01_dlinear_baseline/tests/test_runner.py`

**Interfaces:**
- Consumes: `PreparedETTm1`, `DLinear`, `seasonal_naive`, JSON configuration and `runs/` directory.
- Produces: `run_experiment(config, project_root, run_name) -> dict`, a run directory with `config.json`, `history.json`, `metrics.json`, `predictions.npz`, and for DLinear `best.pt`; CLI process exit code `0` on success.

- [x] **Step 1: Write a failing CPU smoke test for run artifacts and checkpoint selection**

```python
def test_dlinear_cpu_run_writes_reproducible_artifacts(tiny_csv, tmp_path):
    config = tiny_config(dataset_path=str(tiny_csv), input_length=8, prediction_length=2,
                         max_epochs=2, patience=1, batch_size=4)

    result = run_experiment(config, project_root=tmp_path, run_name="tiny_dlinear")

    run_dir = tmp_path / "runs" / "tiny_dlinear"
    assert set(result) >= {"mse", "mae", "rmse", "per_variable_mae"}
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "history.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.npz").is_file()
    assert (run_dir / "best.pt").is_file()
    assert torch.load(run_dir / "best.pt", weights_only=True)["model_state"]


def test_existing_run_name_is_rejected(tiny_csv, tmp_path):
    config = tiny_config(dataset_path=str(tiny_csv), input_length=8, prediction_length=2)
    run_experiment(config, project_root=tmp_path, run_name="already_exists")
    with pytest.raises(FileExistsError, match="already_exists"):
        run_experiment(config, project_root=tmp_path, run_name="already_exists")
```

The fixture must create enough monotonically increasing rows for boundaries `(80, 110, 140)` and a two-channel periodic-but-noisy signal; it must not rely on real ETTm1 downloading.

- [x] **Step 2: Run the smoke tests and verify the expected missing-module failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_runner.py -v
```

Expected: `FAIL` with `ModuleNotFoundError: No module named 'ltsf_baseline.runner'`.

- [x] **Step 3: Implement training and evaluation in the smallest form**

Implement `run_experiment` with the following order:

1. validate config and set `random`, `numpy`, and `torch` seeds;
2. call `prepare_ettm1` with configured lengths and optional test boundaries;
3. make CPU DataLoaders: train shuffled, validation/test not shuffled;
4. for `seasonal_naive`, predict test batches without an optimizer and write metrics/predictions;
5. for `dlinear`, train `DLinear` with Adam and normalized-space `MSELoss`, evaluate validation MSE every epoch, retain the best state, and stop after `patience` non-improving epochs;
6. reload the best state before collecting all test predictions;
7. calculate metrics only through `original_scale_metrics` and atomically create the run directory before writing artifacts.

Store JSON with `indent=2`, prediction archive keys `prediction`, `target`, and `columns`, and checkpoint keys `model_state`, `epoch`, and `validation_mse`. If CSV is missing, a DataLoader is empty, a checkpoint is absent, or config name is unsupported, raise a named Python exception with the offending path/value.

`cli.py` must accept exactly `--config`, `--run-name`, and `--project-root`, resolve relative dataset paths from the config directory, call `run_experiment`, and print one JSON line containing MSE, MAE and RMSE.

- [x] **Step 4: Re-run runner tests**

Run the Step 2 command.

Expected: `PASS` on CPU in under 30 seconds.

- [x] **Step 5: Add and run a Seasonal Naive integration test**

```python
def test_seasonal_naive_run_has_metrics_but_no_checkpoint(tiny_csv, tmp_path):
    config = tiny_config(dataset_path=str(tiny_csv), model_name="seasonal_naive",
                         input_length=8, prediction_length=8, period=8)
    run_experiment(config, project_root=tmp_path, run_name="tiny_naive")
    run_dir = tmp_path / "runs" / "tiny_naive"
    assert (run_dir / "metrics.json").is_file()
    assert not (run_dir / "best.pt").exists()
```

Run:

```bash
PYTHONPATH=src pytest tests/test_runner.py -v
```

Expected: `PASS`.

## Task 5: 可操作文档、真实数据登记与端到端验证

**Files:**
- Create: `reproduction/01_dlinear_baseline/README.md`
- Modify: `dataset.md`
- Modify: `handoff.md`
- Modify: `experiment/results.md` only after an actual successful run

**Interfaces:**
- Consumes: working package and an official ETTm1 CSV placed by the user or downloader at `DataSet/ETTm1/ETTm1.csv`.
- Produces: a beginner-readable CPU command, provenance record and an append-only local result entry.

- [x] **Step 1: Verify executable documentation contracts instead of literal prose fragments**

The executable contracts are already covered by the CLI path-resolution test and the runner integration tests. Verify `python -m ltsf_baseline.cli --help` manually; do not add a brittle test that only searches README text.

- [x] **Step 2: Run executable-contract verification**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m ltsf_baseline.cli --help
```

Expected: help lists exactly `--config`, `--run-name`, and `--project-root`.

- [x] **Step 3: Write README and project records**

README must explain:

- the two model roles and the fixed `L=H=96` first run;
- Python virtual environment creation and `pip install -r requirements.txt`;
- source-data location and a reminder that it is not versioned as experiment output;
- exact test command `PYTHONPATH=src pytest -q`;
- exact run command:

```bash
PYTHONPATH=src python -m ltsf_baseline.cli \
  --config configs/ettm1_l96_h96.json \
  --run-name ettm1_l96_h96_dlinear_seed2026 \
  --project-root .
```

- the definition of MSE, MAE, RMSE and the fact that results are inverse-transformed;
- that a result in `runs/` is local evidence and never a paper result.

Update `dataset.md` only after obtaining the CSV: add its canonical URL, SHA-256, 15-minute frequency, columns, missing-value check, exact row boundaries and the fact that all variables are forecast targets. Update `handoff.md` to state whether data download, tests, Seasonal Naive, DLinear, and real ETTm1 runs have completed. Do not insert unrun numbers into `experiment/results.md`.

- [x] **Step 4: Re-run the full suite**

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected: all unit, integration, documentation and CPU smoke tests `PASS`.

- [x] **Step 5: Obtain and verify official ETTm1 data, then run both baselines**

After the user grants network/dependency permission, obtain ETTm1 from the official benchmark location. Before any model run, verify:

```bash
sha256sum DataSet/ETTm1/ETTm1.csv
python -c "import pandas as pd; d=pd.read_csv('DataSet/ETTm1/ETTm1.csv'); print(d.shape); print(d.columns.tolist()); print(d['date'].iloc[[0,-1]].tolist())"
```

Run Seasonal Naive with a different name, then DLinear with the README command. Append each actual successful run's metrics, split, configuration and artifact path to `experiment/results.md`; record failures as failures rather than replacing them.

## Plan Self-Review

- Spec coverage: Task 1 fixes the first-run contract; Task 2 enforces chronological splits and training-only normalization; Task 3 implements both models and original-scale metrics; Task 4 handles training, checkpointing and run immutability; Task 5 makes the workflow operable and records real evidence.
- Placeholder scan: no unfinished implementation labels are left in the plan; each task names files, interfaces, tests and commands.
- Type consistency: every model input is `[B,L,C]`; `prepare_ettm1` yields data for `run_experiment`; all reporting flows through `original_scale_metrics`.
- Review focus: Task 2 tests split and scaler leakage; Task 3 tests daily copying and scale-sensitive metrics; Task 4 tests run-directory collision and checkpoint reload.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-25-ettm1-dlinear-baseline.md`. Please review the plan. Which execution approach would you prefer?

- **Subagent-driven** — separate implementer/reviewer turns for each task. Most thorough, but requires more coordination.
- **Native** — I implement every task in this session, then run a final review. Faster and appropriate here because the tasks form one small, tightly coupled learning pipeline.

I recommend **Native**. Does the plan capture what you want, and which approach should we use?
