# ETTm1：Seasonal Naive 与 DLinear 最小复现

这是时序预测学习项目的第一条可运行基线。它固定在 ETTm1 的多变量设置上：输入最近 `L=96` 个 15 分钟采样点（一日），一次预测未来 `H=96` 个点。目标是先建立可信的实验协议，而不是追求论文排行榜数字。

包含两个模型：

- `Seasonal Naive`：直接复制上一日的同一时刻，作为必须击败的简单基线。
- `DLinear`：将序列分为平滑趋势和残差季节项，再分别用线性层从 96 点映射到 96 点；首轮使用共享通道参数、移动平均核宽 25。

## 环境

在本目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
```

测试覆盖训练段标准化、时间边界、周期复制、DLinear 形状/分解、反标准化指标、CPU 训练产物与 CLI 的相对路径解析。

## 数据与协议

原始 CSV 应位于 `DataSet/ETTm1/ETTm1.csv`（相对项目根目录），而不是运行产物目录。该文件来自 [ETDataset 作者仓库](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTm1.csv)；当前文件的 SHA-256、列与切分细节记录在项目根目录的 [dataset.md](../../dataset.md)。

训练标签范围是行 `[96, 34560)`，验证标签范围是 `[34560, 46080)`，测试标签范围是 `[46080, 57600)`。验证和测试样本可以回看此前历史，但标签绝不会跨出自己的目标区间。均值和标准差只用原始训练行 `[0, 34560)` 拟合，所有 7 个数值变量均为预测目标。

## 运行

在本目录执行。每个 `--run-name` 只能使用一次；若同名目录已存在，程序会拒绝覆盖。

```bash
PYTHONPATH=src .venv/bin/python -m ltsf_baseline.cli \
  --config configs/ettm1_l96_h96_seasonal_naive.json \
  --run-name ettm1_l96_h96_seasonal_naive_seed2026_rerun1 \
  --project-root .

PYTHONPATH=src .venv/bin/python -m ltsf_baseline.cli \
  --config configs/ettm1_l96_h96.json \
  --run-name ettm1_l96_h96_dlinear_seed2026_rerun1 \
  --project-root .
```

每次运行会在 `runs/<run-name>/` 写入：固定配置 `config.json`、训练/验证历史 `history.json`、原尺度指标 `metrics.json`、预测与标签 `predictions.npz`，以及 DLinear 的最佳验证 checkpoint `best.pt`。

## 指标如何读

预测和标签都会先用训练 scaler 反标准化，再计算：

- **MSE**：平均平方误差，对大误差更敏感；
- **MAE**：平均绝对误差，直接表示平均偏差大小；
- **RMSE**：`sqrt(MSE)`，单位与原变量一致。

`metrics.json` 还保存每个变量的 MAE。`runs/` 中的数值只是本机、本数据副本和本配置下的可追溯证据；它们不是论文数字，也不能直接与不同切分、不同特征设置或不同实现的论文表格比较。
