# PatchTST 官方实现对齐复现设计

## 目标与边界

在现有通用多变量长预测项目中，用官方 `yuqinie98/PatchTST` 的监督学习实现复现 PatchTST，并得到与已有 Seasonal Naive、DLinear、Autoformer 同一 ETTm1 测试窗口、同一标签、同一原始尺度指标下可审计的结果。

首轮是项目协议实验：ETTm1、`features=M`、`L=96`、`H=96`、7 个预测变量、`P=16`、`S=8`、作者默认随机种子 2021。它检验 patch token 与 channel independence 在既有任务上的表现，不声称直接复现论文表格。第二轮才使用论文的 `L=336` 配置，并为所有对照模型重跑同一窗口后单独报告。

不纳入首轮：掩码自监督预训练、迁移学习、Weather/Electricity、跨变量混合消融。这些保留为后续可检验假设。

## 上游与运行边界

- 官方上游：`https://github.com/yuqinie98/PatchTST.git`，固定提交 `204c21efe0b39603ad6e2ca640ef5896646ab1a9`。
- 上游目录放在 `reproduction/03_patchtst_official/upstream/PatchTST/`，视为不可修改的原件并由校验脚本验证提交和工作树。
- 项目侧包装器、兼容层、导出器和测试都放在 `reproduction/03_patchtst_official/`，不得在上游源文件中补丁。
- 新建独立虚拟环境。官方 requirements 固定 `torch==1.11.0`，但它不适配当前 Python 3.12 环境；先以现有 Autoformer 已验证的 PyTorch 2.5.1 CUDA 环境运行 smoke test，再以测试结果决定是否需要更旧 Python 环境。

## 数据、标签与尺度契约

ETTm1 的官方 dataloader 使用训练段 `[0,34560)` 拟合 `StandardScaler`，验证与测试窗口的边界与项目现有 DLinear/Autoformer 协议一致。PatchTST 预测和目标在其 dataloader 的标准化尺度上导出，数组形状固定为 `[test_windows, 96, 7]`。

官方 `exp_main.py` 只保存 `pred.npy`，没有保存 `true.npy`。项目包装器必须在不修改上游的前提下，使用同一 checkpoint、同一 test loader、同一 output slice 导出 `prediction` 和 `target`，并附带列名与运行元数据。比较脚本必须逐元素验证它与 Seasonal Naive、DLinear 的标签一致，验证成功后再反标准化计算原始尺度 MSE、MAE、RMSE 与逐变量指标。

模型内部 RevIN 与数据集 `StandardScaler` 是两层不同的归一化：前者逐窗口、逐变量且预测后还原；后者是训练集级别的评估尺度。两者都要被运行元数据记录。

## 目录和产物

```text
reproduction/03_patchtst_official/
├── upstream/PatchTST/             # ignored; fixed official source
├── compat/                        # only version compatibility code
├── scripts/                       # fetch, verify, env, smoke, GPU run, compare
├── src/patchtst_reproduction/     # export and evaluation glue
├── tests/                         # unit and wrapper-contract tests
├── provenance/                    # upstream pin and environment evidence
└── runs/<unique-run-name>/        # ignored model/output artifacts
```

`runs/<name>/predictions.npz` must contain normalized `prediction`, normalized `target`, and ETTm1 column names. Human-readable comparison Markdown and figures are generated under an ignored `comparison/` directory. A completed, aligned run is appended—not overwritten—to `experiment/results.md`.

## Acceptance criteria

1. The fetch/verify scripts reproduce the exact upstream commit and reject local upstream edits.
2. A CPU smoke path verifies one PatchTST forward pass and its `B×96×7` output without data leakage or upstream edits.
3. A GPU L40 command trains the official supervised model and produces exactly one self-contained normalized prediction archive.
4. Comparison refuses wrong array shapes, scale declarations, channel orders, ambiguous run outputs, and even one shifted label.
5. A successful comparison reports both normalized and original-scale metrics, parameter count, `P/S/N`, runtime evidence, seed, and the limitations of a single seed/run.
