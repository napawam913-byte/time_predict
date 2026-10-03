# ETTm1 DLinear 最小复现设计

## 目的与成功标准

本子项目服务于通用多变量长预测窗时序预测的入门学习。目标不是立刻复刻论文所有表格，而是在笔记项目中建立一条可检查的数据到指标链路：读入 ETTm1，按时间切分，使用训练段统计量标准化，比较 Seasonal Naive 与 DLinear，并在原始数值尺度上报告误差。

成功标准：在一台无 CUDA GPU 的笔记本上，能以一条命令完成 `L=96, H=96` 的训练和测试；每次运行留下可追溯配置、预测和指标；测试能阻止时间切分、标准化和张量形状的常见错误。

## 范围

第一轮只实现下列内容：

- 数据集：ETTm1，原始 CSV 固定于 `DataSet/ETTm1/ETTm1.csv`。
- 任务：多变量预测（CSV 中除 `date` 外的全部数值列均作为输入和预测目标）。
- 协议：历史长度 `L=96`，预测长度 `H=96`，随机种子 `2026`。
- 基线：日周期 Seasonal Naive（周期为 ETTm1 的 96 个 15 分钟点）和 DLinear。
- 指标：在反标准化后的原始尺度上计算 MSE、MAE、RMSE 和逐变量 MAE。

首轮不实现 Informer、时间/位置 embedding、外生变量、超参数搜索、论文的全部预测窗或多随机种子汇总。它们在最小协议稳定后再增加。

## 文件与职责

复现源代码集中在 `reproduction/01_dlinear_baseline/`，以免和论文资料、原始数据和跨实验记录混放：

```text
reproduction/01_dlinear_baseline/
├── README.md                  # 环境、数据来源、运行命令和本地复现边界
├── requirements.txt            # 最小运行依赖
├── configs/ettm1_l96_h96.json # 固定首轮配置
├── src/ltsf_baseline/
│   ├── data.py                 # ETTm1 切分、训练段 scaler、滑动窗口
│   ├── baselines.py            # Seasonal Naive
│   ├── dlinear.py              # 移动平均分解和 DLinear
│   ├── metrics.py              # 原始尺度指标
│   ├── runner.py               # 训练、验证、测试及运行产物
│   └── cli.py                  # `python -m ltsf_baseline.cli` 入口
├── tests/                      # 数据、基线、模型、指标和训练烟雾测试
└── runs/                       # 每次本地运行的配置、指标、预测与 checkpoint；不写论文数字
```

原始数据只放在 `DataSet/ETTm1/`。跨模型、可比较的最终结果只追加到 `experiment/results.md`；数据来源、校验和及切分边界补充到 `dataset.md`；下一步状态更新到 `handoff.md`。

## 数据协议与泄漏防护

ETTm1 的采样间隔是 15 分钟。为与论文官方 loader 的 ETT minute 协议对齐，按连续时间区间切分：

- 训练目标区间：前 `12 × 30 × 24 × 4 = 34,560` 个点；
- 验证目标区间：随后 `4 × 30 × 24 × 4 = 11,520` 个点；
- 测试目标区间：最后 `4 × 30 × 24 × 4 = 11,520` 个点。

验证和测试样本可以读取其目标区间开始之前的 `L=96` 个历史点，但任何预测目标都必须完全落在本 split 的目标区间内。均值和标准差只由训练目标区间拟合；同一 scaler 再变换验证与测试数据。模型预测和真实标签在计算指标前都反标准化。

输入与标签的公共形状是 `[batch, length, channels]`：输入为 `[B, 96, C]`，标签和预测为 `[B, 96, C]`。数据读取应拒绝缺少 `date` 列、缺少数值变量、非递增时间戳或不足以生成一个完整窗口的 CSV。

## 模型与训练

Seasonal Naive 不训练参数。对于每个未来位置 `h`，它复制同一日的历史位置：`prediction[:, h, :] = input[:, h, :]`，因为 `L=H=96` 且周期为 96。

DLinear 按论文官方实现的核心结构：

1. 用核宽 25、两端复制填充的移动平均得到 Trend；
2. `Seasonal = input - Trend`；
3. 以共享的 `Linear(96, 96)` 分别映射 Seasonal 和 Trend；
4. 两个输出相加，返回 `[B, 96, C]`。

首轮使用共享通道线性层（`individual=false`），Adam、学习率 `1e-3`、batch size `32`、最多 20 个 epoch、验证 MSE 早停（patience 3）。训练损失使用标准化空间的 MSE；选择最佳验证 checkpoint；最终测试指标使用原始尺度。

## 运行产物与错误处理

一个运行名由数据集、`L`、`H`、模型和种子组成，例如 `ettm1_l96_h96_dlinear_seed2026`。在对应的 `runs/<run_name>/` 目录保存：配置 JSON、训练历史 JSON、原始尺度的 `metrics.json`、预测数组和最佳 checkpoint。已存在的运行目录必须拒绝覆盖，除非用户明确传入新的运行名。

命令在缺少 CSV、无效配置、NaN/Inf 数据、训练批次数为零或无法加载 checkpoint 时，应给出清晰错误而不是悄悄产出数字。首次 DLinear 成功运行后，才可向 `experiment/results.md` 追加一行；若失败，也应保留失败原因。

## 测试与验收

采用测试先行。实现前必须先看到以下测试因缺少功能而失败：

- scaler 的均值、标准差只受训练段数据影响；验证/测试极值不得改变它；
- 生成的 train/val/test 标签不会越过各自目标区间，且窗口形状为 `[96, C]`；
- Seasonal Naive 对手工构造的日周期序列复制正确位置；
- 移动平均分解满足 `seasonal + trend = input`，DLinear 输出形状为 `[B, 96, C]`；
- 已知的小数组在反标准化后得到手工可验证的 MSE、MAE、RMSE 和逐变量 MAE；
- 极小合成数据可完成一次 CPU 训练、写出运行产物并加载最佳 checkpoint。

在真实 ETTm1 运行前，所有单元测试和 CPU 烟雾测试必须通过。首轮结果不要求等于论文数字：输入长度、训练 epoch、硬件和官方默认配置均可能不同，差异必须记录。
