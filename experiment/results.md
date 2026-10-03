# 实验结果（追加，不覆盖）

| 运行名 | 日期 | 数据/切分 | 输入/预测长度 | 模型与关键配置 | MSE | MAE | RMSE | 状态与备注 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |

| `ettm1_l96_h96_seasonal_naive_seed2026` | 2026-09-25 | ETTm1；标签 `[96,34560)` / `[34560,46080)` / `[46080,57600)` | 96 / 96 | Seasonal Naive，周期 96，7 个变量 | 7.994722 | 1.351231 | 2.827494 | 成功；原尺度指标；产物：`reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_seasonal_naive_seed2026/` |
| `ettm1_l96_h96_dlinear_seed2026` | 2026-09-25 | ETTm1；标签 `[96,34560)` / `[34560,46080)` / `[46080,57600)` | 96 / 96 | 共享通道 DLinear，MA 核 25，Adam 1e-3，batch 32，至多 20 epoch，patience 3 | 7.320728 | 1.404013 | 2.705684 | 成功；第 8 epoch 停止，最佳验证 checkpoint 为第 5 epoch；原尺度指标；产物：`reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_dlinear_seed2026/` |
| `ettm1_l96_h96_dlinear_seed2026_weightedval` | 2026-09-25 | ETTm1；标签 `[96,34560)` / `[34560,46080)` / `[46080,57600)` | 96 / 96 | 共享通道 DLinear，MA 核 25，Adam 1e-3，batch 32，至多 20 epoch，patience 3；验证 MSE 按全元素加权 | 7.320728 | 1.404013 | 2.705684 | 成功；第 8 epoch 停止，最佳验证 checkpoint 为第 5 epoch。此行是后续比较的权威本地 DLinear 结果；替代上行未加权批次验证汇总的运行。产物：`reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_dlinear_seed2026_weightedval/` |
| `ettm1_l96_h96_autoformer_seed2021_gpu_20261003` | 2026-10-03 | ETTm1；标签 `[96,34560)` / `[34560,46080)` / `[46080,57600)` | 96 / 96 | 官方 Autoformer 固定源码快照；M，`e_layers=2`，`d_layers=1`，`factor=3`，作者固定 seed 2021；L40 GPU | 9.044054 | 1.611376 | 3.007333 | 成功；与云端 Seasonal Naive `..._r2`、DLinear `..._r2` 测试标签逐窗对齐后比较。当前单次设置下全局原始尺度 MSE 高于两条基线；比较报告：`reproduction/02_autoformer_official/comparison/ettm1_l96_h96_initial_metrics.md`。 |

以上是此文件摘要和固定协议下的本机或云端运行结果，不是论文表格数字。
