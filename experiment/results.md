# 实验结果（追加，不覆盖）

| 运行名 | 日期 | 数据/切分 | 输入/预测长度 | 模型与关键配置 | MSE | MAE | RMSE | 状态与备注 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |

| `ettm1_l96_h96_seasonal_naive_seed2026` | 2026-09-25 | ETTm1；标签 `[96,34560)` / `[34560,46080)` / `[46080,57600)` | 96 / 96 | Seasonal Naive，周期 96，7 个变量 | 7.994722 | 1.351231 | 2.827494 | 成功；原尺度指标；产物：`reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_seasonal_naive_seed2026/` |
| `ettm1_l96_h96_dlinear_seed2026` | 2026-09-25 | ETTm1；标签 `[96,34560)` / `[34560,46080)` / `[46080,57600)` | 96 / 96 | 共享通道 DLinear，MA 核 25，Adam 1e-3，batch 32，至多 20 epoch，patience 3 | 7.320728 | 1.404013 | 2.705684 | 成功；第 8 epoch 停止，最佳验证 checkpoint 为第 5 epoch；原尺度指标；产物：`reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_dlinear_seed2026/` |
| `ettm1_l96_h96_dlinear_seed2026_weightedval` | 2026-09-25 | ETTm1；标签 `[96,34560)` / `[34560,46080)` / `[46080,57600)` | 96 / 96 | 共享通道 DLinear，MA 核 25，Adam 1e-3，batch 32，至多 20 epoch，patience 3；验证 MSE 按全元素加权 | 7.320728 | 1.404013 | 2.705684 | 成功；第 8 epoch 停止，最佳验证 checkpoint 为第 5 epoch。此行是后续比较的权威本地 DLinear 结果；替代上行未加权批次验证汇总的运行。产物：`reproduction/01_dlinear_baseline/runs/ettm1_l96_h96_dlinear_seed2026_weightedval/` |

以上是本机、此文件摘要和此固定协议下的本地运行结果，不是论文表格数字。
