# 交接

## 当前状态（2026-09-25）

- 已建立基于 research-workflow 的最小研究骨架。
- 已保存并核验 Informer、Autoformer、DLinear、PatchTST、TimesNet、iTransformer 的公开 PDF 与 arXiv LaTeX 源码。
- 已完成首轮中文调研笔记，并在当前 CPU 电脑建立了独立 Python 环境与 CPU 版 PyTorch。
- 已从 ETTDataset 作者仓库下载并核验 ETTm1；文件摘要、字段、15 分钟频率、缺失检查与固定切分见 `dataset.md`。
- `reproduction/01_dlinear_baseline/` 已实现 Seasonal Naive、共享通道 DLinear、训练段标准化、时间安全滑窗、原尺度指标、早停 checkpoint、不可覆盖运行目录与 CLI。
- 14 项单元/合成集/CPU 集成测试通过。真实 ETTm1 的 Seasonal Naive 与 DLinear 首轮 `L=H=96` 已完成；精确配置、预测和指标保存在各自 `runs/` 目录，汇总见 `experiment/results.md`。
- DLinear 的权威本地结果是 `ettm1_l96_h96_dlinear_seed2026_weightedval`：验证 MSE 按全元素而非按批次平均汇总。较早的同名基础运行保留为可追溯产物，但不应用于后续比较；当前完整测试数为 16。

## 下一步（只做这一件）

先阅读两个真实运行的 `history.json`、`metrics.json` 和每变量 MAE，理解“DLinear 的 MSE 更低但 MAE 未必更低”的含义。确认后再扩展到预测窗 192、336、720，或进入 Autoformer/PatchTST；每次只改变一个因素并追加记录，不复用既有运行名。

## 已知风险

- 论文之间的数据版本、变量数、输入长度和实现细节可能不同；不要直接比较表格数字。
- 任何高于 DLinear 的结果先检查时间切分、归一化拟合范围和未来特征是否泄漏。
