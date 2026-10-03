# 交接

## 当前状态（2026-10-03）

- 已建立基于 research-workflow 的最小研究骨架。
- 已保存并核验 Informer、Autoformer、DLinear、PatchTST、TimesNet、iTransformer 的公开 PDF 与 arXiv LaTeX 源码。
- 已完成首轮中文调研笔记，并在当前 CPU 电脑建立了独立 Python 环境与 CPU 版 PyTorch。
- 已从 ETTDataset 作者仓库下载并核验 ETTm1；文件摘要、字段、15 分钟频率、缺失检查与固定切分见 `dataset.md`。
- `reproduction/01_dlinear_baseline/` 已实现 Seasonal Naive、共享通道 DLinear、训练段标准化、时间安全滑窗、原尺度指标、早停 checkpoint、不可覆盖运行目录与 CLI。
- 真实 ETTm1 的 Seasonal Naive 与 DLinear 首轮 `L=H=96` 已完成；精确配置、预测和指标保存在各自 `runs/` 目录，汇总见 `experiment/results.md`。
- DLinear 的权威本地结果是 `ettm1_l96_h96_dlinear_seed2026_weightedval`：验证 MSE 按全元素而非按批次平均汇总。较早的同名基础运行保留为可追溯产物，但不应用于后续比较；当前完整测试数为 16。
- 官方 Autoformer 已在 L40 完成首轮 `96→96`，原始尺度 MSE/MAE 为 `9.044054/1.611376`；与云端 Seasonal Naive、DLinear 的标签已逐元素对齐，结果已追加到 `experiment/results.md`。
- 已完成 PatchTST 官方监督版包装，固定提交 `204c21efe0b39603ad6e2ca640ef5896646ab1a9`。本机 CPU 冒烟实际验证了官方模型的 `B×96×7 → B×7×16×12 → B×96×7` 形状；完整测试套件为 36 项。尚未执行 PatchTST 的完整训练，故 `experiment/results.md` 中没有 PatchTST 数值。

## 下一步（只做这一件）

在 L40 上运行官方监督版 PatchTST 的 ETTm1 `L=96 → H=96`：先 `fetch_upstream.sh`、`create_gpu_env.sh cu121`，再以新运行名执行 `run_official_ettm1_96_96_gpu.sh`。完成后使用 `compare_ettm1_l96_h96.py` 与现有云端 Seasonal Naive/DLinear `..._r2` 产物比较；只有标签对齐通过，才把结果追加到 `experiment/results.md`。不要将这个 `L=96` 结果称为论文 `L=336` 表格复现。

## 已知风险

- 论文之间的数据版本、变量数、输入长度和实现细节可能不同；不要直接比较表格数字。
- 任何高于 DLinear 的结果先检查时间切分、归一化拟合范围和未来特征是否泄漏。
