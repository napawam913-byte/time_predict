# 交接

## 当前状态（2026-10-05）

- PatchTST 官方监督版已在 L40 完成 ETTm1 `96→96` 运行：`ettm1_l96_h96_patchtst_seed2021_gpu_20261004_r3`。模型固定官方提交 `204c21efe0b39603ad6e2ca640ef5896646ab1a9`，结果为原始尺度 MSE/MAE `6.401452/1.266859`。
- 比较脚本已经以完整的 11425 个测试窗口逐元素验证 PatchTST、Seasonal Naive 与 DLinear 标签一致。PatchTST 的全局原始尺度 MSE 比 DLinear `7.320728` 低 12.56%，比 Seasonal Naive `7.994722` 低 19.93%；完整表格与各变量结果见 `reproduction/03_patchtst_official/comparison/ettm1_l96_h96_initial_metrics.md`（云端产物）和 `experiment/results.md`（版本化摘要）。
- 发现并修复了官方 PatchTST 测试 DataLoader 在 batch 128 时以 `drop_last=True` 丢失最后 33 个窗口的问题。项目导出器现在从同一官方测试 Dataset 构建 `drop_last=False` 的比较专用 DataLoader；不修改官方源码、训练权重或 checkpoint。
- 当前完整测试套件为 38 项；PatchTST 的本机形状冒烟已验证 `B×96×7 → B×7×16×12 → B×96×7`。

## 下一步（只做这一件）

为 TimesNet 建立与上述协议一致的官方复现包装，并先完成 ETTm1 `L=96 → H=96` 的单次、标签对齐比较。完成后再进行 `L=336` 的跨模型研究实验；不要把当前 `L=96` 结果称为论文表格复现。

## 历史状态（2026-10-03）

- 已建立基于 research-workflow 的最小研究骨架。
- 已保存并核验 Informer、Autoformer、DLinear、PatchTST、TimesNet、iTransformer 的公开 PDF 与 arXiv LaTeX 源码。
- 已完成首轮中文调研笔记，并在当前 CPU 电脑建立了独立 Python 环境与 CPU 版 PyTorch。
- 已从 ETTDataset 作者仓库下载并核验 ETTm1；文件摘要、字段、15 分钟频率、缺失检查与固定切分见 `dataset.md`。
- `reproduction/01_dlinear_baseline/` 已实现 Seasonal Naive、共享通道 DLinear、训练段标准化、时间安全滑窗、原尺度指标、早停 checkpoint、不可覆盖运行目录与 CLI。
- 真实 ETTm1 的 Seasonal Naive 与 DLinear 首轮 `L=H=96` 已完成；精确配置、预测和指标保存在各自 `runs/` 目录，汇总见 `experiment/results.md`。
- DLinear 的权威本地结果是 `ettm1_l96_h96_dlinear_seed2026_weightedval`：验证 MSE 按全元素而非按批次平均汇总。较早的同名基础运行保留为可追溯产物，但不应用于后续比较；当前完整测试数为 16。
- 官方 Autoformer 已在 L40 完成首轮 `96→96`，原始尺度 MSE/MAE 为 `9.044054/1.611376`；与云端 Seasonal Naive、DLinear 的标签已逐元素对齐，结果已追加到 `experiment/results.md`。
- 已完成 PatchTST 官方监督版包装，固定提交 `204c21efe0b39603ad6e2ca640ef5896646ab1a9`。本机 CPU 冒烟实际验证了官方模型的 `B×96×7 → B×7×16×12 → B×96×7` 形状；完整测试套件为 36 项。尚未执行 PatchTST 的完整训练，故 `experiment/results.md` 中没有 PatchTST 数值。

## 当时的下一步（已完成）

在 L40 上运行官方监督版 PatchTST 的 ETTm1 `L=96 → H=96`：先 `fetch_upstream.sh`、`create_gpu_env.sh cu121`，再以新运行名执行 `run_official_ettm1_96_96_gpu.sh`。完成后使用 `compare_ettm1_l96_h96.py` 与现有云端 Seasonal Naive/DLinear `..._r2` 产物比较；只有标签对齐通过，才把结果追加到 `experiment/results.md`。不要将这个 `L=96` 结果称为论文 `L=336` 表格复现。

## 已知风险

- 论文之间的数据版本、变量数、输入长度和实现细节可能不同；不要直接比较表格数字。
- 任何高于 DLinear 的结果先检查时间切分、归一化拟合范围和未来特征是否泄漏。
