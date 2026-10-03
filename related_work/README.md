# 相关工作：通用多变量长预测

## 如何使用本目录

每个目录保存不可变的 `paper.pdf`、作者提交的 LaTeX `source/`、来源校验 `provenance.md` 和中文阅读笔记 `README.md`。阅读笔记中的“作者报告”不是本地复现结论；实际运行只记录在 `../experiment/results.md`。

## 建议学习路线

| 顺序 | 工作 | 先学会什么 | 为什么现在读 |
| ---: | --- | --- | --- |
| 0 | [基础](00_foundations/README.md) | 时间切分、缩放、Naive、指标 | 先能识别泄漏和不公平比较。 |
| 1 | [DLinear](03_dlinear_2023/README.md) | 分解后的线性预测 | 最小但很强的可复现 baseline。 |
| 2 | [Autoformer](02_autoformer_2021/README.md) | 内置分解和周期相关 | 理解“时序结构”怎样进入模型。 |
| 3 | [Informer](01_informer_2021/README.md) | 高效长序列注意力 | 理解 Transformer 的计算瓶颈。 |
| 4 | [PatchTST](04_patchtst_2023/README.md) | patch token 和通道独立 | 把局部片段与长回看窗口结合。 |
| 5 | [TimesNet](05_timesnet_2023/README.md) | 多周期二维变化 | 从另一种周期归纳偏置看问题。 |
| 6 | [iTransformer](06_itransformer_2024/README.md) | 变量 token 与变量间相关 | 直接面向多变量依赖。 |

## 横向比较

| 方法 | 基本 token / 操作单位 | 主要归纳偏置 | 跨变量交互 | 核心动机 | 首轮复现优先级 |
| --- | --- | --- | --- | --- | --- |
| Seasonal Naive / ARIMA | 单变量历史值 | 持续性、季节重复、线性自回归 | 无 | 最低风险参照 | P0 |
| DLinear | 每变量历史窗口 | 趋势/残差的线性映射 | 默认弱/无 | 检验复杂 Transformer 是否真的必要 | P0 |
| Informer | 时间点 token | 稀疏注意力、蒸馏 | 同一时间点变量混合 | 降低长序列注意力开销 | P3 |
| Autoformer | 子序列与周期延迟 | 渐进分解、周期相关 | 依赖实现的嵌入混合 | 用周期替代点对点注意力 | P2 |
| PatchTST | 每变量的时间 patch | 局部连续片段 | 通道独立（权重共享） | 更长回看且少注意力 token | P2 |
| TimesNet | 多周期重排后的二维张量 | 周内/周期间二维变化 | 由张量通道表示 | 自适应发现多周期 | P3 |
| iTransformer | 变量 token | 每变量时间历史表示 | 自注意力直接建模变量间关系 | 避免时间 token 混合异质变量 | P2 |

## 研究判断

这六篇论文不是“越新越好”的模型清单，而是六种可检验主张：效率（Informer）、分解（Autoformer）、简单性（DLinear）、局部 patch（PatchTST）、多周期（TimesNet）和变量相关（iTransformer）。每次比较只检验其中一项变化，先固定数据与评价协议。

## 来源政策

论文 PDF 与 `source/` 都来自各 `provenance.md` 所列 arXiv 端点；`source.tar.gz` 是作者上传原件，解压后的文件供检索。任何无法得到的材料都必须明确标为不可用，不能用第三方重建版本替代。

