# DLinear：必须击败的简单强基线

## 引用与材料

Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023。论文：[arXiv:2205.13504](https://arxiv.org/abs/2205.13504)；官方代码：[cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)。本地 `paper.pdf` 为 arXiv v3。

## 要解决的问题

该工作直接质疑：把时间序列当成 Transformer 的时间 token 序列，是否真能可靠提取时间关系？它提出一层线性模型作为严格、廉价的对照，防止“复杂模型自然更好”的错觉。

## 核心思路与数据流

1. 以移动平均把输入拆为 trend 与 seasonal/residual。
2. 两个独立线性层分别把长度 `L` 的两部分直接映射到长度 `H` 的未来。
3. 相加后得到预测。没有自注意力、递归解码或深层非线性。

这不是“什么都不做”的 baseline：它保留了分解这一领域假设，却把预测器压到最小。因此一旦它表现很强，就要求复杂模型给出额外价值。

## 作者报告的证据

作者在九个真实数据集上将 LTSF-Linear 与多种当时 Transformer 比较，摘要称线性模型在全部案例中超过这些复杂方法。该说法依赖原文的数据管线与实现；复现时应逐预测窗核对，而不要只引用摘要结论。

## 与前后工作的关系

DLinear 是本项目的 P0 baseline。它继承了 Autoformer 的“分解”直觉，但去掉 attention，从而把“分解有效”与“Transformer 有效”拆开检验。PatchTST 和 iTransformer 都应至少与它在完全相同管线下比较。

## 复现契约

- 候选数据：ETTm1（7 变量）为第一站，随后 Electricity 和 Weather。
- 协议：`L=96`，`H={96,192,336,720}`；训练/验证/测试时间顺序固定；仅训练段缩放。
- 指标：MSE、MAE、RMSE、逐变量 MAE；保存 Seasonal Naive 的同表结果。
- 最小里程碑：实现季节/趋势移动平均、两个 `Linear(L,H)` 分支、求和输出；先过一个小窗口形状与无未来泄漏检查。

## 局限与待验证问题

- 每变量独立线性层对跨变量依赖的利用很有限。
- 对非平稳突变、复杂非线性或很长历史，线性映射可能不足。
- 它强不等于 Transformer 无价值；关键问题是增益在统一协议和多个数据集上是否可重复。

