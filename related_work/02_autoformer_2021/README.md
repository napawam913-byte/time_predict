# Autoformer：把分解和周期相关放进网络

## 引用与材料

Wu et al., *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*, NeurIPS 2021。论文：[arXiv:2106.13008](https://arxiv.org/abs/2106.13008)；官方代码：[THUML/Autoformer](https://github.com/thuml/Autoformer)。本地 `paper.pdf` 为 arXiv v5。

## 要解决的问题

长预测的未来常同时含有缓慢趋势、季节性和局部噪声。作者认为即使稀疏化，点对点 attention 仍不自然地表示这些复杂周期，因此需要模型内生地分离不同部分，并聚合相似的周期相位。

## 核心思路与数据流

1. 在每个网络层以移动平均把序列分为趋势项和季节/残差项，而非只在模型外预处理一次。
2. **Auto-Correlation** 通过寻找高相关的时间延迟，按子序列而非单点聚合周期模式。
3. Encoder/decoder 的逐层分解让趋势逐步累积、季节成分继续被预测。

一句话理解：若昨天与一周前的相位相似，模型尝试直接聚合那一段，而不是让每个点单独寻找所有点。

## 作者报告的证据

作者报告在六个基准、五类实际应用上取得结果，并在摘要中声称相对改进 38%。这是作者的汇总口径，不可直接等同于任一数据集、任一预测窗的本地改善。

## 与前后工作的关系

相对 Informer，Autoformer 把重点从“近似注意力以降低复杂度”移到“分解与周期依赖是否更适合时序”。DLinear 将这一思想压缩为很简单的分解后线性映射，是非常重要的对照。

## 复现契约

- 候选数据：ETTm1、ETTh1、Weather、Electricity；首轮用 ETTm1 统一协议。
- 切分/缩放：严格遵守根目录 `AGENTS.md`；特别检查移动平均或任何归一化没有跨测试边界。
- 指标：MSE/MAE 为主，按 `H` 分开报告。
- 最小里程碑：独立实现或调用一个训练段内的移动平均分解，再在 DLinear 上验证；随后才加入 Auto-Correlation。

## 局限与待验证问题

- 移动平均窗口是强超参数，选错可能把真实快速变化当噪声。
- 周期性弱、突发强的变量不一定适合延迟聚合。
- “分解有效”与“模型更大/训练配置不同”必须通过与 DLinear、相同预算的消融区分。

