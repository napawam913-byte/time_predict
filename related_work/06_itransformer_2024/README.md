# iTransformer：让变量而非时间点成为 token

## 引用与材料

Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024。论文：[arXiv:2310.06625](https://arxiv.org/abs/2310.06625)；官方代码：[THUML/iTransformer](https://github.com/thuml/iTransformer)。本地 `paper.pdf` 为 arXiv v4。

## 要解决的问题

常规时间 token Transformer 把同一时刻的多个变量融合为一个 token。这些变量可能代表不同物理量和滞后事件，融合后 attention 的语义可能不清晰；同时更长的回看会增加时间 token 数与计算。iTransformer 问：若一个 token 对应一个变量的完整历史，是否更适合多变量预测？

## 核心思路与数据流

1. 输入 `B×L×C` 转置观点：变量 `c` 的 `L` 个历史点先嵌入为一个 variate token。
2. 标准 self-attention 在 `C` 个变量 token 上运行，直接学习变量间相关性。
3. 每变量前馈网络沿该变量的表示学习非线性时间特征；再用预测头输出未来 `H` 点。

“inverted” 指改变 Transformer 组件施加的维度，不是发明新的 attention 算子。

## 作者报告的证据

作者摘要报告在多个现实数据集上获得强结果，并声称更能处理不同变量数与任意回看窗口。应把它看成待复核的模型主张：变量数 `C` 很小或变量互不相关时，变量注意力的额外价值可能有限。

## 与前后工作的关系

PatchTST 把每变量时间序列独立编码、共享权重；iTransformer 则将变量作为 token，用 attention 显式建模交互。它因此是验证“跨变量关系是否值得建模”的直接实验对象，也是对 DLinear 变量独立预测的补充。

## 复现契约

- 候选数据：ETTm1（7 个变量）、Electricity（较多变量）、Weather；逐步增加变量数检验扩展性。
- 协议：统一 `L=96` 与四个预测窗；额外记录变量数 `C` 和 attention 矩阵大小 `C×C`。
- 公平比较：与 PatchTST、DLinear 使用同一缩放、时间切分、训练预算与预测头输出形状。
- 最小里程碑：实现变量维度转置并断言输出仍为 `B×H×C`；在人工构造的两相关变量序列上检查 attention 是否能改善独立预测。

## 局限与待验证问题

- `C` 很大时变量 attention 也会二次增长；并非免费扩展。
- attention 权重是模型内部量，不可直接解释为因果关系。
- 将整段历史压成单变量 token 可能错过细粒度时间对齐，需要与 patch/时间 token 方法做控制实验。

