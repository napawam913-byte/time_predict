# Informer：高效长序列 Transformer

## 引用与材料

Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021。论文：[arXiv:2012.07436](https://arxiv.org/abs/2012.07436)；官方代码：[Informer2020](https://github.com/zhouhaoyi/Informer2020)。本地 `paper.pdf` 为 arXiv v3，`source/` 为作者提交的 LaTeX 源码。

## 要解决的问题

长预测窗需要模型从很长的历史中找到依赖，但标准自注意力对长度 `L` 的时间/显存开销为二次方。Informer 的问题是：怎样在不逐步自回归解码的前提下，用更低开销处理长输入和长输出？

## 核心思路与数据流

1. 把每个时间点（包含各变量的观测/时间特征）嵌入为时间 token。
2. **ProbSparse attention** 只重点计算少数“活跃”query 的注意力，论文目标复杂度是 `O(L log L)`。
3. Encoder 层间用 self-attention distilling 下采样 token，突出主要模式并缩短序列。
4. Decoder 一次前向生成整段未来，而不是每预测一步再把结果喂回去。

## 作者报告的证据

作者在四个大规模数据集上比较多个预测窗，声称该模型优于当时方法。这里仅记录作者主张；数值、数据版本与每个预测窗必须在复现前从 PDF 表格与代码配置再次核验。

## 与前后工作的关系

Informer 关注的是**时间 token 注意力如何变便宜**。Autoformer 问“周期结构是否比稀疏点对点注意力更好”；PatchTST 减少 token 数量；iTransformer 则反转 token 的含义，先建模变量关系。

## 复现契约

- 候选数据：ETTm1，随后 Electricity/Weather；先完成统一的 `L=96`、`H={96,192,336,720}` 协议。
- 切分：只用时间顺序切分；每变量缩放仅由训练段拟合。
- 指标：MSE、MAE、RMSE、逐变量 MAE；同时记录参数量、显存/训练时间，以检验“高效”主张。
- 最小里程碑：先让标准 Transformer 与 DLinear 共用数据管线，再替换为 ProbSparse；否则无法归因于注意力近似。

## 局限与待验证问题

- 稀疏注意力能否在短输入或变量较少时仍有收益？
- 时间 token 中混合了不同物理量，注意力图未必对应有意义的变量关系。
- 下采样可能损害局部尖峰；需按变量与预测窗检查，而不是只看平均误差。

