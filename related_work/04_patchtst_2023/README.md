# PatchTST：把连续片段作为 token

## 引用与材料

Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023。论文：[arXiv:2211.14730](https://arxiv.org/abs/2211.14730)；官方代码：[yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)。本地 `paper.pdf` 为 arXiv v2。

## 要解决的问题

把每个时间点当 token 时，长回看窗口会产生很多 token、注意力矩阵很大，并且单点嵌入可能丢失局部形状。PatchTST 的问题是：能否让每个 token 是一段连续历史，从而同时保留局部语义并扩大回看范围？

## 核心思路与数据流

1. 对每个变量的历史序列按长度 `P`、步长 `S` 切成可重叠 patch。
2. 将一个 patch 映射为一个 token，送入 Transformer；token 数约从 `L` 降为 `(L-P)/S+1`。
3. **Channel independence**：每个变量独立经过同一套 embedding/Transformer 权重，而不是把同一时刻的变量混成一个 token。
4. 预测头把表示还原为每变量未来；论文也探索了掩码自监督预训练。

## 作者报告的证据

摘要声称 patch 同时保留局部信息、在相同回看窗口下二次降低注意力图的计算/显存，并在多变量长预测和迁移预训练中取得很强结果。这里的计算收益来自 token 数减少；真实速度仍取决于 patch、模型宽度和实现。

## 与前后工作的关系

Informer 近似注意力矩阵，PatchTST 则首先减少矩阵的边长。与 iTransformer 相反，PatchTST 的重点是每个变量的时间表示，并刻意不在注意力中直接建模变量间关系。

## 复现契约

- 候选数据：ETTm1、Weather、Electricity；统一主协议为 `L=96` 和四种 `H`。
- 首轮消融：固定 Transformer 宽度，比较点 token 与 patch token；再比较 channel independence 与变量混合。
- 记录：patch 长度、步长、patch 数、参数量、吞吐/显存、预测指标。
- 最小里程碑：输入张量 `B×L×C` 被无泄漏地切为按变量组织的 patch，输出仍为 `B×H×C`；先用形状测试和单变量可视化验证。

## 局限与待验证问题

- 通道独立会丢弃即时的跨变量因果/相关线索；变量相关很强时未必理想。
- patch 长度和步长会决定哪些短期尖峰被平滑或遗漏。
- 自监督预训练是额外实验阶段，不能与从头监督训练混在同一结论里。

