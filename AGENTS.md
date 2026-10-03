# time_predict：项目总纲

## 目标

从零学习**通用多变量长预测窗时序预测**，建立可复现的基线与研究判断能力。第一阶段场景是电力、交通、天气和负荷，不做金融交易、异常检测、分类或插补。

## 每次会话的起点

1. 阅读本文件和 `handoff.md`。
2. 阅读 `related_work/README.md`，确认当前论文或基线的前置知识。
3. 动手前在 `experiment/evaluation.md` 固定切分、预测窗和指标。
4. 结束前更新 `handoff.md`；新假设追加到 `brainstorm.md`，实验结果只追加到 `experiment/results.md`。

## 不可违反的研究规则

- 时序训练、验证、测试必须按时间先后排列；禁止随机切分。
- 标准化、填补和任何统计特征只可在训练段拟合，再应用到验证/测试段。
- 与论文比较时保持同一数据、切分、输入长度、预测窗和指标；任何差异必须写出。
- `related_work/<paper>/paper.pdf` 和 `source/` 是源材料，阅读结论写在同目录 `README.md`，下载依据写在 `provenance.md`。
- 论文报告的数字不能写成自己的实验结果；本地结果必须有配置名并写入 `experiment/results.md`。
- 失败实验与负面结论同样保留，禁止覆盖历史记录。

## 首轮阅读顺序

`00_foundations` → DLinear → Autoformer → Informer → PatchTST → TimesNet → iTransformer。

