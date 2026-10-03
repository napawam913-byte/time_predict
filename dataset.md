# 数据集说明

## ETTm1（首轮基线数据）

- **本地文件：** `DataSet/ETTm1/ETTm1.csv`
- **规范来源：** [zhouhaoyi/ETDataset 的 ETTm1.csv](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTm1.csv)
- **下载 URL：** `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv`
- **SHA-256：** `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`
- **读取结果：** 69,680 行 × 8 列；时间从 `2016-07-01 00:00:00` 到 `2018-06-26 19:45:00`。
- **时间粒度：** 严格单调、无重复的 15 分钟间隔；相邻 69,679 个间隔均为 15 分钟。
- **字段：** `date`, `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`。其中后 7 列全为数值预测目标；`OT` 是油温，其他列是负载特征。
- **缺失检查：** 全部 8 列缺失值为 0；日期可解析、无重复、严格递增。

### 固定首轮切分

行索引以左闭右开区间表示。scaler 只在原始训练行 `[0, 34560)` 拟合；训练/验证/测试的**标签**范围依次为 `[96, 34560)`、`[34560, 46080)`、`[46080, 57600)`。验证和测试窗口可以使用此前历史，但标签不能跨越自己的边界。首轮使用 `L=96`、`H=96`，因此只使用前 57,600 行；剩余行不参与本轮训练、早停或测试。

数据原件不应被当作实验输出版本化。每次实际运行的配置、预测和指标都保存在对应的 `reproduction/01_dlinear_baseline/runs/<run-name>/`。
