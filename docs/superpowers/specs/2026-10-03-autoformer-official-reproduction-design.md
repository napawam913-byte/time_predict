# Autoformer 官方复现与基线对比设计

## 目标

在 ETTm1 多变量长时预测任务上运行 `thuml/Autoformer` 的固定官方源码版本，复现其 `96 -> 96` 配置；随后将结果与本项目已有的 Seasonal Naive 和 DLinear 基线在严格相同的数据协议下比较。产物应让学习者可以追溯“Autoformer 的分解与 Auto-Correlation 在此任务上带来了什么”，而不是宣称复现完整论文排行榜。

## 已确认的约束

- 工作区：`/home/xc_ubantu/homelab/projects/time_predict`。
- 不重写、不改动 Autoformer 的模型源码；官方源码以 Git commit 固定并保留在独立目录。
- 首轮数据为已有 `DataSet/ETTm1/ETTm1.csv`，多变量预测多变量（7 个数值列），输入长度和预测长度均为 96。
- 先在当前 CPU 笔记本上运行。当前可用环境为 Python 3.12.3、PyTorch 2.5.1+cpu、无 CUDA。
- 当前 DLinear 结果是可追溯基线，但其现有结果使用 seed 2026；首轮公平比较要将随机种子差异明确记录，后续使用多 seed 报告均值与标准差。

## 外部基准与协议

作者仓库为 <https://github.com/thuml/Autoformer>。其 ETTm1 脚本包含 `features=M`、`seq_len=96`、`label_len=48`、`pred_len=96`、`e_layers=2`、`d_layers=1`、`factor=3`、`enc_in=dec_in=c_out=7` 的实验条目。

官方 `Dataset_ETT_minute` 的边界为训练 `[0, 34560)`、验证数据 `[34464, 46080)`、测试数据 `[45984, 57600)`；在 `seq_len=96`、`pred_len=96` 下，对应的预测标签范围分别为：

- train：`[96, 34560)`；
- validation：`[34560, 46080)`；
- test：`[46080, 57600)`。

这与现有 DLinear 数据窗协议一致。两方标准化都只拟合原始训练行 `[0,34560)`，并采用逐变量、总体方差（`ddof=0`）的标准差。

官方仓库保存并报告的是标准化尺度的 `pred.npy`/`true.npy` 指标；已有 DLinear 基线保存的是反标准化后的原尺度指标。因此比较报告必须对所有模型同时提供：

1. 标准化尺度 MSE、MAE：用于与官方 Autoformer 输出、论文惯例对应；
2. 原始尺度 MSE、MAE：便于理解实际变量上的误差；
3. 每变量指标：避免只由量纲较大的变量主导总体 MSE。

在生成比较表之前，评估工具必须验证各模型的测试标签在标准化后数值相同、形状相同、时间窗数量相同；否则报告为不可比，不生成排名结论。

## 目录与产物

新增目录 `reproduction/02_autoformer_official/`，职责如下：

```text
02_autoformer_official/
  upstream/Autoformer/       # 官方 git 仓库的固定快照；不编辑模型源码
  scripts/                   # 仓库外的运行包装命令与评估入口
  provenance/                # upstream commit、SHA-256、环境冻结、精确命令
  runs/                      # 每次运行的 checkpoint、官方结果副本、日志和元数据
  comparison/                # 标准化/原尺度指标、每变量指标、预测图和报告
  README.md                  # 初学者可重复执行的说明
```

现有 `reproduction/01_dlinear_baseline/` 保持不变。比较代码只读取其已有的 `runs/` 产物，不覆盖任何运行。

## 执行设计

### 1. 官方代码获取与溯源

从官方 URL 克隆到 `upstream/Autoformer`，立即记录完整 commit SHA、远程 URL、源码树状态和作者 `requirements.txt` 的 SHA-256。`upstream/Autoformer` 任何被跟踪文件发生修改即失败；本项目只在外层放置运行包装和评估代码。

### 2. 隔离环境与 CPU 冒烟验证

建立 Autoformer 专用 Python 环境，并记录实际 Python、PyTorch、NumPy、Pandas、scikit-learn 版本。先以官方参数执行一次短暂 CPU 冒烟运行，只验证：数据能读取、forward/backward 可执行、checkpoint 与结果格式能够产生。冒烟结果不得写入比较表。

作者 README 标注的 Python 3.6 / PyTorch 1.9 与当前环境不同。先运行未修改官方源码；若实际发生兼容错误，只允许增加仓库外的启动包装或改用兼容的隔离环境，官方源码保持零修改。任何环境差异都必须记录在 `provenance/`，并使该轮运行标记为“兼容环境运行”，不能标为原始环境复现。不得改变模型结构、损失、数据切分或评估逻辑。

### 3. 正式 Autoformer 运行

以官方 `ETTm1_96_96` 条目的模型参数执行一次正式 CPU 训练。数据路径指向本项目已有 CSV，不复制也不下载第二份数据。固定并记录官方代码的 seed 2021、batch size、学习率、epoch 上限、early stopping、运行时长和 CPU 信息。

每次正式运行拥有唯一 run id，保存：完整命令、stdout/stderr、参数 JSON、环境冻结、checkpoint 哈希、官方 `pred.npy`、`true.npy`、`metrics.npy` 和拷贝后的结果路径。已有输出绝不覆盖。

### 4. 独立比较评估

新增仓库外评估工具，输入 Autoformer 的标准化预测/标签以及 DLinear、Seasonal Naive 的原尺度预测/标签。工具用训练集统计量在两种尺度之间无损转换，计算总体和每变量的 MSE、MAE、RMSE；同时进行标签对齐验证。

首张比较表包含：Seasonal Naive、DLinear（已有 seed 2026 记录）、Autoformer（官方固定 seed 2021）。标题明确注明“单次运行、不同固定 seed，仅作学习用初步比较”。若重新运行 DLinear，使用 seed 2021 作为更接近的单 seed 对照，但不覆写 seed 2026 结果。

### 5. 图与结论边界

为同一个已验证对齐的测试窗口绘制输入历史、真实未来、三种模型预测。图只用于观察趋势、相位和振幅；模型优劣以完整测试集的量化指标为准。报告不得根据单条曲线或单个随机种子宣称普遍优越。

## 不在首轮范围内

- 不复现论文所有数据集、所有预测长度或全部十个基线。
- 不重新实现 Autoformer、FFT 或 Auto-Correlation。
- 不修改作者模型以追求更优数字。
- 不将初次 CPU 结果直接与论文表格宣称相等。

## 验收条件

1. 官方源码 URL 与 commit 可复查，且模型源码未被编辑。
2. `96 -> 96` 的 ETTm1 正式运行在 CPU 上完成并有 checkpoint、预测、标签、指标、日志和环境记录。
3. 比较器确认三种模型的测试标签与窗口对齐，分别输出标准化与原尺度总体/每变量指标。
4. `comparison/report.md` 说明 Autoformer 相对于现有基线的观察、不可比风险和单 seed 限制。
5. README 中的命令可从该目录重新运行，不依赖未记录的手工步骤。
