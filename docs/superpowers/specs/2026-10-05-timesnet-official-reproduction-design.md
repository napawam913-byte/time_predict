# TimesNet 官方复现设计

**日期：** 2026-10-05  
**状态：** 已确认，实施中  
**范围：** ETTm1 多变量长期预测，首轮仅 `L=96 → H=96`

## 1. 目标与边界

为论文 *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis* 建立可追溯的官方代码复现，并把它放入本项目既有的、严格标签对齐的横向比较中。

本轮完成的定义是：使用 TimesNet 作者维护的、论文时期 Time-Series-Library 冻结版本，在 ETTm1 上完成单次 `96 → 96` 训练和测试；导出完整的 11,425 个测试窗口预测；与已有 Seasonal Naive、DLinear、Autoformer、PatchTST 使用同一测试标签计算原始尺度指标。

以下内容**不在本轮范围**：

- 不自行实现或重写 TimesBlock、FFT 选周期、二维 reshape、Inception 卷积或训练循环；
- 不修改作者源码以改变模型行为；
- 不把 `L=96` 的结果表述为论文中常见的 `L=336` 表格复现；
- 不开始金融预测、分类、插补、异常检测或多数据集扫参。

## 2. 源码选择

采用作者维护的 [THUML/Time-Series-Library](https://github.com/thuml/Time-Series-Library)，固定到论文时期提交 `2665a3143dae12d1cbcc31ddd396bbff48773bce`（2023-03-31）。该提交同时含有 `models/TimesNet.py`、`run.py`、ETTm1 官方脚本以及长预测数据管线。

这是一次来源更正，而非改用第三方实现：`THUML/TimesNet` 的 README 明确说明完整代码和脚本已包含在 Time-Series-Library；实际核查也确认前者没有 `run.py` 或 `models/TimesNet.py`。选择该 TSLib 提交可避免使用今天主分支在论文之后加入的实现、依赖和任务扩展，同时仍完全使用论文作者的模型与训练代码。拉取时必须把仓库 URL 与固定 commit 写入 `provenance/upstream.json`，以后任何人均应能检出同一版本。

作者仓库作为不可变上游边界：

```text
reproduction/04_timesnet_official/upstream/Time-Series-Library/
```

该目录不进入本项目版本控制；验证脚本必须检查远程地址、固定 commit 与上游工作树干净状态。项目只版本化外围复现脚本、配置、测试与元数据。

## 3. 目录与职责

```text
reproduction/04_timesnet_official/
├── README.md                         # 命令、边界、与论文差异
├── provenance/upstream.json          # 作者 TSLib URL 与精确 commit
├── requirements.common.txt           # 外围运行依赖的兼容约束
├── scripts/
│   ├── fetch_upstream.sh              # 拉取并固定官方源
│   ├── verify_upstream.sh             # 检查 URL / commit / 干净状态
│   ├── create_cpu_env.sh              # CPU 冒烟环境
│   ├── create_gpu_env.sh              # L40 CUDA 环境
│   ├── run_smoke_cpu.sh               # 真实官方模型形状冒烟
│   ├── run_official_ettm1_96_96_gpu.sh# 只调用作者训练代码
│   ├── export_ettm1_predictions.py    # 外围导出完整测试集预测
│   └── compare_ettm1_l96_h96.py       # 对齐后计算统一指标
├── src/timesnet_reproduction/         # 导出、评测、形状检查；不含模型实现
├── tests/                             # 外围契约与回归测试
└── runs/                              # 忽略的本地/云端运行产物
```

`reproduction/common/ltsf_evaluation/` 是已经被 PatchTST 使用的共享评测实现。本轮可以复用它，但不得改变它的既有比较含义而影响前面模型的已记录结果。

## 4. 数据与公平测评契约

固定使用本项目的 `DataSet/ETTm1/ETTm1.csv`：7 个变量 `HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`，15 分钟采样。切分、标准化与指标遵循 `experiment/evaluation.md`：

- 时间先后切分，任何统计量仅从训练段拟合；
- 输入长度 `L=96`，预测长度 `H=96`；
- 多变量设定（`M`），预测全部 7 个变量；
- 在反标准化的原始尺度上报告全局 MSE、MAE、RMSE 与逐变量 MSE/MAE；
- 与基线比较前，预测和真实标签都必须是 `(11425, 96, 7)`，并逐元素验证同一真实标签。

如果 TimesNet 官方数据读取器的切分或标准化约定与项目基线不兼容，先停止比较、记录差异并调整**外围数据/导出契约**；不能通过改作者模型实现来制造可比性。

## 5. 运行与数据流

```text
ETTm1 CSV
  → 作者官方数据集与训练代码
  → 作者 TimesNet checkpoint
  → 同一官方测试 Dataset（导出时 drop_last=False）
  → predictions.npz: prediction / target / columns
  → 严格标签对齐检查
  → 原始尺度指标与比较 Markdown
```

训练启动脚本只传递明确记录的官方参数，例如任务、数据文件、`seq_len=96`、`pred_len=96`、多变量模式、随机种子和运行目录。它以作者 ETTm1 脚本 `scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh` 的 `d_model=64`、`d_ff=64`、`e_layers=2`、`top_k=5` 为基础，且不能替换官方 `models/TimesNet.py` 或训练器。

导出器可以导入官方模型、官方 checkpoint、官方 Dataset 和官方 scaler，但只承担两项外围职责：

1. 用 `drop_last=False` 取得测试段的全部批次，避免尾部窗口被静默丢弃；
2. 把模型输出与对应真实标签按时间顺序保存为统一 NPZ 契约。

导出器不重写模型前向计算。若为学习记录周期，需要通过官方模型暴露的结果或只读观测包装记录某一个真实输入的 `top_k`、选中频率/周期和二维 reshape 尺寸；该观测不得参与训练、验证或预测数值。

## 6. 配置、产物与不可覆盖性

每次运行使用唯一 run name，例如：

```text
ettm1_l96_h96_timesnet_seed2021_gpu_YYYYMMDD
```

运行目录已存在时启动脚本必须失败，不能覆盖 checkpoint、日志或预测。一次成功运行至少保留：

- `config.json`：实际参数、作者 commit、数据路径与种子；
- `official/`：作者代码产生的日志、checkpoint 与原始结果；
- `predictions.npz`：标准化尺度的 `prediction`、`target`、`columns`；
- 比较输出：统一指标表与可选的定性窗口图。

版本化摘要仅追加到 `experiment/results.md`；过程与下一步只追加到 `handoff.md`；新的可检验研究判断追加到 `brainstorm.md`。

## 7. 失败防护

| 风险 | 防护 |
| --- | --- |
| 拉到了不同版本的作者代码 | 固定 commit；运行前验证远程、HEAD、工作树状态。 |
| 依赖版本与旧代码不兼容 | 建立独立 `.venv`；CPU 冒烟先于 GPU 长训练；固定 NumPy 1.26.4，并以外部兼容层适配作者代码的 pandas 1.x 调用，绝不修改上游文件；记录 Python、Torch、NumPy、CUDA 版本。 |
| 最后一个测试批次被丢弃 | 导出专用的同官方 Dataset DataLoader 必须 `drop_last=False`；测试检查完整 `11425` 样本。 |
| 看似比较、实际标签不同 | 比较器对形状和反标准化后的标签逐元素检查，任一差异立即失败。 |
| FFT 或周期选择偷看预测未来 | 只从每个输入历史窗口读取观测；不向 TimesNet 前向输入未来目标。 |
| 偶然单次结果被过度解读 | 明确标注单种子、单窗口长度的初步比较；后续才做 `L=336` 和多种子检查。 |

## 8. 验证策略与验收条件

### 自动化验证

1. 上游验证测试：错误 URL、错误 commit、上游有未提交改动均应拒绝执行。
2. CPU 真实模型冒烟：使用作者 TimesNet 代码，验证多变量小批次输入能完成前向，并记录形状。
3. NPZ 导出契约测试：prediction、target、columns 存在且形状一致；测试尾部小批次也被保留。
4. 比较器测试：相同标签可计算；长度、列顺序、任一真实标签不同都必须报错。
5. 共享评测回归测试：此前 Autoformer、PatchTST 的比较测试继续通过。

### 用户可验收结果

- 云端 L40 训练能以一条项目脚本命令启动，且日志清楚表明实际使用 GPU；
- 有完整 `(11425, 96, 7)` TimesNet 预测归档；
- 生成包含 TimesNet、Seasonal Naive、DLinear 的对齐指标表；Autoformer/PatchTST 若采用相同标签归档可在后续同表汇总；
- `experiment/results.md`、`handoff.md`、`brainstorm.md` 已追加，不覆盖历史实验；
- 结论只描述本项目 `ETTm1 96→96` 单次实验，不挪用论文原始表格数字。

## 9. 已确认的取舍

- **论文忠实性优先于最新维护性：** 采用作者 TSLib 的 2023-03-31 固定提交，而非已迁移为介绍页的 TimesNet 仓库或今天的 TSLib 主分支。
- **官方模型优先于自写实现：** 所有模型行为由上游代码决定；项目代码是可审计的复现与测评外壳。
- **可比性优先于直接读取官方打印指标：** 统一 NPZ、完整测试窗口与严格标签检查是模型横向结论的前提。
- **先窄后宽：** 先完成一次 `L=96 → H=96`，成功后才开展论文协议的多预测窗与 `L=336` 实验。
