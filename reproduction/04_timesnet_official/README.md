# TimesNet 官方源码复现（ETTm1 `96 → 96`）

本目录是 [THUML/Time-Series-Library](https://github.com/thuml/Time-Series-Library) 中 TimesNet 的可审计外壳。固定作者提交为 `2665a3143dae12d1cbcc31ddd396bbff48773bce`（2023-03-31）；模型、FFT 选周期、二维卷积和训练循环都来自作者源码，本项目不重写也不修改它们。

## 云端 L40 运行

在项目根目录执行。`<run-id>` 必须是新的目录名，脚本拒绝覆盖已有运行。

```bash
git pull origin main

bash reproduction/04_timesnet_official/scripts/fetch_upstream.sh
bash reproduction/04_timesnet_official/scripts/create_gpu_env.sh cu121

bash reproduction/04_timesnet_official/scripts/run_official_ettm1_96_96_gpu.sh \
  ettm1_l96_h96_timesnet_seed2021_gpu_YYYYMMDD
```

训练期间查看作者输出：

```bash
tail -f reproduction/04_timesnet_official/runs/ettm1_l96_h96_timesnet_seed2021_gpu_YYYYMMDD/official/stdout.log
tail -n 120 reproduction/04_timesnet_official/runs/ettm1_l96_h96_timesnet_seed2021_gpu_YYYYMMDD/official/stderr.log
```

脚本依次完成：验证固定作者源码、写入实际配置与环境记录、在独立 `official/` 目录执行作者 `run.py`、读取作者生成的 `pred.npy`/`true.npy`，并在完整 11,425 个测试窗口均存在时生成 `predictions.npz`。训练失败时不会导出预测归档。

## 固定的首轮配置

使用作者 ETTm1 脚本的 `TimesNet`、`M` 多变量、`L=96`、`H=96`、`d_model=64`、`d_ff=64`、`top_k=5`、`e_layers=2`、`factor=3`，并显式记录作者脚本隐含的 `freq=h` 默认值。

`ETTm1.csv` 的实际采样间隔是 **15 分钟**。这里的 `freq=h` 是作者当时 ETTm1 脚本未传参时得到的默认时间特征编码选择，保证与该官方脚本一致；它不是把数据声称为小时采样，也不代表本项目重新定义了数据频率。

## 本地快速验证

CPU 冒烟测试只检查作者模型能以前向方式处理一个小批次，并记录作者 `FFT_for_Period` 选择的周期；它不训练模型。

```bash
bash reproduction/04_timesnet_official/scripts/fetch_upstream.sh
bash reproduction/04_timesnet_official/scripts/create_cpu_env.sh
bash reproduction/04_timesnet_official/scripts/run_smoke_cpu.sh
```

GPU 运行完成后，下一步使用 `compare_ettm1_l96_h96.py` 与 Seasonal Naive、DLinear 做严格标签对齐的比较。该比较器将在本复现外壳完成后提供；不要直接比较作者日志中的单一打印指标。
