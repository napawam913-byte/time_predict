# PatchTST 官方监督版复现

本目录包装官方 `yuqinie98/PatchTST` 提交 `204c21e`，不修改 `upstream/PatchTST/`。首轮实验是本项目的公平比较：ETTm1、7 变量、`L=96 → H=96`、`P=16`、`S=8`、作者种子 2021。

它不是论文表格的逐数值复刻。论文主配置 PatchTST/42 为 `L=336, P=16, S=8`；该设置将在所有基线也改为 `L=336` 后另行比较。

## 本地准备与检查

```bash
bash reproduction/03_patchtst_official/scripts/fetch_upstream.sh
bash reproduction/03_patchtst_official/scripts/create_cpu_env.sh
bash reproduction/03_patchtst_official/scripts/run_smoke_cpu.sh
```

`run_smoke_cpu.sh` 不读取 ETTm1，也不训练；它只确认官方模型将 `B×96×7` 前向为 `B×96×7`，其内部 patch 布局为 `B×7×16×12`。

## L40 GPU 训练

```bash
bash reproduction/03_patchtst_official/scripts/create_gpu_env.sh cu121
bash reproduction/03_patchtst_official/scripts/run_official_ettm1_96_96_gpu.sh \
  ettm1_l96_h96_patchtst_seed2021_gpu_20261003
```

运行目录不能重名。训练包装器会在 `runs/<run>/official/` 运行作者代码，再在 `runs/<run>/predictions.npz` 写出**标准化尺度**的 prediction、target、columns 与 scale。作者原实现只保存 `pred.npy`；本项目导出器使用同一 checkpoint 和 test loader 同时保存标签，以便进行逐元素对齐。

## 与既有基线比较

```bash
PYTHONPATH=reproduction/common:reproduction/03_patchtst_official/src \
reproduction/03_patchtst_official/.venv/bin/python \
reproduction/03_patchtst_official/scripts/compare_ettm1_l96_h96.py \
  --patchtst-run reproduction/03_patchtst_official/runs/ettm1_l96_h96_patchtst_seed2021_gpu_20261003 \
  --seasonal-naive reproduction/01_dlinear_baseline/runs/<seasonal-run>/predictions.npz \
  --dlinear reproduction/01_dlinear_baseline/runs/<dlinear-run>/predictions.npz \
  --csv DataSet/ETTm1/ETTm1.csv \
  --output-dir reproduction/03_patchtst_official/comparison
```

比较脚本先检查三个模型的变量列顺序和每一个标签值是否完全对齐；任何一个不同都会停止，不产生模型排名。通过后同时报告标准化尺度与原始尺度 MSE/MAE/RMSE，及逐变量误差。
