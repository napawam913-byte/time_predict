# Autoformer 官方代码复现

本目录复现 `thuml/Autoformer` 在 ETTm1 上的多变量 `96 -> 96` 预测。`upstream/Autoformer` 只存放作者的固定 Git 快照，不编辑其被跟踪源码；本目录的脚本从各自唯一的 `runs/<run-id>/` 中调用该源码，因此训练输出不会污染上游目录。

第一轮结果会与项目已有的 Seasonal Naive 和 DLinear 结果比较。比较器先验证三者的测试标签逐窗一致，然后同时报告标准化尺度和原始尺度的 MSE、MAE、RMSE，以及每变量误差。

详细设计见 [`../../docs/superpowers/specs/2026-10-03-autoformer-official-reproduction-design.md`](../../docs/superpowers/specs/2026-10-03-autoformer-official-reproduction-design.md)。

## 不可变源码边界

```bash
bash scripts/fetch_upstream.sh
bash scripts/verify_upstream.sh
```

两条命令都会验证上游 Git 工作树干净、远程地址为作者仓库、ETTm1 CSV 存在，并在 `provenance/upstream.json` 中记录 commit 和 SHA-256。任何上游源码改动都会使训练被拒绝。

## 当前运行时兼容层

作者代码使用的 Pandas `drop(labels, axis)` 和 NumPy `np.Inf` 在当前 Pandas 2 / NumPy 2 中已经移除。`compat/sitecustomize.py` 仅在运行包装中通过 `PYTHONPATH` 加载，恢复这两个 API；它不修改作者 Git 快照、不改变模型结构、数据切分或指标逻辑。正式结果会标记为当前 Python 3.12 / PyTorch 2.5 CPU 的兼容环境运行。

## 云端 GPU 运行

本机 CPU 冒烟运行已确认作者入口能够开始训练，但完整 ETTm1 `96 -> 96` 训练不适合在当前笔记本上执行。云端从全新的虚拟环境开始，不上传本机 `.venv`、数据或 `runs/`：

```bash
# 项目根目录：先取得数据和固定的作者源码
curl -L --fail --output DataSet/ETTm1/ETTm1.csv \
  https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
bash reproduction/02_autoformer_official/scripts/fetch_upstream.sh

# 用 nvidia-smi 确认 GPU；本例安装 CUDA 12.1 对应的 PyTorch 2.5.1 wheel
bash reproduction/02_autoformer_official/scripts/create_gpu_env.sh \
  https://download.pytorch.org/whl/cu121

# 一次正式的官方 ETTm1、多变量、L=96、H=96 运行
bash reproduction/02_autoformer_official/scripts/run_official_ettm1_96_96_gpu.sh \
  ettm1_l96_h96_autoformer_seed2021_gpu
```

`create_gpu_env.sh` 会在 PyTorch 不能识别 CUDA GPU 时失败，避免误在 CPU 上启动耗时训练。若云端 CUDA 运行时与示例不兼容，请依据 `nvidia-smi` 和 [PyTorch Start Locally](https://pytorch.org/get-started/locally/) 选择相应的 wheel 索引；模型参数和训练协议不需要改动。
