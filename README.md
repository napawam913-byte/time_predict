# 通用多变量时序预测：学习与复现

这是一个从零学习通用多变量长预测窗（LSTF）的可复现工作区。首轮聚焦电力、交通、天气与负荷等场景；当前以 ETTm1 的多变量 `96 -> 96` 预测建立统一协议。

仓库包含：

- `related_work/`：Informer、Autoformer、DLinear、PatchTST、TimesNet 与 iTransformer 的中文阅读笔记和来源记录；论文 PDF 与 LaTeX 原件不公开分发，按各目录的 `provenance.md` 自行获取。
- `reproduction/01_dlinear_baseline/`：Seasonal Naive 和 DLinear 的最小、可测试基线实现。
- `reproduction/02_autoformer_official/`：对 [thuml/Autoformer](https://github.com/thuml/Autoformer) 固定提交的非侵入式复现包装；作者源码会在本地重新拉取，训练输出写到被忽略的 `runs/`。
- `dataset.md` 与 `experiment/results.md`：数据来源、固定切分和已经得到的本地结果。

## 首次准备

```bash
git clone <你的 GitHub 仓库地址>
cd time_predict
mkdir -p DataSet/ETTm1
curl -L --fail --output DataSet/ETTm1/ETTm1.csv \
  https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
echo '6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e  DataSet/ETTm1/ETTm1.csv' | sha256sum -c -
bash reproduction/02_autoformer_official/scripts/fetch_upstream.sh
```

数据的 SHA-256、列、时间边界和评估协议在 [`dataset.md`](dataset.md) 中固定。请先校验下载数据的哈希，不要把 CSV、虚拟环境、模型权重或 `runs/` 提交到仓库。

## GPU 上运行 Autoformer

先根据云端 `nvidia-smi` 的 CUDA 驱动能力选择 PyTorch wheel 索引。例如 CUDA 12.1：

```bash
bash reproduction/02_autoformer_official/scripts/create_gpu_env.sh \
  https://download.pytorch.org/whl/cu121
bash reproduction/02_autoformer_official/scripts/run_official_ettm1_96_96_gpu.sh \
  ettm1_l96_h96_autoformer_seed2021_gpu
```

运行结束后，使用 `scripts/compare_ettm1_l96_h96.py` 将 Autoformer 与 Seasonal Naive、DLinear 放在同一批测试标签上比较。详细命令见 [`reproduction/02_autoformer_official/README.md`](reproduction/02_autoformer_official/README.md)。
