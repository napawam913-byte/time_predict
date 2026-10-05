# 上游来源

TimesNet 的公开仓库 `THUML/TimesNet` 已说明完整代码与脚本位于作者维护的
`THUML/Time-Series-Library`。本复现固定后者的论文时期提交
`2665a3143dae12d1cbcc31ddd396bbff48773bce`（2023-03-31）。

`../upstream/Time-Series-Library/` 是可重新获取的外部输入：不会提交到本项目，
也不得在其中加入兼容代码或改动模型。运行 `scripts/fetch_upstream.sh` 后，
`scripts/verify_upstream.sh` 会检查 origin、提交、工作树洁净状态和所需文件。
