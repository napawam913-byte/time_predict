# 电力碳因子 Persistence 独立运行说明

本包是公开研究案例的最小运行包：一个预测脚本、一份冻结 CSV 和本说明。
它不包含完整取数、情景配置、数据生成及开发测试管线。
运行预测不会下载数据、训练网络或访问网络。

## 环境和文件

需要 Python 3.10 或更高版本，仅使用标准库；CPU 即可，无需 GPU 或安装依赖。
从仓库项目根目录执行，以下路径均相对于该目录：

- 脚本：`research/01_park_multienergy_carbon_factor/scripts/persistence_forecast.py`
- 数据：`research/01_park_multienergy_carbon_factor/runs/park_e_g_h_v0_1_20261008/history_hourly.csv`

脚本的默认数据路径根据脚本自身位置计算，不依赖当前工作目录。
命令中的输出路径则相对于当前工作目录，因此建议从项目根执行。
冻结 CSV 的 SHA-256 为：

```text
ca206f709fe33dffe9785e01656927c8dcfd07dcba541250d35026645fa4a200
```

CLI 默认检查该哈希，不修改输入；哈希不一致会拒绝运行。

## 首次运行：L168 → H24 验证集

在仓库项目根目录执行：

```bash
python3 research/01_park_multienergy_carbon_factor/scripts/persistence_forecast.py \
  --split validation \
  --lookback 168 \
  --horizon 24 \
  --output-dir research/01_park_multienergy_carbon_factor/runs/electricity_persistence_l168_h24_validation_v1
```

完成后查看输出目录的 `summary.md`、`audit.json` 和 `metrics.json`。
`first_window.svg` 可在浏览器中打开；它是首个时间窗口，不代表整体精度。
输出还包含预测 CSV、首窗 CSV、配置、脚本快照和文件哈希清单。
已有同名输出目录会拒绝运行；重跑请改用新的目录名，例如 `_v2`。
失败后可能保留部分文件及失败标记，不能据此当作成功结果。

## 数据性质与来源

数据覆盖 2023 年 UTC 的 8760 个小时，其中 49 个小时电因子缺失。
电因子来自 NESO Carbon Intensity API 的 Great Britain national `intensity.actual`。
该字段是估计实际碳强度（estimated actual），不是仪器实测或中国电网因子。
原生半小时值由 gCO₂/kWh 转为 kgCO₂/kWh；两段均有效才形成小时值。
小时聚合假设合成小时电量在两段半小时均分，缺失不补零或替换为 forecast。
园区用电、用气、热需求及效率等是合成情景，不是实测园区数据。

- [NESO 官方 API](https://api.carbonintensity.org.uk/)
- [官方 API 文档](https://carbon-intensity.github.io/api-definitions/)
- [官方许可与条款](https://github.com/carbon-intensity/terms)
- [IPCC 2006 固定燃烧表 2.3](https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_2_Ch2_Stationary_Combustion.pdf)
- [GHG Protocol Scope 2 Guidance 附录 A](https://ghgprotocol.org/sites/default/files/ghgp/standards/Scope%202%20Guidance_Final_0.pdf)

已有来源声明：NESO 电力源数据按 CC BY 4.0 使用，保留 NESO 署名与上述来源。
此来源许可声明不等于为脚本或整个混合数据包另行授予许可。
天然气采用 IPCC 默认值 56.1 kgCO₂/GJ，基础情景恒定；其他情景系数为模拟假设。
自产热排放是锅炉燃气排放的分配量，不能再与全部燃气排放相加计总排放。

## 时间安全与结果口径

严格按 UTC 标签时间切分：1—8 月训练、9—10 月验证、11—12 月测试。
所有预测目标必须落在所选切分内，历史可来自之前时段；本命令只运行验证集。
每小时一个起点，回看前 168 个日历小时，预测未来 24 小时。
Persistence 只取回看区间内最新有效且已可获得的电因子，并重复到各预测步。
电因子在 `period_end + 1h` 可用是模拟信息集假设，不是 NESO 发布保证。
归档历史也不能证明相应版本在当时已发布，因此这是快照上的回溯基准。
输入必须满足 `available_at <= origin`；无可用值时预测为空，合法零值保留。
未来实际需求、效率、合成扰动和通过种子重放得到的未来隐藏量不可作为输入。

目标与预测有效性分别记录，评分只取两者交集，并报告覆盖率。
指标为原始尺度 MAE、RMSE、MSE 和平均有符号偏差，不报告 MAPE。
本配置预期 1441 个起点、34584 个输出位置、1104 个缺失目标和 33480 个评分位置。
46 个起点会使用额外陈旧的输入；这些数量用于核对协议，不是模型优劣结论。
正常状态为 `completed`；`completed_without_scores` 的 null 指标不能解释为零误差。
重叠窗口会重复评分同一小时，不能把预测行直接累加为园区总排放。
