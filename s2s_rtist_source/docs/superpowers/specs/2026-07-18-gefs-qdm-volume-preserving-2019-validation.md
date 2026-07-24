# GEFS QDM 7 天水量保持 2019 验证 v1

## 冻结候选

唯一候选为 `qdm_global_7d_volume_preserving`。该候选由 2015-2018 扩展窗口 OOF gate 选出；2019 阶段不得改变 QDM 分组、零值策略、水量保持公式、阈值或 gate。

## 拟合与验证

- QDM 历史 CDF：2000-2018 GEFS reforecast 与冻结 GHCN-D 主站记录。
- QDM 目标 CDF：完整 2019 GEFS 批次，不使用 2019 GHCN-D 值。
- 2019 GHCN-D：只用于完整 7 天站点-周期评分。
- 每个站点-周期-成员的订正后 7 天总量严格保持 raw GEFS 总量。

## Gate

- 7 天 MAE 不劣于 raw；
- CRPS、平均 Brier 不劣于 raw；
- 重事件 P10-P90 与 min-max 覆盖率不能同时下降；
- 干湿发生频率误差不劣于 raw；
- 无负值、无非有限值；
- 每成员 7 天总量误差 `<=1e-8 mm`。

2019 起报日此前用于过探索性实验，因此本结果不能称为独立留出证据。若通过，下一步仍须设计不使用未来 GEFS 批次的因果目标 CDF；不得直接应用于 2024。
