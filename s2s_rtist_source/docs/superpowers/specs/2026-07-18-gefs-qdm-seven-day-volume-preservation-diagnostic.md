# GEFS QDM 7 天水量保持诊断 v1

## 目的

2000-2018 扩展窗口交叉验证表明，QM/QDM 普遍改善日尺度 CRPS 和 Brier，但恶化 7 天累计降水 MAE。该诊断检验：在不改变 raw GEFS 每个集合成员 7 天总降水的前提下，QDM 对逐日降水分配、发生频率和概率分布是否仍有价值。

## 输入边界

- 只读取冻结的 2015-2018 扩展窗口 OOF 预测。
- 不重新拟合 QM/QDM。
- 不使用 2019、2024 或任何新观测。
- 不进行网络下载或代理模型训练。

## 变换

对每个 `site_id + decision_date + gefs_member` 的完整 7 天窗口：

1. `S_raw = sum(P_raw[d])`
2. `S_qdm = sum(P_qdm[d])`
3. 若 `S_raw = 0`，则 7 天输出全部为 0。
4. 若 `S_raw > 0` 且 `S_qdm > 0`，则 `P_vp[d] = P_qdm[d] * S_raw / S_qdm`。
5. 若 `S_raw > 0` 且 `S_qdm = 0`，则回退为原始逐日序列 `P_raw[d]`。

最后把浮点残差加到订正量最大的非零日，使输出 7 天总量与 raw 严格一致。该方法不使用参考降水，因此不会泄漏验证观测。

## 候选

- `qdm_global_7d_volume_preserving`
- `qdm_site_only_7d_volume_preserving`

## Gate

- 每个成员每周期 7 天总量绝对误差 `<=1e-8 mm`；
- 无负值、无非有限值；
- pooled 7 天 MAE 不劣于 raw（理论上应持平）；
- pooled CRPS、Brier 不劣于 raw；
- 重事件 P10-P90 与 min-max 覆盖率不能同时下降；
- pooled 干湿发生频率误差不劣于 raw；
- 7 天 MAE、CRPS、Brier 分别至少 `3/4` 年不劣于 raw。

本诊断若通过，只能支持冻结一个新候选用于 2019 探索性验证；不能直接用于 2024 实时部署。
