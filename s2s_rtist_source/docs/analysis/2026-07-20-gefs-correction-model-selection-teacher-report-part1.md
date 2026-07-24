# 第一部分：GEFS 降水矫正模型选择

## 1. 研究问题

历史 SWAP 标签主要由 ERA5 驱动，而未来部署输入来自 GEFS 预报。若直接使用原始 GEFS，降水分布偏差会进入未来 7 天的灌溉收益、实际蒸散发和土壤湿度预测。因此先在不使用 2024 的条件下，选择一个能同时满足以下要求的降水矫正模型：

- 2015-2018 严格 OOF 无年份泄漏；
- 2019 只作为验证和超参数选择；
- 2024 不参与拟合、选择或调参；
- 七日累计误差、逐日误差、CRPS、Brier 和集合覆盖不能顾此失彼；
- 输出必须有限、非负、可追溯。

本节讨论的是 GEFS 降水矫正模型，不是后续三输出代理神经网络。

## 2. 论文依据与候选路线

候选方法来自三条文献路线：

1. Piani 等的经验 Quantile Mapping（QM）；
2. Cannon、Sobie 与 Murdock（2015）的 Quantile Delta Mapping（QDM）；
3. Shah 与 Mishra（2016，`Utility of GEFS Reforecast for Medium-Range Drought Prediction`）的七日累计两阶段线性缩放。

没有预先假定 QM 或 QDM 一定优于线性缩放，而是统一用预锁定 gate 比较。当前最终采用的是第三条路线的项目适配版本。

## 3. QM：小样本有效，扩大验证不稳定

第一版 QM 使用站点当地日边界、站点/提前期分组和乘法上尾外推。它失败后发现主要问题是 GEFS 与 ERA5 日界没有严格对齐，且乘法上尾会放大极端值。

第二版同时改为 UTC 00-24 日边界和常数加性上尾。在最初参与选择的 2019 周期上，七日 MAE 从 `15.0539` 降至 `13.5846 mm`，CRPS 从 `2.6558` 降至 `2.6530 mm`，平均 Brier 也改善；但扩大到未参与选择的 2019 周期后，七日 MAE、CRPS 和 Brier 均恶化。

随后做了 2015-2018 四折 OOF，比较全局、按站点、按提前期、月份分组和收缩版本。部分 QM 方案能改善平均 MAE 或 CRPS，但会缩窄五成员集合，强降水 P10-P90/min-max 覆盖下降，且逐年稳定性不够。因此普通 QM 没有获得进入正式天气输入的资格。

## 4. QDM：概率指标改善，但逐日天气输入变差

QDM 使用扩展后的 `2000-2019` GEFS reforecast 与对应站点 NOAA GHCN-D 降水参考，实施了：

- 当前周期因果目标 CDF；
- 乘法 QDM 分布变化保持；
- 每个成员七日总量保持；
- 将 QDM 日分配向 raw GEFS 收缩。

QDM 的优势是 CRPS、Brier 和集合覆盖改善。例如因果 QDM 的 2019 探索性验证中：

| 指标 | raw | QDM | 差值 |
|---|---:|---:|---:|
| 七日 MAE | 17.9896 | 17.9896 | 0 |
| 逐日集合均值 MAE | 4.3922 | 4.4647 | +0.0724 |
| 逐日集合均值 RMSE | 8.8107 | 9.1872 | +0.3764 |
| CRPS | 3.1027 | 3.0051 | -0.0975 |
| 平均 Brier | 0.1301 | 0.1178 | -0.0122 |

七日总量保持意味着 QDM 主要是在一周内部重新分配降水。它可以改善概率分布，却牺牲逐日集合均值 MAE/RMSE，而逐日天气正是后续 SWAP 和代理模型需要的输入。

进一步测试日分配收缩：

`P_alpha = P_raw + alpha * (P_QDM - P_raw)`

| alpha | OOF 日 MAE 差值 | OOF 日 RMSE 差值 | OOF CRPS 差值 | OOF Brier 差值 | 结果 |
|---:|---:|---:|---:|---:|---|
| 0.25 | -0.0056 | +0.0289 | -0.0193 | -0.0034 | 不晋级 |
| 0.50 | -0.0018 | +0.0698 | -0.0331 | -0.0044 | 不晋级 |
| 0.75 | +0.0050 | +0.1227 | -0.0444 | -0.0076 | 不晋级 |
| 1.00 | +0.0127 | +0.1872 | -0.0537 | -0.0088 | 不晋级 |

四个候选都因逐日 RMSE 等输入相关门槛失败。结论不是 QDM 没有概率预报价值，而是它不适合当前需要逐日确定性天气驱动的 SWAP/代理输入协议。

## 5. 最终采用：Shah-Mishra 七日累计两阶段线性缩放

项目最终采用：

`weekly_two_stage_linear_site_factor_shrink_a075`

对每个站点和 GEFS 周期先计算五成员七日累计总量，再按以下步骤计算因子：

1. 在训练期计算 GEFS 七日集合平均总量的 `q90`；
2. 对超过 `q90` 的训练周事件估计极端因子；
3. 先对强事件做极端缩放；
4. 再用全部训练周估计总体缩放因子；
5. 普通事件使用总体因子，强事件使用总体因子与极端因子的组合；
6. 同一站点、同一周期的全部成员和 7 天使用同一最终因子。

这种方法直接矫正七日总量，同时保持：

- 一周内降水时序；
- 零降水位置；
- 成员排序；
- 集合内部相对结构。

### 5.1 空间分组选择

预锁定比较了全局和按站点两种版本。2015-2018 OOF pooled 结果如下：

| 候选 | 七日 MAE 差值 | CRPS 差值 | Brier 差值 | 强事件覆盖 | 结论 |
|---|---:|---:|---:|---|---|
| global | -0.5806 mm | -0.0441 mm | -0.0011 | 失败 | 淘汰 |
| site-only | -0.7294 mm | -0.0502 mm | -0.0025 | 通过 | 进入 2019 |

全局版本平均指标改善，但强降水集合覆盖同时下降，因此不符合硬 gate。按站点版本保留了站点差异和集合覆盖。

### 5.2 因子收缩选择

原始按站点因子进一步向 raw 收缩：

`f_eff = 1 + alpha * (f_base - 1)`

| alpha | OOF 七日 MAE 差值 | OOF 日 RMSE 差值 | OOF Brier 差值 | 2019 七日 MAE 差值 | 2019 日 RMSE 差值 | 2019 Brier 差值 | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | -0.2584 | -0.0349 | -0.0015 | -0.3721 | -0.0392 | +0.0002 | 不晋级 |
| 0.50 | -0.4752 | -0.0608 | -0.0017 | -0.7266 | -0.0627 | +0.0004 | 不晋级 |
| 0.75 | -0.6291 | -0.0777 | -0.0025 | -1.0811 | -0.0703 | -0.0003 | 晋级 |
| 1.00 | -0.7294 | -0.0856 | -0.0025 | -0.9844 | -0.0620 | +0.0010 | 不晋级 |

只有 `alpha=0.75` 同时通过 2015-2018 OOF 和 2019 全部门槛，因此冻结为最终方案。`alpha=0.75` 是本项目基于验证证据做的收缩适配，不是把论文原公式擅自改写成另一种模型。

## 6. 最终结论与边界

最终不是“QM/QDM 完全错误”，而是：

- QM：小样本平均指标可能改善，但扩大周期后不稳定；
- QDM：CRPS、Brier 和集合覆盖有优势，但七日总量保持造成逐日 MAE/RMSE 恶化；
- Shah-Mishra 两阶段线性缩放：更符合七日灌溉决策窗口，保留逐日结构和集合相对关系；按站点分组并收缩到 `alpha=0.75` 后通过全部预锁定 gate。

因此当前正式 GEFS 降水矫正模型冻结为：

`weekly_two_stage_linear_site_factor_shrink_a075`

它已经用于情景一致 SWAP pilot。该 pilot 已通过天气、物理和响应覆盖 gate，但这只说明天气矫正与标签生成链路可用，不等于代理模型已经训练完成，也不等于 2024 TTA 已经执行。

## 7. 证据文件

- Shah-Mishra 两阶段线性缩放定义：`docs/superpowers/specs/2026-07-19-gefs-weekly-two-stage-linear-scaling-cv.md`
- QM 训练期 OOF 设计：`docs/superpowers/specs/2026-07-17-gefs-qm-training-period-cross-validation-design.md`
- QDM 与 GHCN-D 扩展设计：`docs/superpowers/specs/2026-07-18-gefs-qdm-expanded-station-reference-design.md`
- QDM raw 收缩 gate：`site_general_surrogate_eval/gefs_qdm_station_reference_v1/_local_gefs_qdm_raw_allocation_shrinkage_cv_v1/raw_allocation_shrinkage_candidate_gate_v1.json`
- 线性缩放 OOF gate：`site_general_surrogate_eval/gefs_qdm_station_reference_v1/_local_gefs_weekly_two_stage_linear_scaling_cv_v1/weekly_two_stage_linear_candidate_gate_v1.json`
- alpha 收缩选择：`site_general_surrogate_eval/gefs_qdm_station_reference_v1/_local_gefs_weekly_linear_factor_shrinkage_selection_v1/weekly_factor_shrinkage_selection_gate_v1.json`
- 最终冻结及 2024 诊断规范：`docs/superpowers/specs/2026-07-19-gefs-weekly-linear-final-fit-and-2024-diagnostic.md`
