# 正式 NPrintDay=24 五站点 Smoke 结果

## 1. 实验范围

- 站点：`P1`、`P15`、`P2`、`P3`、`P4`；
- 决策日期：`16-Jul-2024`；
- 每站点灌溉候选：`0、10、15、20、25、30、40、60 mm`；
- 总候选数：`5 × 1 × 8 = 40`；
- SWAP 版本：`4.0.1`；
- 正式输出频率：`NPrintDay=24`；
- 通量积分：按实际 `Time` 间隔做梯形积分；
- 子日增量：按 `Dcum=1..7` 聚合；
- 不训练模型，不批量生成数据。

服务器运行目录：

```text
site_general_surrogate_eval/confirmed_5site_restart_generation_smoke_v1/formal_npd24_5site_smoke_20260714_v1
```

本地返回目录：

```text
site_general_surrogate_eval/formal_npd24_5site_smoke_received_20260714_v1/formal_npd24_5site_smoke_20260714_v1
```

## 2. 自动验收结果

- 五个站点均正常完成，返回码均为 `0`；
- `40/40` 个候选具有完整 7 天标签；
- 正式校验结果为 `Passed=True`；
- 所有候选均为 `NPrintDay=24`；
- AET 组成恒等式最大误差为 `0 mm`；
- 水量平衡公式重构最大误差为 `0 mm`；
- 10 组原始审计留档全部存在：每站点保存 `0 mm` 和 `60 mm` 两个端点候选；
- 确认站点工作区未发生通用 Maize 模板回退。

## 3. 站点结果

| 站点 | 最佳灌溉量 | 最佳收益标签 | 最大绝对残差 | 中位绝对残差 | 最大根深变化 | 最大移动边界项 |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 25 mm | 211.0 | 0.1902 mm | 0.0878 mm | 0 cm | 0 mm |
| P15 | 30 mm | 223.4 | 0.2401 mm | 0.0942 mm | 0 cm | 0 mm |
| P2 | 0 mm | 0.0 | 0.1990 mm | 0.1748 mm | 0 cm | 0 mm |
| P3 | 0 mm | 0.0 | 0.3083 mm | 0.2689 mm | 0 cm | 0 mm |
| P4 | 10 mm | 54.6 | 0.1730 mm | 0.1137 mm | 0 cm | 0 mm |

完整数值见 `three_output_formal_npd24_5site_smoke_summary_2026-07-14.csv`。

## 4. 最大残差样本

最大绝对残差出现在：

```text
P3 / 16-Jul-2024 / 10 mm
water_balance_residual_7d_mm = -0.308267 mm
```

该样本主要水量项为：

```text
rain                       = 3.7999 mm
irrigation                 = 10.0008 mm
AET                        = 34.2961 mm
delta root-zone storage    = -18.4600 mm
direct residual flux       = -1.727133 mm
moving-boundary term       = 0 mm
```

负的直接通量表示该窗口根区边界为净向上通量，不是符号错误。全部 40 个候选的最大绝对残差为 `0.308267 mm`，中位绝对残差为 `0.153387 mm`；相对于各候选水量周转量，最大比例约为 `0.451%`，中位比例约为 `0.182%`。本次不设置老师尚未确认的残差淘汰阈值。

## 5. 移动边界解释

五站点全部候选均满足：

```text
max root-depth change within horizon = 0 cm
max predecision-to-daily root-depth change = 0 cm
moving_root_boundary_term_7d_mm = 0 mm
```

因此移动边界项为零是固定根区结果，不是漏算。该 smoke 扩展验证了 `NPrintDay=24` 在五站点、40 个固定根区候选中的时间积分稳定性，但没有重新覆盖动态根区情形。动态根区结构仍由 `code_N2 / 15-May-2024 / 30 mm` 诊断验证：移动边界项 `58.4250 mm` 将残差从 `-58.3264 mm` 修正到 `0.0986 mm`。

## 6. 正式结论

1. `NPrintDay=24`、实际时间梯形积分和 `Dcum` 聚合已通过五站点 smoke；
2. 直接物理通量标签、水量平衡审计字段和独立移动边界项均能稳定生成；
3. 正式标签生成链路可继续进入后续小批量数据准备；
4. 本结果不授权直接开始 2015-2019 全量生成或模型训练；GEFS 协议和后续实验范围仍按老师确认流程执行。
