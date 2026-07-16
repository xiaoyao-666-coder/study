# 根区通量输出频率与移动边界诊断结果

日期：2026-07-14

## 1. 诊断范围

老师批准的诊断严格限定为：

```text
code_C2 / 16-Jul-2024 / 30 mm
code_C2 / 16-Jul-2024 / 60 mm
code_N2 / 15-May-2024 / 30 mm
NPrintDay = 1、4、24
```

共完成 `9` 个 restart 案例。所有案例的 SWAP 日志均包含 `normal completion`。没有训练模型，也没有批量生成正式数据。

## 2. 计算方法

### 2.1 根区边界通量

`result_restart.vap` 中 `waterflux` 为瞬时速率 `cm/day`，原生符号为正向上、负向下。累计向下净流出采用实际子日间隔的梯形积分：

```text
root_boundary_outflow_7d_mm
= -10 × trapezoid_integral(waterflux, time)
```

每个案例包含一个决策前初始剖面和未来 7 天剖面：

| NPrintDay | 剖面样本数 | 时间间隔 |
|---:|---:|---:|
| 1 | 8 | 1 day |
| 4 | 29 | 0.25 day |
| 24 | 169 | 1/24 day |

### 2.2 高频 `result.inc` 聚合

高频 `result.inc` 的午夜行已经使用下一日历日期，但 `Day/Dcum` 仍属于前一个模拟日。例如 `2024-07-17 00:00:00` 的 `Dcum=1` 属于第一个预测日。

因此，降水、灌溉、AET 和径流等增量必须按 `Dcum=1...7` 汇总，不能直接按 `Date` 分组。服务器首次生成的 v2 汇总 CSV 曾按日历日期聚合，高频 AET 和残差存在末时段遗漏；本文和正式汇总 CSV 使用本地复算的 v3 数值。

### 2.3 移动边界项

N2 动态根区逐日计算：

```text
moving_root_boundary_term_day_mm
= 10 × mean(theta_before, theta_after) × root_depth_change_day_cm
```

7 天累计为每日新增根层储水量之和。该项单独加入动态控制体平衡，不并入实际水分流出。

## 3. 固定根区结果

| 样本 | NPrintDay | 根区边界向下流出 | 平衡反推流出 | 平衡残差 |
|---|---:|---:|---:|---:|
| C2 / 30 mm | 1 | 17.1285 mm | 25.6860 mm | 8.5575 mm |
| C2 / 30 mm | 4 | 25.3047 mm | 24.0122 mm | -1.2925 mm |
| C2 / 30 mm | 24 | 23.7116 mm | 23.5561 mm | -0.1555 mm |
| C2 / 60 mm | 1 | 50.7215 mm | 54.2097 mm | 3.4882 mm |
| C2 / 60 mm | 4 | 48.9146 mm | 52.6720 mm | 3.7574 mm |
| C2 / 60 mm | 24 | 53.4105 mm | 53.2712 mm | -0.1393 mm |

结论：

- `NPrintDay=1` 的每日一次瞬时采样不能可靠表示 7 天累计根区边界通量；
- `NPrintDay=4` 在 30 mm 样本中有所改善，但 60 mm 样本仍有 `3.76 mm` 残差，不能视为稳定收敛；
- `NPrintDay=24` 的两个固定根区样本均闭合到 `0.16 mm` 以内；
- 所有固定根区样本的空间边界误差为 `0 cm`，异常不是空间边界选择造成的。

原始 `7.6426 mm` 来自每天一次瞬时值的右端矩形求和；本诊断统一改用包含初始时刻的梯形积分，因此 `NPrintDay=1` 对应残差为 `8.5575 mm`。这两个数不能直接相减解释为某个单一物理水量。

可以确认的是：提高时间分辨率后，同一运行内部的直接通量与水量平衡从毫米级不闭合降至约 `0.15 mm`，因此原固定根区异常主要来自瞬时通量时间采样和积分不足。

## 4. 动态根区结果

| NPrintDay | 根深起点/终点 | 直接向下流出 | 未加移动项残差 | 移动边界项 | 修正后残差 |
|---:|---:|---:|---:|---:|---:|
| 1 | 69/84 cm | 72.3320 mm | -66.6825 mm | 55.3500 mm | -11.3325 mm |
| 4 | 69/85 cm | 62.5918 mm | -62.5878 mm | 58.6550 mm | -3.9328 mm |
| 24 | 69/85 cm | 59.3838 mm | -58.3264 mm | 58.4250 mm | 0.0986 mm |

`NPrintDay=24` 时：

```text
未加移动边界项残差 = -58.3264 mm
移动边界项          = +58.4250 mm
修正后残差          = +0.0986 mm
```

这确认动态根区的大残差主要来自根系生长时新纳入土层携带的储水，而不是约 `58 mm` 的未记录实际排水。

移动边界项对应新纳入 `16 cm` 土层的平均体积含水率约为：

```text
58.425 / (16 × 10) = 0.3652
```

处于物理合理范围。

## 5. 正式采用方案

基于这三个诊断样本：

1. 当前 `NPrintDay=1` 通量标签不能作为正式监督标签；
2. `NPrintDay=4` 尚不足以保证高通量样本闭合；
3. 正式通量标签采用 `NPrintDay=24`；
4. `waterflux` 使用原生符号保存，并使用实际 `Time` 间隔做梯形积分；
5. `result.inc` 必须按 `Dcum` 汇总子日增量；
6. 动态根区移动边界项必须独立计算和保存，不能并入普通排水通量；
7. 正式采用“直接物理通量监督 + 移动边界项独立计算”的结构，水量平衡反推量不替代直接物理通量标签；
8. 在批量生成前，使用正式 `NPrintDay=24` 配置做小规模多站点 smoke，并评估输出体积、运行成本和跨样本闭合稳定性。

## 6. 数据与代码

- 正式复算汇总：`three_output_rootzone_flux_frequency_validation_summary_2026-07-14.csv`；
- 原始返回目录：`site_general_surrogate_eval/rootzone_flux_frequency_validation_received_20260714/validation_20260714_v2`；
- 诊断分析模块：`src/s2s_rtist/physics/rootzone_flux_frequency.py`（原 `rootzone_flux_frequency_diagnostic_v1.py`）；
- 服务器运行器：`scripts/diagnostics/run_rootzone_flux_frequency_validation_v1.py`（CLI id: `rootzone-frequency`）；
- 回归测试：`tests/test_rootzone_flux_frequency_diagnostic_v1.py`。
