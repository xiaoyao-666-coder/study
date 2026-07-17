# 固定 0-100 cm 控制体汇报证据索引

## 1. 汇报时的主结论

固定 `0-100 cm` 适合作为代理模型的水量平衡控制体，依据来自三条相互独立的证据链：作物最大根深参数、SWAP 精确空间边界、真实 SWAP 输出的水量平衡闭合。它不等同于“当前动态根区”，正式名称应为“0-100 cm 土层平均含水率”。

汇报用关键数值已集中到：

`fixed_0_100cm_reporting_key_numbers_2026-07-15.csv`

## 2. 参数和空间边界原始证据

### 2.1 玉米最大根深

原始作物配置：

`rootzone_flux_frequency_validation_received_20260714/validation_20260714_v2/cases/code_N2_20240515_ir30_npd24/GmaizeDOriginal.crp`

对应行：

- 第 366 行：`RDI = 10.00 cm`；
- 第 367 行：`RRI = 2.20 cm/day`；
- 第 368 行：`RDC = 100.00 cm`。

`RDC=100 cm` 是当前玉米配置允许达到的最大根深，因此 100 cm 不是为闭合公式临时选择的深度。

### 2.2 100 cm 精确边界

原始土壤离散配置：

`rootzone_flux_frequency_validation_received_20260714/validation_20260714_v2/cases/code_N2_20240515_ir30_npd24/SwapOriginal.swp`

第 321-327 行的前五个子层厚度为 `5 + 10 + 15 + 30 + 40 = 100 cm`，且每个 compartment 高度均为 `1 cm`。因此 100 cm 正好是子层边界，不需要截断或插值到相邻土层。

## 3. N2 动态根深样本

样本：`code_N2 / 15-May-2024 / 30 mm / NPrintDay=24`。

完整案例目录：

`rootzone_flux_frequency_validation_received_20260714/validation_20260714_v2/cases/code_N2_20240515_ir30_npd24/`

### 3.1 原始 SWAP 文件

- 决策前根深与剖面：`result_forec.crp`、`result_forec.vap`；
- 7 天 restart 根深与剖面：`result_restart.crp`、`result_restart.vap`；
- 降水、灌溉、蒸散和径流增量：`result_restart.inc`；
- SWAP 原始水量平衡辅助输出：`result_restart.wba`；
- 末时刻剖面：`result_restart.end`。

### 3.2 动态根区直接证据

正式重算表：

`rootzone_flux_frequency_validation_received_20260714/validation_20260714_v2/rootzone_flux_frequency_validation_summary_recomputed_v3.csv`

N2 的关键结果：

| 项目 | 数值 |
|---|---:|
| 根深 | 69 -> 85 cm |
| 未加移动项残差 | -58.3264 mm |
| 独立移动边界项 | +58.4250 mm |
| 加入后残差 | +0.0986 mm |

每日根深和移动项见：

`cases/code_N2_20240515_ir30_npd24/diagnostic_daily_moving_boundary_recomputed_v3.csv`

### 3.3 同一原始数据固定为 0-100 cm 的重算

使用当前 `rootzone_flux_frequency_diagnostic_v1.py`，对上述同一组原始文件调用 `analyze_case_outputs(..., control_depth_cm=100.0)`。重算结果已经写入本目录的关键数值表：

| 项目 | 数值 |
|---|---:|
| 水分输入 | 42.6996 mm |
| AET | 24.6022 mm |
| 0-100 cm 储水变化 | -38.0400 mm |
| 100 cm 边界向下净流出 | 56.2964 mm |
| 移动边界项 | 0 mm |
| 水量平衡残差 | -0.1590 mm |

核算式：

`42.6996 - 24.6022 - 56.2964 - (-38.0400) = -0.1590 mm`

这里不是把 `58.4250 mm` 加到固定根区中强行闭合。固定控制体从决策开始就统计完整 0-100 cm，边界始终不动，所以移动边界项按定义为零。

## 4. 五站点、40候选正式 smoke

完整运行目录：

`confirmed_5site_restart_generation_smoke_v1/fixed_0_100cm_npd24_5site_smoke_20260715_v1/`

### 4.1 最适合汇报的直接表格

- 分站点残差：`three_output_smoke_validation_summary_v1.csv`；
- 验证结论：`three_output_smoke_validation_v1.md`；
- 5 个站点运行完成情况：`confirmed_5site_restart_generation_smoke_summary_v1.csv`；
- 每个候选的完整固定控制体字段：`P1/P2/P3/P4/P15/site_restart_generation_smoke.csv`。

| 站点 | 候选数 | 最大绝对残差 | 中位绝对残差 |
|---|---:|---:|---:|
| P1 | 8 | 0.190152 mm | 0.087806 mm |
| P15 | 8 | 0.240092 mm | 0.094230 mm |
| P2 | 8 | 0.199031 mm | 0.174818 mm |
| P3 | 8 | 0.308267 mm | 0.268902 mm |
| P4 | 8 | 0.172981 mm | 0.113665 mm |

40 个候选的 `control_volume_type` 全部为 `fixed_0_100cm`，`control_depth_cm` 全部为 `100`，`NPrintDay` 全部为 `24`。最大绝对水量平衡残差为 `0.308267 mm`。

### 4.2 原始端点审计文件

每个站点保存 `0 mm` 和 `60 mm` 两个候选的原始端点，共 `5 x 2 = 10` 组，位置为：

`P*/candidate_raw_audit/2024/16jul2024/ir_0mm/`

`P*/candidate_raw_audit/2024/16jul2024/ir_60mm/`

每组包含 `result_restart.inc/vap/crp/wba/end`、`restart_initial.end`、`swap.swp` 和 `raw_audit_manifest.json`。其余 30 个候选保留在各站点的 `site_restart_generation_smoke.csv` 中，没有逐候选保存完整原始文件。

## 5. 本地独立重算证据

汇总报告：

`three_output_fixed_0_100cm_npd24_5site_smoke_results_2026-07-15.md`

关键检查：

| 检查项 | 最大误差 |
|---|---:|
| `storage_mm = VWC x 1000 mm` | 1.14e-13 mm |
| 每日边界深度相对 100 cm | 0 cm |
| 首末储水与 `delta_storage` | 5.68e-14 mm |
| 直接流出组成恒等式 | 0 mm |
| 原生通量与向下流出符号关系 | 0 mm |
| 水量平衡公式重构 | 1.07e-14 mm |

## 6. 结果包与完整性

### 根区频率诊断结果包

`D:/study/s2s_rtist_source/rootzone_flux_frequency_validation_results_20260714_v2.tar.gz`

SHA256：`f99dd0196826e4d46143f4bc03f159291263941950536a83c9968b0ee16e75f4`

### 固定 0-100 cm 五站点结果包

`D:/study/s2s_rtist_source/fixed_0_100cm_npd24_5site_smoke_results_20260715_v1.tar.gz`

SHA256：`4dfd618f40dec1149651dd42a66eef3c20b90e23a0f5329dce726d78d2235567`

## 7. 汇报时必须说明的边界

1. 五站点 smoke 的日期是 `16-Jul-2024`，该 7 天窗口内五站点根深变化均为 0 cm；它验证的是固定字段契约和多站点闭合，不单独证明动态生长期行为。
2. 动态根深变化由独立的 N2 样本验证，不能把五站点 smoke 和 N2 诊断说成同一批样本。
3. 根系较浅时，固定 0-100 cm VWC 会包含尚未被根系到达的深层水。已有派生比较中最大差为 `0.058320`，但目前该比较只有报告表，没有单独保存逐样本原始对照表；若老师要求逐样本证据，应重新导出后再引用。
4. 不要引用 `cases/code_N2_20240515_ir30_npd24/diagnostic_metadata.json` 或无 `recomputed_v3` 后缀的旧 summary 作为正式 AET/残差证据；它们保留的是修正 AET 聚合前的旧结果。

## 8. 现成结论报告

- 固定控制体论证：`fixed_0_100cm_control_volume_validation_2026-07-15.md`；
- 输出频率与动态移动项：`three_output_rootzone_flux_frequency_validation_results_2026-07-14.md`；
- 五站点固定 smoke：`three_output_fixed_0_100cm_npd24_5site_smoke_results_2026-07-15.md`；
- 正式数据处理定义：`three_output_surrogate_data_processing_spec_v1.md`。
