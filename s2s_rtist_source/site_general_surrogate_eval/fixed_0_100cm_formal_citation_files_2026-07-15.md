# 固定 0-100 cm 正式引用候选文件清单

## 1. 建议直接引用的项目文件

以下文件已经提炼了研究问题、方法、数据范围、数值结果和结论边界，适合在汇报、论文方法说明或补充材料中直接引用。

| 文件名 | 用途 |
|---|---|
| `fixed_0_100cm_reporting_evidence_2026-07-15.md` | 固定 0-100 cm 证据链总索引，包含原始数据位置和汇报限制 |
| `fixed_0_100cm_reporting_key_numbers_2026-07-15.csv` | 参数、N2对照、五站点残差和独立重算的关键数值 |
| `fixed_0_100cm_control_volume_validation_2026-07-15.md` | 固定控制体是否合适的完整论证 |
| `three_output_rootzone_flux_frequency_validation_results_2026-07-14.md` | `NPrintDay=1/4/24` 收敛结果和动态根区移动边界项证据 |
| `three_output_fixed_0_100cm_npd24_5site_smoke_results_2026-07-15.md` | 五站点、40候选固定控制体 smoke 结果 |
| `three_output_surrogate_data_processing_spec_v1.md` | 正式字段、单位、符号和水量平衡公式定义 |

以上文件均位于：

`site_general_surrogate_eval/`

## 2. 建议引用的直接数据表

以下文件用于支撑正文中的具体数字，引用时应同时说明样本、日期、灌溉量和 `NPrintDay`。

| 文件名 | 位置 | 可支撑的结论 |
|---|---|---|
| `rootzone_flux_frequency_validation_summary_recomputed_v3.csv` | `rootzone_flux_frequency_validation_received_20260714/validation_20260714_v2/` | N2根深69到85 cm、移动项58.4250 mm、修正前后残差，以及C2频率收敛 |
| `diagnostic_daily_moving_boundary_recomputed_v3.csv` | `.../cases/code_N2_20240515_ir30_npd24/` | N2七天逐日根深变化和逐日移动边界项 |
| `three_output_smoke_validation_summary_v1.csv` | `confirmed_5site_restart_generation_smoke_v1/fixed_0_100cm_npd24_5site_smoke_20260715_v1/` | 五站点各8候选的最大和中位绝对残差 |
| `confirmed_5site_restart_generation_smoke_summary_v1.csv` | 同上 | 5/5站点完成、每站点8候选 |
| `site_restart_generation_smoke.csv` | 同上各 `P1/P2/P3/P4/P15/` 子目录 | 40个候选的固定控制深度、逐日VWC、储水、通量、AET和水量平衡残差 |
| `raw_audit_manifest.json` | 各站点 `candidate_raw_audit/.../ir_0mm` 和 `ir_60mm` | 说明10组原始端点文件的保存范围与运行参数 |

## 3. 可用于方法追溯的原始 SWAP 文件

N2动态样本目录：

`rootzone_flux_frequency_validation_received_20260714/validation_20260714_v2/cases/code_N2_20240515_ir30_npd24/`

| 文件名 | 内容 |
|---|---|
| `GmaizeDOriginal.crp` | 玉米 `RDI=10 cm`、`RRI=2.2 cm/day`、`RDC=100 cm` 原始配置 |
| `SwapOriginal.swp` | 土壤子层和compartment离散，证明100 cm为精确边界 |
| `result_forec.crp` | 决策前作物状态和根深 |
| `result_forec.vap` | 决策前土壤剖面状态 |
| `result_restart.crp` | 7天restart期间作物状态和根深 |
| `result_restart.vap` | 高频剖面含水量和100 cm边界瞬时通量 |
| `result_restart.inc` | 降水、灌溉、`Tact`、`Eact`、`Interc`和径流增量 |
| `result_restart.wba` | SWAP水量平衡辅助输出 |
| `result_restart.end` | 预测窗口末端剖面 |

这些文件是SWAP模拟输出和模型输入，不是站点观测真值。正式表述应为“基于SWAP原始输出重算”或“SWAP模拟证据”。

## 4. 代码与验证器引用

若论文或补充材料需要说明数据处理实现，可引用以下文件名：

| 文件名 | 用途 |
|---|---|
| `rootzone_flux_frequency_diagnostic_v1.py` | 高频通量积分、储水计算、动态移动项和固定控制体重算 |
| `swap_three_output_labels_v1.py` | 正式固定0-100 cm三输出标签提取 |
| `validate_three_output_smoke_v1.py` | 固定字段、深度、恒等式和候选完整性验证 |
| `tests/test_rootzone_flux_frequency_diagnostic_v1.py` | 动态/固定控制体与通量计算回归测试 |
| `tests/test_swap_three_output_labels_v1.py` | 固定0-100 cm标签回归测试 |
| `tests/test_three_output_smoke_validation_v1.py` | smoke验证规则回归测试 |

## 5. 不应正式引用的文件

| 文件名或模式 | 原因 |
|---|---|
| `diagnostic_metadata.json` | N2目录中的该文件保留修正AET聚合前的旧结果 |
| `diagnostic_summary.csv` | 无 `recomputed_v3` 后缀，属于旧版汇总 |
| `diagnostic_daily_moving_boundary.csv` | 无 `recomputed_v3` 后缀，属于旧版逐日结果 |
| `rootzone_flux_frequency_validation_summary_v1.partial.csv` | 运行中间状态，不是最终汇总 |
| `*.log`、`stdout_tail`、`stderr_tail` | 仅用于运行诊断，不是正式数据表 |
| `*.tar.gz` | 仅用于完整结果归档和校验，不应替代包内具体文件作为数值来源 |

## 6. 引用顺序建议

1. 结论与边界引用 `fixed_0_100cm_control_volume_validation_2026-07-15.md`。
2. 关键数字引用 `fixed_0_100cm_reporting_key_numbers_2026-07-15.csv`。
3. 动态根区对照引用 `rootzone_flux_frequency_validation_summary_recomputed_v3.csv`。
4. 五站点结果引用 `three_output_smoke_validation_summary_v1.csv`。
5. 被追问数据来源时，再下钻到 `result_forec.*`、`result_restart.*`、作物配置和土层配置。
