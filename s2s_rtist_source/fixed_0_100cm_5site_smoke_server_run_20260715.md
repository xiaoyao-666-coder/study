# 固定 0–100 cm 五站点 Smoke 运行说明

本次只重新运行已确认的五站点、一个决策日和每站点 8 个灌溉候选，用于验证固定 `0–100 cm` 新字段契约。不会训练模型，不会批量生成 2015–2019 数据，也不会下载 GEFS。

## 范围

```text
sites: P1, P2, P3, P4, P15
decision_date: 16-Jul-2024
irrigation_options_mm: 0,10,15,20,25,30,40,60
NPrintDay: 24
control_volume_type: fixed_0_100cm
control_depth_cm: 100
```

## 验证器必须检查

- 每个候选恰好 7 天；
- `soil_vwc_0_100cm` 与固定 100 cm 储水量一致；
- 每日土壤边界深度均为 100 cm；
- 原生有符号通量与向下净流出符号相反；
- 日累计等于 7 天累计；
- 水量平衡公式可重构；
- 不存在 `rootzone_*` 或 `moving_root_boundary_*` 正式字段；
- 两个端点候选保留原始 SWAP 审计文件。

## 运行标识

建议使用：

```text
fixed_0_100cm_npd24_5site_smoke_20260715_v1
```

输出目录：

```text
site_general_surrogate_eval/confirmed_5site_restart_generation_smoke_v1/
fixed_0_100cm_npd24_5site_smoke_20260715_v1
```

## 完成状态

服务器运行已完成：5 个站点、40 个候选全部通过固定控制体验证器，最大绝对水量平衡残差为 `0.308267 mm`。结果说明见 `site_general_surrogate_eval/three_output_fixed_0_100cm_npd24_5site_smoke_results_2026-07-15.md`。
