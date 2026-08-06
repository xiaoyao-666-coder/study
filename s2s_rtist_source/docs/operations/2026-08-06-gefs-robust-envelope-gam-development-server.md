# 稳健下包络 GAM 第一阶段服务器运行说明

日期：2026-08-06

## 实验边界

- 只使用 2015--2018 年开发数据，严格留站点并滚动留年份，共 15 折、196 个验证周期。
- 每折拟合 1 个四源站 GAM-VC 基线和 4 个删一源站三源站 GAM-VC 子模型，共 75 个模型。
- 四个子模型使用旧嵌套 GAM-VC 已冻结的折级惩罚值，不重新搜索惩罚。
- 连续推荐最大化四条相对零灌溉增量收益曲线的解析下包络；0.25 mm 密网格只做诊断。
- 2019 不读取特征和目标，2024 完全不读取；不运行 SWAP、MoE、TTA，也不自动推广模型。
- 只有第一阶段七项门禁全部通过，才允许设计后续 2015--2018 配对连续 SWAP 协议。

## 解压和包内核验

用户把代码包传到服务器根目录后执行：

```bash
sha256sum /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_robust_envelope_gam_development_code_20260806_v1.tar.gz
tar -tzf /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_robust_envelope_gam_development_code_20260806_v1.tar.gz
tar -xzf /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_robust_envelope_gam_development_code_20260806_v1.tar.gz -C /media/data_hot/lzx_projs/soil_moisture_otw
```

## 包内测试和语法检查

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source

env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -m unittest tests.test_robust_gam_envelope_v1 tests.test_run_gefs_robust_envelope_gam_development_v1 -v

env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -m unittest tests.test_gefs_hierarchical_gam_bspline_v1 tests.test_run_gefs_hierarchical_gam_bspline_development_v1 tests.test_gefs_hierarchical_gam_continuous_qualification_v1 -v

python3 -m py_compile /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src/s2s_rtist/models/robust_gam_envelope_v1.py /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py

python3 -m json.tool /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json > /dev/null
```

## 启动第一阶段开发实验

输出目录为新目录；已有目录时不要覆盖，改用后面的续跑命令。

```bash
nohup env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -u /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py --protocol /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json --dataset-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_exact_schedule_surrogate_dataset_v1 --source-development-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_hierarchical_gam_nested_shared_selection_development_v2 --output-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1 > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log 2>&1 & echo $! > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.pid
```

## 运行检查

```bash
ps -fp "$(cat /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.pid)"

tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log

grep -c '^fold=' /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log

grep -nE 'Traceback|Error|Exception|failed|nonfinite|leak|hash changed' /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log
```

正常结束时日志最后应有 `fold=15/15` 和 `development_complete folds=15 cycles=196`。`tail -f` 按用户习惯使用，查看完按 `Ctrl+C` 退出查看，不会停止后台进程。

## 中断后续跑

只有输出目录已经存在且其中部分折已生成时才加 `--resume`：

```bash
nohup env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -u /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py --protocol /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json --dataset-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_exact_schedule_surrogate_dataset_v1 --source-development-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_hierarchical_gam_nested_shared_selection_development_v2 --output-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1 --resume > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1_resume_001.log 2>&1 & echo $! > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1_resume_001.pid

tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1_resume_001.log
```

## 查看结果和汇总

```bash
python3 -m json.tool /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_development_audit_v1.json

python3 -m json.tool /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_development_gate_v1.json

python3 - <<'PY'
import json
import pandas as pd

audit = json.load(open("/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_development_audit_v1.json", encoding="utf-8"))
gate = json.load(open("/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_development_gate_v1.json", encoding="utf-8"))
print("status =", audit["status"])
print("next_gate =", audit["next_gate"])
print("gate_passed =", gate["passed"])
print("conditions =", gate["conditions"])
print("observed =", gate["observed"])
print("\nfold summary:")
print(pd.read_csv("/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_fold_metrics_v1.csv").to_string(index=False))
print("\nsite summary:")
print(pd.read_csv("/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_site_metrics_v1.csv").to_string(index=False))
print("\ncycle key columns:")
cycle = pd.read_csv("/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_cycle_metrics_v1.csv")
print(cycle[[
    "fold_id", "target_site", "decision_date",
    "true_fixed_list_irrigation_mm", "baseline_continuous_irrigation_mm",
    "robust_continuous_irrigation_mm", "baseline_peak_absolute_error_mm",
    "robust_peak_absolute_error_mm", "robust_dense_diagnostic_gain_gap",
]].head(20).to_string(index=False))
PY
```

## 关键结果打包

只打包审计、门禁、逐周期/逐折/逐站点原始表、模型审计、预测指标、清单和日志，不打包全部折工作目录：

```bash
tar -czf /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_robust_envelope_gam_development_key_results_20260806_v1.tar.gz \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_cycle_metrics_v1.csv \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_fold_metrics_v1.csv \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_site_metrics_v1.csv \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_model_audit_v1.csv \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_prediction_metrics_v1.csv \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_development_gate_v1.json \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_development_audit_v1.json \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1/gefs_robust_envelope_development_manifest_v1.json \
  /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log

sha256sum /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_robust_envelope_gam_development_key_results_20260806_v1.tar.gz
tar -tzf /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_robust_envelope_gam_development_key_results_20260806_v1.tar.gz
```

门禁失败时也要打包结果；失败结果只允许作为本路线负证据，不能继续读取 2019 或启动第二阶段 SWAP。
