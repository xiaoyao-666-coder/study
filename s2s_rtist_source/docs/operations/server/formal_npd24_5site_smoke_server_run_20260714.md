# 正式 NPrintDay=24 五站点 Smoke 运行说明

## 实验范围

- 站点：`P1`、`P2`、`P3`、`P4`、`P15`；
- 决策日期：`16-Jul-2024`；
- 灌溉候选：`0、10、15、20、25、30、40、60 mm`；
- 每站点 `8` 个候选，共 `40` 个候选；
- 正式输出频率：`NPrintDay=24`；
- 不训练模型，不批量生成数据；
- 确认站点工作区缺失时直接停止，不允许静默回退到通用 Maize 模板。

## 上传包

将以下文件上传到服务器项目根目录：

```text
formal_npd24_5site_smoke_bundle_20260714.tar.gz
```

服务器项目根目录：

```text
/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
```

## 解压和检查

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
tar -xzf /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/formal_npd24_5site_smoke_bundle_20260714.tar.gz
python3 -m py_compile \
  project_cli.py \
  scripts/simulation/run_confirmed_5site_restart_generation_smoke_v1.py \
  scripts/diagnostics/restart_raw_audit_v1.py \
  src/s2s_rtist/pipelines/restart_decision_dataset.py \
  src/s2s_rtist/labels/swap_three_output_labels.py \
  src/s2s_rtist/physics/rootzone_flux_frequency.py \
  src/s2s_rtist/validation/three_output_smoke.py
```

## 后台运行

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
export LD_LIBRARY_PATH="/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/local_libs/gcc_runtime/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
nohup python3 project_cli.py run confirmed-5site-smoke -- --run-id formal_npd24_5site_smoke_20260714_v1 --timeout-per-site 7200 > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/formal_npd24_5site_smoke_20260714_v1.nohup.log 2>&1 &
```

查看日志：

```bash
tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/formal_npd24_5site_smoke_20260714_v1.nohup.log
```

## 完成标志

日志末尾应列出：

```text
validation_summary: .../three_output_smoke_validation_summary_v1.csv
validation_report: .../three_output_smoke_validation_v1.md
```

正式结果目录：

```text
/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/confirmed_5site_restart_generation_smoke_v1/formal_npd24_5site_smoke_20260714_v1
```

重点查看：

```bash
cat /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/confirmed_5site_restart_generation_smoke_v1/formal_npd24_5site_smoke_20260714_v1/three_output_smoke_validation_v1.md
column -s, -t < /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/confirmed_5site_restart_generation_smoke_v1/formal_npd24_5site_smoke_20260714_v1/three_output_smoke_validation_summary_v1.csv
```

校验器检查正式字段、7 天完整性、AET 组成恒等式、直接通量组成、移动边界项、`NPrintDay=24`、实际时间梯形积分和 `Dcum` 聚合。每个站点同时保留 `0 mm` 与 `60 mm` 两个端点候选的原始 SWAP 审计文件。水量平衡残差的绝对值只做统计，不使用未经老师确认的阈值淘汰样本。

## 打包返回结果

```bash
cp /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/formal_npd24_5site_smoke_20260714_v1.nohup.log /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/confirmed_5site_restart_generation_smoke_v1/formal_npd24_5site_smoke_20260714_v1/
tar -czf /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/formal_npd24_5site_smoke_results_20260714_v1.tar.gz -C /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/confirmed_5site_restart_generation_smoke_v1 formal_npd24_5site_smoke_20260714_v1
```

返回文件：

```text
/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/formal_npd24_5site_smoke_results_20260714_v1.tar.gz
```
