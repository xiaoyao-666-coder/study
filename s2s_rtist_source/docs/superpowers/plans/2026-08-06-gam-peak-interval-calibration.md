# GAM 峰值位置区间校准实施计划

> **执行方式：** 当前仓库依赖大量未提交实验文件，直接在现有 `codex/python-script-organization` 分支内联实施。严格按 TDD 执行；第一阶段通过前不实现或运行连续 SWAP。

**目标：** 实现 `2026-08-06-teacher-guided-gam-peak-interval-calibration-design.md` 的第一阶段：在 2015--2018 严格嵌套 15 折上，用低容量加权岭回归校准原四源 GAM-VC 连续峰值，生成有符号残差区间、中点推荐、八项门禁、审计、清单和可续跑逐折产物。

**架构：** 新增一个无 I/O 的校准数学模块，负责特征顺序、带截距加权岭、分组交叉验证、字典序选参、经验分位数、区间裁剪和门禁。新增训练运行器复用现有 GAM 数据读取、状态构建、外层折惩罚和连续优化接口；每个外层折内部重新生成严格源站点-年份留出样本，优先复用上一阶段外层模型，缺失或哈希不匹配时只在同一冻结训练行上重拟合。正式运行只读 2015--2018，不读取 2019/2024，不运行 SWAP。

**技术栈：** Python 3、NumPy、pandas、标准库 `unittest`、现有 `HierarchicalGamBSpline` 与精确连续优化器。

---

## 任务 1：冻结协议并锁死校准数学合同

**文件：**

- 新建：`docs/superpowers/specs/2026-08-06-teacher-guided-gam-peak-interval-calibration-development-v1.json`
- 新建：`tests/test_gam_peak_interval_calibration_v1.py`
- 新建：`src/s2s_rtist/models/gam_peak_interval_calibration_v1.py`

1. 先写失败测试，覆盖：训练子集标准化、截距不惩罚、80% 高风险权重、`gamma/lambda` 网格、字典序选参和不可变特征顺序。
2. 运行 RED：

```bash
python -m unittest tests.test_gam_peak_interval_calibration_v1 -v
```

3. 实现最小数学核心，公开接口固定为：

```python
model = fit_weighted_ridge(features, delta, baseline_error, gamma=2.0, ridge_lambda=1.0)
prediction = model.predict(features)
selection = select_hyperparameters(samples, groups, gamma_grid, lambda_grid)
```

4. 实现经验分位数和区间：

```python
q05, q95 = empirical_signed_quantiles(cross_fitted_residuals, (0.05, 0.95))
point, lower, upper, midpoint = calibrated_interval(raw_peak, predicted_delta, q05, q95)
```

5. 运行 GREEN 并提交本任务。

## 任务 2：锁死目标无关特征和严格内层切分

**文件：**

- 修改：`tests/test_gam_peak_interval_calibration_v1.py`
- 新建：`tests/test_run_gefs_gam_peak_interval_calibration_development_v1.py`
- 新建：`scripts/training/run_gefs_gam_peak_interval_calibration_development_v1.py`

1. 先写失败测试：
   - 外层目标站点永远不进入内层训练或预处理；
   - 内层验证单元为完整 `(site_id, year)`；
   - 训练年份不晚于内层验证年份；
   - 同年其他站点允许使用；
   - 三模型和四模型诊断使用同一聚合接口；
   - 特征构建对象只接收模型诊断值，不能接触 SWAP 目标列。
2. 运行 RED。
3. 实现：

```python
for held_out_site, held_out_year in inner_units:
    train = outer_train[
        (outer_train.site_id != held_out_site)
        & (outer_train.target_year <= held_out_year)
    ]
    validation = outer_train[
        (outer_train.site_id == held_out_site)
        & (outer_train.target_year == held_out_year)
    ]
```

4. 从原四源峰值、边界、margin、曲率、删一源峰值和收益聚合量构造固定特征；拒绝缺失、重复、非有限和错误维度。
5. 运行 GREEN 并提交本任务。

## 任务 3：实现单折严格嵌套校准

**文件：**

- 修改：`tests/test_run_gefs_gam_peak_interval_calibration_development_v1.py`
- 修改：`scripts/training/run_gefs_gam_peak_interval_calibration_development_v1.py`

1. 先写失败的合成单折集成测试，验证：
   - 内层样本只来自外层训练范围；
   - 选参使用按 `(site_id, year)` 分组的交叉预测；
   - 分位数来自选定参数的交叉拟合残差；
   - 最终模型使用全部内层样本；
   - 外层周期输出含全部正式列，区间合法且保留连续精度。
2. 运行 RED。
3. 实现每折流程：生成内层模型诊断 -> 选参 -> 交叉残差 -> 全量拟合 -> 外层预测 -> 原子写入。
4. 每折保存：`inner_samples.csv`、`cycle_metrics.csv`、`model_audit.csv`、`interval_audit.csv`、校准器 NPZ、选参 JSON 和带文件 SHA256 的 `fold_complete.json`。
5. 运行 GREEN 并提交本任务。

## 任务 4：实现 15 折汇总、八项门禁和断点续跑

**文件：**

- 修改：`tests/test_run_gefs_gam_peak_interval_calibration_development_v1.py`
- 修改：`scripts/training/run_gefs_gam_peak_interval_calibration_development_v1.py`

1. 先写失败测试，逐项令八个门禁单独失败并确认总门失败；补充 196/15/5 计数、重复键、非有限、区间反转、模型预测哈希变化和完成标记哈希变化拒绝测试。
2. 实现八项门禁：最大误差、P95、正参考均值、整体均值、零参考假阳性、至少 10/15 折、覆盖率、执行/隔离/多输出哈希审计。
3. 实现 `--resume`：只有完成标记、协议哈希和逐折全部文件哈希均匹配时跳过；不匹配立即失败，不静默重算。
4. 汇总固定输出文件并生成不自哈希的 manifest。
5. 运行 GREEN 并提交本任务。

## 任务 5：服务器交付与净包验证

**文件：**

- 修改：`scripts/script_catalog.csv`
- 生成：`gefs_gam_peak_interval_calibration_development_code_20260806_v1.tar.gz`

1. 服务器命令不写入项目文档，只在交付回复中一次性提供完整绝对路径命令：解压、测试、语法/JSON 检查、`nohup` 启动、PID、`tail -f`、15 折进度、异常 grep、`--resume`、结果读取和关键结果打包。
2. 明确源输入为上一阶段**完整** robust-envelope 输出目录；第一阶段失败也保留并打包，不得启动 SWAP 或读取 2019。
3. 更新脚本目录的 SHA256。
4. 在干净临时目录解包，运行目标测试、相关 GAM 回归测试、`py_compile`、`json.tool` 和包路径安全检查。
5. 选择性提交本计划涉及文件，不暂存或修改无关工作树内容。

## 完成验证

```bash
python -m unittest \
  tests.test_gam_peak_interval_calibration_v1 \
  tests.test_run_gefs_gam_peak_interval_calibration_development_v1 \
  tests.test_gefs_hierarchical_gam_bspline_v1 \
  tests.test_run_gefs_hierarchical_gam_bspline_development_v1 \
  tests.test_robust_gam_envelope_v1 \
  tests.test_run_gefs_robust_envelope_gam_development_v1 -v
```

正式 15 折及所有 SWAP 只在服务器执行；本地不读取正式数据、不声称第一阶段门禁结果。
