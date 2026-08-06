# 稳健下包络 GAM 第一阶段实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在完全不读取 2019/2024、不开启新惩罚搜索和不运行新 SWAP 的前提下，实现“删一源站三站点 GAM-VC 集成 + 最差增量收益下包络连续优化”，并在既有 2015--2018 严格留站点滚动年份的 15 折、196 个验证周期上完成预声明开发门禁。

**Architecture:** 复用已经通过门禁的嵌套 GAM-VC 数据划分、状态构造、拟合器和惩罚选择结果。每个外层折拟合一个四源站基线模型及四个删一源站子模型；新建纯数学模块，在所有样条区间边界、单曲线驻点和两曲线交点组成的完备候选集上，精确最大化四条相对零灌溉增量收益曲线的下包络。运行器只生成开发期预测、曲线诊断、指标、审计和门禁；第一阶段通过后才另行设计开发期连续 SWAP 第二阶段。

**Tech Stack:** Python 3.10、NumPy、pandas、标准库 `unittest`、现有 `HierarchicalGamBSpline`/GAM-VC、JSON/CSV 审计输出。

---

## 文件结构

- 新建 `src/s2s_rtist/models/robust_gam_envelope_v1.py`：纯数学的分段三次下包络连续求解器，以及面向 GAM 模型的增量收益适配函数。
- 新建 `scripts/training/run_gefs_robust_envelope_gam_development_v1.py`：协议校验、15 折恢复、冻结惩罚读取、四个删一源站模型拟合、196 周期评估、门禁、审计和断点续跑。
- 新建 `tests/test_robust_gam_envelope_v1.py`：解析候选完备性、交点折点、边界解、并列解、密网格仅诊断等单元测试。
- 新建 `tests/test_run_gefs_robust_envelope_gam_development_v1.py`：协议、数据隔离、删站拟合、指标和门禁测试。
- 新建 `docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json`：第一阶段冻结协议。
- 新建 `docs/operations/2026-08-06-gefs-robust-envelope-gam-development-server.md`：服务器测试、运行、`tail -f`、结果查看、汇总和打包命令，全部使用绝对路径。
- 修改 `scripts/script_catalog.csv`：登记新运行器。

## 第一阶段固定输出

正式输出目录固定为：

`site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1`

必须生成：

- `gefs_robust_envelope_cycle_metrics_v1.csv`：196 行逐周期结果。
- `gefs_robust_envelope_fold_metrics_v1.csv`：15 行逐折结果。
- `gefs_robust_envelope_site_metrics_v1.csv`：5 行逐站点结果。
- `gefs_robust_envelope_model_audit_v1.csv`：至少 75 行模型审计，即 15 个四源站基线模型和 60 个三源站子模型。
- `gefs_robust_envelope_prediction_metrics_v1.csv`：基线与四子模型均值的多输出预测指标。
- `gefs_robust_envelope_development_gate_v1.json`：七项预声明门禁。
- `gefs_robust_envelope_development_audit_v1.json`：数据隔离、哈希、计数和禁止事项审计。
- `gefs_robust_envelope_development_manifest_v1.json`：输出文件 SHA256。

### Task 1：冻结第一阶段协议

**Files:**

- Create: `docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json`
- Create: `tests/test_run_gefs_robust_envelope_gam_development_v1.py`
- Create: `scripts/training/run_gefs_robust_envelope_gam_development_v1.py`

- [ ] 先在测试中定义协议的不可变要求：开发年份为 2015--2018，2019 仅记录为已查看背景年份，2024 封存；5 站点乘 3 滚动年份共 15 折；验证周期总数 196，站点计数固定为 P1=38、P2=38、P3=41、P4=39、P15=40；每折四个删一源站子模型；不增加惩罚搜索；不运行 SWAP、MoE、TTA 或模型推广。

- [ ] 写入以下失败测试，确认运行器尚不存在或协议校验尚未实现时失败：

```python
class RobustEnvelopeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).resolve().parents[1] / (
            "docs/superpowers/specs/"
            "2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json"
        )
        self.protocol = json.loads(path.read_text(encoding="utf-8"))

    def test_frozen_protocol_validates(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(self.protocol["validation"]["strict_outer_fold_count"], 15)
        self.assertEqual(self.protocol["validation"]["validation_cycle_count"], 196)
        self.assertEqual(self.protocol["ensemble"]["jackknife_model_count_per_fold"], 4)

    def test_protocol_rejects_2019_training(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["forbidden"]["2019_rows_read_for_training_or_selection"] = False
        with self.assertRaisesRegex(ValueError, "2019"):
            validate_protocol(changed)

    def test_protocol_rejects_new_penalty_search(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["penalties"]["new_search_performed"] = True
        with self.assertRaisesRegex(ValueError, "penalty"):
            validate_protocol(changed)
```

- [ ] 运行测试并确认因缺少模块或 `validate_protocol` 而失败：

```powershell
python -m unittest tests.test_run_gefs_robust_envelope_gam_development_v1 -v
```

预期：`FAILED` 或导入错误，且失败原因指向新运行器尚未实现。

- [ ] 创建协议 JSON，明确写入以下内容：

```json
{
  "protocol_id": "teacher-guided-robust-envelope-gam-development-v1",
  "frozen_at": "2026-08-06",
  "data": {
    "development_years": [2015, 2016, 2017, 2018],
    "reused_context_year": 2019,
    "sealed_final_test_year": 2024,
    "sites": ["P1", "P2", "P3", "P4", "P15"],
    "candidate_count_per_cycle": 8,
    "dataset_sha256": "3470776d0fa29928b27e0099fc5a84d6fc9f8a84c1c778b1def6325f80fd2917",
    "contract_sha256": "b523800533c28b7dc25e682c0e6dd842cd406c41b710b81db4e8ca6901b46880"
  },
  "source_development": {
    "protocol_id": "teacher-guided-hierarchical-gam-nested-shared-selection-development-v2",
    "formal_variant": "GAM-VC",
    "must_have_passed_gate": true,
    "reuse_selected_penalties_without_search": true
  },
  "validation": {
    "rolling_year_folds": [
      {"train_years": [2015], "validation_year": 2016},
      {"train_years": [2015, 2016], "validation_year": 2017},
      {"train_years": [2015, 2016, 2017], "validation_year": 2018}
    ],
    "strict_outer_fold_count": 15,
    "validation_cycle_count": 196,
    "site_cycle_counts": {"P1": 38, "P2": 38, "P3": 41, "P4": 39, "P15": 40},
    "held_out_site_uses_shared_terms_only": true
  },
  "ensemble": {
    "jackknife_model_count_per_fold": 4,
    "source_site_count_per_submodel": 3,
    "decision_target": "minimum_incremental_net_gain_across_submodels",
    "other_output_aggregation": "arithmetic_mean"
  },
  "continuous_optimization": {
    "range_mm": [0.0, 60.0],
    "deployment_resolution_mm": 0.000001,
    "dense_diagnostic_step_mm": 0.25,
    "maximum_dense_gain_gap": 0.05,
    "candidate_sources": ["interval_boundaries", "stationary_points", "pairwise_intersections"],
    "dense_grid_used_for_recommendation": false,
    "tie_break": "smaller_irrigation"
  },
  "penalties": {"new_search_performed": false, "reuse_outer_fold_selected_values": true},
  "forbidden": {
    "2019_rows_read_for_training_or_selection": true,
    "2019_features_read": true,
    "2019_targets_read": true,
    "2024_rows_read": true,
    "swap_rerun": true,
    "moe_training": true,
    "tta": true,
    "model_promotion": true
  },
  "passing_action": "authorize_2015_2018_robust_envelope_development_swap_protocol_design_only"
}
```

- [ ] 在运行器中实现 `validate_protocol(protocol)`，对上述字段逐项做显式断言，禁止仅检查协议编号。

- [ ] 重新运行协议测试，预期全部通过。

- [ ] 提交本任务涉及的三个文件：

```powershell
git -C D:\study add s2s_rtist_source/docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json s2s_rtist_source/tests/test_run_gefs_robust_envelope_gam_development_v1.py s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py
git -C D:\study commit -m "test: freeze robust envelope GAM development protocol"
```

### Task 2：实现解析下包络连续求解器

**Files:**

- Create: `src/s2s_rtist/models/robust_gam_envelope_v1.py`
- Create: `tests/test_robust_gam_envelope_v1.py`

- [ ] 先写五类数学测试：内部驻点最优、两曲线交点形成下包络折点最优、零灌溉保守边界最优、目标并列时取较小灌溉量、密网格结果只用于审计而不改变解析推荐。

- [ ] 测试使用可直接核验的三次以下多项式，不依赖正式数据。核心交点测试使用：

```python
def test_pairwise_intersection_can_be_lower_envelope_maximum(self) -> None:
    objectives = (
        lambda x: 12.0 - (np.asarray(x) - 18.0) ** 2 / 18.0,
        lambda x: 12.0 - (np.asarray(x) - 42.0) ** 2 / 18.0,
    )
    result = optimize_piecewise_cubic_lower_envelope(
        objectives=objectives,
        interval_boundaries=np.asarray([0.0, 60.0]),
        deployment_resolution_mm=1.0e-6,
        dense_step_mm=0.25,
    )
    self.assertAlmostEqual(result.irrigation_mm, 30.0, places=6)
    self.assertGreaterEqual(result.pairwise_intersection_count, 1)
```

- [ ] 运行测试，预期因模块不存在失败：

```powershell
python -m unittest tests.test_robust_gam_envelope_v1 -v
```

- [ ] 实现不可变结果对象：

```python
@dataclass(frozen=True)
class EnvelopeOptimizationResult:
    raw_irrigation_mm: float
    irrigation_mm: float
    robust_incremental_gain: float
    mean_incremental_gain: float
    minimum_model_index: int
    candidate_count: int
    stationary_point_count: int
    pairwise_intersection_count: int
    dense_diagnostic_irrigation_mm: float
    dense_diagnostic_robust_gain: float
    dense_diagnostic_gain_gap: float
```

- [ ] 实现 `optimize_piecewise_cubic_lower_envelope(...)`：在每个公共样条区间内，用四个固定归一化采样点恢复每条三次多项式；收集区间边界、每条多项式导数的区间内实根、每对多项式差的区间内实根；去重后在全部候选上计算 `min_m Delta_m(I)`；最大值并列时取最小灌溉量。

- [ ] 增加三项数值保护：多项式重构点的最大残差不超过 `1e-8`；所有目标值和根必须有限；解析最优下包络值不得比 0.25 mm 密网格最优低超过 0.05。密网格不能加入候选集。

- [ ] 实现 GAM 适配函数，始终使用外层未见站点共享项：

```python
def gam_incremental_objective(model, state: pd.DataFrame):
    zero = _cycle_probe(state, np.asarray([0.0]))
    zero_gain = float(model.predict_shared(zero)["pred_target_net_gain_7d"].iloc[0])

    def objective(irrigation: np.ndarray) -> np.ndarray:
        probe = _cycle_probe(state, irrigation)
        gain = model.predict_shared(probe)["pred_target_net_gain_7d"].to_numpy(float)
        return gain - zero_gain

    return objective
```

- [ ] 运行数学测试，预期全部通过；同时运行现有连续求解测试，确认无回归：

```powershell
python -m unittest tests.test_robust_gam_envelope_v1 tests.test_gefs_hierarchical_gam_continuous_qualification_v1.PiecewiseStationaryOptimizerTests -v
```

- [ ] 提交纯数学模块和测试：

```powershell
git -C D:\study add s2s_rtist_source/src/s2s_rtist/models/robust_gam_envelope_v1.py s2s_rtist_source/tests/test_robust_gam_envelope_v1.py
git -C D:\study commit -m "feat: add exact robust GAM envelope optimizer"
```

### Task 3：实现严格删一源站拟合与审计

**Files:**

- Modify: `scripts/training/run_gefs_robust_envelope_gam_development_v1.py`
- Modify: `tests/test_run_gefs_robust_envelope_gam_development_v1.py`

- [ ] 增加合成折数据测试。每折有一个外层目标站点和四个源站点；`build_jackknife_training_sets` 必须返回四个子集，每个子集只含三个源站点，且外层目标站点与被删源站点的拟合行数、标准化行数均为 0。

```python
def test_jackknife_training_sets_exclude_target_and_deleted_source(self) -> None:
    outer_train = synthetic_outer_train()
    splits = build_jackknife_training_sets(
        outer_train,
        target_site="P1",
        source_sites=("P2", "P3", "P4", "P15"),
    )
    self.assertEqual(len(splits), 4)
    for split in splits:
        self.assertNotIn("P1", set(split.frame["site_id"]))
        self.assertNotIn(split.deleted_source_site, set(split.frame["site_id"]))
        self.assertEqual(split.frame["site_id"].nunique(), 3)
```

- [ ] 增加冻结惩罚读取测试：从外层折 `GAM-VC/summary.csv` 只读取 `lambda_main`、`lambda_site`、`lambda_interaction`；文件缺失、重复行、非有限数、协议或折编号不符都必须停止；不得调用 `_select_penalty`。

- [ ] 运行测试并确认新函数尚未实现而失败。

- [ ] 复用现有函数：

```python
from scripts.training.run_gefs_hierarchical_gam_bspline_development_v1 import (
    CYCLE_KEYS,
    _fit_model,
    add_states,
    build_outer_fold,
    prediction_metrics,
    validate_complete_cycles,
)
```

- [ ] 每个外层折按原滚动年份重新构造训练和验证数据；以冻结惩罚拟合一个四源站 GAM-VC 基线，并分别删除 P2/P3/P4/P15 中与目标站点不同的每个源站，拟合四个三源站 GAM-VC。所有模型的状态缩放、基函数边界和系数只从本模型训练行拟合。

- [ ] 每个模型写一行审计，字段至少包含：`fold_id`、`target_site`、`validation_year`、`model_role`、`deleted_source_site`、`included_source_sites`、`training_rows`、`training_cycles`、`target_site_training_rows=0`、`target_site_preprocessing_rows=0`、`deleted_source_training_rows=0/NA`、`deleted_source_preprocessing_rows=0/NA`、三个惩罚值、参数量、拟合秒数。

- [ ] 模型对象不要求持久化全部系数；为断点续跑，每折完成后原子写入 `folds/<fold_id>/fold_complete.json`、逐周期 CSV、模型审计 CSV 和必要系数 NPZ。只有完成标记和文件哈希均匹配才允许跳过该折。

- [ ] 运行专项测试，预期通过：

```powershell
python -m unittest tests.test_run_gefs_robust_envelope_gam_development_v1.RobustEnvelopeIsolationTests -v
```

- [ ] 提交拟合与隔离实现：

```powershell
git -C D:\study add s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py s2s_rtist_source/tests/test_run_gefs_robust_envelope_gam_development_v1.py
git -C D:\study commit -m "feat: fit strict jackknife GAM source models"
```

### Task 4：生成 196 周期连续决策与多输出指标

**Files:**

- Modify: `scripts/training/run_gefs_robust_envelope_gam_development_v1.py`
- Modify: `tests/test_run_gefs_robust_envelope_gam_development_v1.py`

- [ ] 写逐周期测试，要求：八点真实参考峰值在目标净收益并列时选择较小灌溉量；基线连续解使用原四源站模型共享项；稳健连续解使用四个三源站模型增量收益下包络；目标标签只用于求参考峰值和事后指标，不能传入连续求解器。

- [ ] 实现真实八点参考峰值：

```python
def true_fixed_grid_peak(cycle: pd.DataFrame) -> tuple[float, float]:
    ordered = cycle.sort_values("irrigation_mm", kind="mergesort")
    gains = ordered["target_net_gain_7d"].to_numpy(dtype=float)
    maximum = float(np.max(gains))
    tied = np.isclose(gains, maximum, rtol=1.0e-12, atol=1.0e-12)
    selected = ordered.loc[tied].sort_values("irrigation_mm", kind="mergesort").iloc[0]
    return float(selected["irrigation_mm"]), float(selected["target_net_gain_7d"])
```

- [ ] 对每个验证周期输出：基线与稳健连续灌溉量、八点真实峰值、两种峰值绝对距离、真实零灌溉时的正灌溉假阳性、四子模型在稳健推荐处的增量收益最小值/均值/标准差/符号一致数、解析候选数、交点数、密网格差、求解耗时。

- [ ] 基线连续求解调用现有 `optimize_cycle`，不得从八点中选最终答案；稳健求解调用 Task 2 的解析下包络求解器。

- [ ] 四子模型其他输出按算术均值形成集成预测。在同一验证折上，用与基线相同的目标尺度和水量平衡尺度调用现有 `prediction_metrics`；输出基线与集成均值的 `macro_composite`，防止通过改变归一化口径获得假改善。

- [ ] 完成计数断言：逐周期表必须正好 196 行，折为 15 个，站点周期计数必须等于协议；每周期必须恰有八个原始标签；任何重复周期、缺失周期、非有限预测或越界灌溉量立即失败。

- [ ] 运行周期与指标测试：

```powershell
python -m unittest tests.test_run_gefs_robust_envelope_gam_development_v1.RobustEnvelopeCycleMetricTests -v
```

- [ ] 提交周期评估实现：

```powershell
git -C D:\study add s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py s2s_rtist_source/tests/test_run_gefs_robust_envelope_gam_development_v1.py
git -C D:\study commit -m "feat: evaluate robust envelope GAM development cycles"
```

### Task 5：实现七项门禁、审计和停止规则

**Files:**

- Modify: `scripts/training/run_gefs_robust_envelope_gam_development_v1.py`
- Modify: `tests/test_run_gefs_robust_envelope_gam_development_v1.py`

- [ ] 构造一组明确通过和七组分别只破坏一项条件的测试数据，证明门禁使用逻辑“全部为真”，不是多数表决。

- [ ] 实现以下七项条件，容差固定为 0：

1. 稳健方案总体平均峰值绝对距离不大于基线连续方案。
2. 真实正灌溉参考周期的平均峰值绝对距离不大于基线。
3. 真实零灌溉参考周期的正灌溉假阳性数量不大于基线。
4. 至少 10/15 个外层折平均峰值绝对距离不差。
5. 全局最大峰值绝对距离不增加。
6. 四子模型均值的总体 `macro_composite` 不大于四源站基线模型。
7. 解析求解、密网格差、周期/折/模型计数、数据隔离、哈希与有限数审计全部通过。

- [ ] 门禁 JSON 的 `observed` 必须同时写候选和基线原始数值，不能只写布尔值；写出正灌溉和零灌溉周期样本数、非劣折数、最差折和最差站点。

- [ ] 审计 JSON 必须明确记录：2019 训练/选择/特征/目标读取行数均为 0，2024 读取行数为 0，新惩罚搜索次数为 0，SWAP/MoE/TTA/模型推广均未执行，外层目标站点和被删源站点隔离均通过，2019 已被用于形成假设所以后续只能是复用背景年份探索。

- [ ] 状态字段固定为：

```python
if gate["passed"]:
    status = "robust_envelope_gam_development_pregate_passed_pending_development_swap_protocol"
    next_gate = "design_2015_2018_paired_continuous_swap_protocol_only"
else:
    status = "robust_envelope_gam_development_failed_stop_before_swap_or_2019"
    next_gate = "stop_robust_envelope_route"
```

- [ ] 生成清单时对所有正式 CSV/JSON/NPZ 计算 SHA256；清单自身不包含自身哈希。写文件使用临时文件后 `replace`，避免中断留下貌似完整的 JSON。

- [ ] 运行门禁测试和整个新测试集：

```powershell
python -m unittest tests.test_robust_gam_envelope_v1 tests.test_run_gefs_robust_envelope_gam_development_v1 -v
```

- [ ] 提交门禁和审计实现：

```powershell
git -C D:\study add s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py s2s_rtist_source/tests/test_run_gefs_robust_envelope_gam_development_v1.py
git -C D:\study commit -m "feat: gate robust envelope GAM development"
```

### Task 6：登记脚本并编写服务器操作文档

**Files:**

- Modify: `scripts/script_catalog.csv`
- Create: `docs/operations/2026-08-06-gefs-robust-envelope-gam-development-server.md`

- [ ] 在目录中登记运行器用途、输入、输出、阶段和禁止读取年份。

- [ ] 操作文档使用完整绝对路径，不定义路径变量；必须一次性列出：服务器解包、`python3 -m unittest`、语法检查、后台启动、PID 检查、`tail -f`、折计数、错误检索、结果 JSON 查看、关键 CSV 汇总、打包、压缩包列表和 SHA256。

- [ ] 正式启动命令固定为：

```bash
nohup env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -u /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/training/run_gefs_robust_envelope_gam_development_v1.py --protocol /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json --dataset-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_exact_schedule_surrogate_dataset_v1 --source-development-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_hierarchical_gam_nested_shared_selection_development_v2 --output-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1 > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log 2>&1 & echo $! > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.pid
```

- [ ] 检查命令必须含用户习惯的实时日志：

```bash
ps -fp "$(cat /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.pid)"
tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log
grep -c '^fold=' /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log
grep -nE 'Traceback|Error|Exception|failed|nonfinite|leak' /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_teacher_guided_robust_envelope_gam_development_v1.log
```

- [ ] 结果查看命令必须直接打印两个 JSON，并用内嵌 Python 打印七项条件、观测值、15 折和 5 站点关键表；不要求用户手工打开文件。

- [ ] 打包只包含关键原始表、JSON、运行日志和必要系数审计，不包含全部折工作目录。压缩包名固定为 `gefs_robust_envelope_gam_development_key_results_20260806_v1.tar.gz`。

- [ ] 提交目录和操作文档：

```powershell
git -C D:\study add s2s_rtist_source/scripts/script_catalog.csv s2s_rtist_source/docs/operations/2026-08-06-gefs-robust-envelope-gam-development-server.md
git -C D:\study commit -m "docs: add robust envelope GAM server workflow"
```

### Task 7：本地验证、制作服务器代码包并停止在第一阶段边界

**Files:**

- Verify only: all files introduced above

- [ ] 运行新旧相关测试，不能只运行新测试：

```powershell
python -m unittest tests.test_gefs_hierarchical_gam_bspline_v1 tests.test_run_gefs_hierarchical_gam_bspline_development_v1 tests.test_gefs_hierarchical_gam_continuous_qualification_v1 tests.test_robust_gam_envelope_v1 tests.test_run_gefs_robust_envelope_gam_development_v1 -v
```

- [ ] 运行语法检查：

```powershell
python -m py_compile src/s2s_rtist/models/robust_gam_envelope_v1.py scripts/training/run_gefs_robust_envelope_gam_development_v1.py tests/test_robust_gam_envelope_v1.py tests/test_run_gefs_robust_envelope_gam_development_v1.py
```

- [ ] 检查协议 JSON：

```powershell
python -m json.tool docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json > $null
```

- [ ] 检查差异中不存在 2019 数据加载、第二阶段 SWAP、MoE、TTA、密网格选解或站点专用阈值：

```powershell
rg -n "2019|2024|SWAP|swap|MoE|moe|TTA|tta|dense" src/s2s_rtist/models/robust_gam_envelope_v1.py scripts/training/run_gefs_robust_envelope_gam_development_v1.py docs/superpowers/specs/2026-08-06-teacher-guided-robust-envelope-gam-development-v1.json
git -C D:\study diff --check
```

逐条人工确认命中仅为禁止/审计字段或密网格诊断，不存在越权执行路径。

- [ ] 制作服务器代码包，包内路径以 `s2s_rtist_source/` 为根，包含模型、运行器、两份测试、协议、设计、计划、操作文档和脚本目录登记。不要包含本地正式结果或用户无关改动。

- [ ] 在全新临时目录解包并再次运行包内测试，确认没有依赖未打包的新文件。

- [ ] 输出代码包 SHA256 和包内文件清单。用户传到服务器后，只按 Task 6 文档执行；本地不运行正式 15 折数据实验。

- [ ] 第一阶段服务器结果返回后，只做以下判断：门禁通过则新建第二阶段“2015--2018 配对连续 SWAP”设计与计划；门禁失败则把本路线冻结为负结果。无论结果如何，本计划均不得自动读取 2019、运行 2024、进入 MoE/TTA 或修改门槛。

## 完成标准

- 新代码测试和相关回归测试全部通过。
- 正式第一阶段恰有 15 折、196 周期、60 个删一源站子模型，且所有隔离计数为 0。
- 稳健推荐来自解析连续下包络，不来自八点选择或密网格选择。
- 七项门禁全部以预声明原始数值输出；任何一项失败即停止。
- 服务器操作文档同时提供运行、`tail -f`、查看、汇总和打包命令，并且所有服务器路径均为绝对路径。
- 第二阶段不在本实现中提前编码或运行。
