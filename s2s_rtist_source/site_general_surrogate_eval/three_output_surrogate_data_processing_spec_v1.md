# 三输出连续灌溉代理模型数据处理规范 v1

日期：2026-07-13

## 1. 文档目的

本文档固定三输出连续灌溉代理模型现阶段已经确认的数据处理方式，作为后续修改 SWAP 标签生成脚本、构建训练集和验证集、设计物理一致性损失以及整理正式数据报告的依据。

本文档只讨论数据生成与预训练所需标签。TTA 阶段的土壤湿度监督来源和在线更新方式暂不纳入本版本。

## 2. 已确认的模型输出

代理模型对外报告三个输出：

1. 未来 7 天净收益，标量；
2. 未来 7 天累计实际蒸散发，标量；
3. 未来 7 天逐日固定 `0–100 cm` 土层平均体积含水率，长度为 7 的序列。

为计算物理一致性损失，数据中还需要保留根区水量平衡相关的中间变量和监督标签。这些中间变量不一定都作为最终对外输出。

## 3. 正式 7 天时间窗口

### 3.1 当前问题

现有 restart 生成程序采用：

```python
end_doy = decision_doy + HORIZON_DAYS
```

SWAP 的起止日期均包含在输出中。当 `HORIZON_DAYS = 7` 时，上述写法会生成从决策日到决策日后第 7 天的 8 个日期。

P1 站点审计结果为：

```text
2024-07-16 至 2024-07-23，共 8 行
```

### 3.2 正式处理方式

正式 7 天窗口定义为：

```text
决策日、决策日+1、……、决策日+6
```

程序应修改为：

```python
end_doy = decision_doy + HORIZON_DAYS - 1
```

以 `16-Jul-2024` 为例，正式标签日期为：

```text
2024-07-16 至 2024-07-22，共 7 天
```

决策日前一日的 SWAP 状态作为初始状态。灌溉量在决策日施加，未来 7 天的通量、土壤湿度和作物状态均从上述 7 个日期提取。

## 4. 实际蒸散发标签

### 4.1 SWAP 字段

当前使用的 SWAP 4.0.1 在 `result_restart.inc` 中输出：

```text
Tact   实际作物蒸腾
Eact   实际土壤蒸发
Interc 冠层截留水蒸发
```

三个字段的单位均为 `cm/day`。

### 4.2 计算方式

逐日实际蒸散发定义为：

```text
ETa_day = Tact + Eact + Interc
```

转换为 `mm/day`：

```text
ETa_day_mm = 10 × (Tact + Eact + Interc)
```

未来 7 天累计实际蒸散发为：

```text
ETa_7d_mm = Σ ETa_day_mm，day = 1...7
```

数据中应同时保留：

```text
tact_day01_mm ... tact_day07_mm
eact_day01_mm ... eact_day07_mm
interc_day01_mm ... interc_day07_mm
aet_day01_mm ... aet_day07_mm
aet_7d_mm
```

对外输出为 `aet_7d_mm`。逐日分量用于数据审计、物理损失核查和问题追踪。

### 4.3 审计依据

P1 站点的 7 天水量平衡只有在包含 `Interc` 时才能闭合，因此本 SWAP 版本不能仅使用 `Tact + Eact`。

## 5. 固定 0–100 cm 土层含水率

### 5.1 数据来源

- 每日根深：`result_restart.crp` 中的 `Rootd`；
- 各土层体积含水量：`result_restart.vap` 中的 `wcontent`；
- 土层顶部和底部深度：`result_restart.vap` 中的 `top` 和 `bottom`。

### 5.2 厚度加权平均

对每个日期，固定选择 `[0, 100 cm]`。对所有与该固定控制体相交的土层计算相交厚度：

```text
overlap_0_100_i = 第 i 层与 [0, 100 cm] 的相交厚度
```

当日固定土层平均体积含水量为：

```text
soil_vwc_0_100cm_day
= Σ(wcontent_i × overlap_0_100_i) / 100 cm
```

当前剖面在 `100 cm` 处存在精确边界。实现仍按相交厚度计算，以便遇到其他网格时不会把底层整层错误纳入或丢弃。

### 5.3 固定土层储水量

当日固定土层储水深度为：

```text
soil_storage_0_100cm_cm
= Σ(wcontent_i × overlap_0_100_i)
```

转换为毫米：

```text
soil_storage_0_100cm_mm = 10 × soil_storage_0_100cm_cm
```

7 天固定土层储水变化为：

```text
delta_soil_storage_0_100cm_7d_mm
= day7_end_soil_storage_0_100cm_mm
- predecision_soil_storage_0_100cm_mm
```

初始储水量使用决策日前一日的 `result_forec.vap` 计算。不能使用第 1 个预测日结束后的土壤湿度代替初始状态。`result_forec.crp` 的根深仍作为模型状态输入和审计字段，但不再改变储水控制体边界。

### 5.4 输出字段

数据中应保留：

```text
root_depth_day01_cm ... root_depth_day07_cm
soil_vwc_0_100cm_day01 ... soil_vwc_0_100cm_day07
soil_storage_0_100cm_day01_mm ... soil_storage_0_100cm_day07_mm
predecision_root_depth_cm
predecision_soil_vwc_0_100cm
predecision_soil_storage_0_100cm_mm
delta_soil_storage_0_100cm_7d_mm
```

代理模型正式土壤湿度输出为 `soil_vwc_0_100cm_day01 ... day07`。

### 5.5 动态根区审计分支

动态根区 `[0, Rootd]` 的厚度加权 VWC、储水量和移动边界项继续计算并保存在诊断数据中，用于验证固定控制体改造前后的物理一致性。它们不进入正式三输出契约，也不能与固定 `0–100 cm` 字段同名。

## 6. RootBoundaryFlux 与 RootDrainage

### 6.1 RootBoundaryFlux 的含义

`result_restart.vap` 文件头明确说明 `waterflux` 和 `drainage` 是瞬时通量速率，单位为 `cm/day`。`waterflux` 作用于对应土层的顶部边界。

正式模型固定选择 `100 cm` 土层边界；动态根区审计分支才根据当日根深选择根区底部边界。原生有符号通量记为：

```text
root_boundary_waterflux_signed_cm_day
```

SWAP 配置文件和输出平衡共同确认其原生符号为：

```text
waterflux > 0：向上
waterflux < 0：向下
```

若另行定义“向下流出根区为正”的累计量，则应写为：

```text
root_boundary_outflow_7d_mm
= -10 × integral(root_boundary_waterflux_signed_cm_day, time)
```

不能把正的原生 `waterflux` 直接解释为向下流出。此前 P1 审计中由该错误符号得到的近零残差不再作为闭合证据。

当前 `swap_three_output_labels_v1.py` 在 `NPrintDay=1` 时，把每天一次的瞬时值乘以 `1 day` 后求和，等价于日间隔矩形积分近似。该处理只有在瞬时通量足够平稳或每日采样能够代表日平均时才可靠；正式累计标签需要通过更高 `NPrintDay` 的收敛试验确认。

### 6.2 固定根区的空间边界

当前已审计固定根区样本的根深为整数厘米，`result_restart.vap` 中存在与根深完全对应的土层顶部边界。例如根深 `100 cm` 对应 `top=-100 cm`，因此这些样本的边界选择是精确匹配，不是 C2 固定根区异常的来源。

这一结论只适用于当前已审计网格。后续若出现无法与网格边界精确匹配的非整数根深，必须单独记录边界偏差并确定处理方法。

### 6.3 正式模型采用固定 0–100 cm 控制体

2026-07-15 根据老师提出的固定根区方向完成复核。当前玉米配置的最大根深 `RDC=100 cm`，且剖面在 `100 cm` 处有精确边界。`code_N2 / 15-May-2024 / 30 mm / NPrintDay=24` 在根深由 `69 cm` 增至 `85 cm` 时，固定 `0–100 cm` 重算的水量平衡残差为 `-0.1590 mm`，不需要移动边界项。

因此正式模型第三输出改为：

```text
soil_vwc_0_100cm_day01 ... soil_vwc_0_100cm_day07
```

该字段表示固定土层平均含水率，不再表示动态根区平均含水率。逐日根深仍作为输入或状态特征。动态根区 VWC 与 `moving_root_boundary_term_7d_mm` 仅在原始物理审计分支保留，不作为正式模型输出头。

完整依据与浅根期差异见 `fixed_0_100cm_control_volume_validation_2026-07-15.md`。

### 6.4 RootDrainage 的含义

`result_restart.vap` 中的 `drainage` 为土层排水量。正式固定控制体按土层与 `[0, 100 cm]` 的相交比例累计；动态根区审计分支改用 `[0, Rootd]`：

```text
soil_drainage_0_100cm_rate_cm_day(t)
= sum(drainage_i(t) × overlap_0_100_i / layer_thickness_i)
```

未来 7 天累计值为：

```text
soil_drainage_0_100cm_7d_mm
= 10 × integral(soil_drainage_0_100cm_rate_cm_day, time)
```

与 `waterflux` 相同，每天一次的 `drainage` 瞬时值求和也只是 `1 day` 步长的矩形积分近似。

### 6.5 为什么需要保留这些字段

即使某个站点和日期的根区底部通量、根区排水量接近 0，也不能在原始数据生成阶段直接删除。原因包括：

- 不同站点的土壤水力参数不同；
- 不同年份和天气条件下深层渗漏可能显著变化；
- 大灌溉量或强降水后根区底部通量可能增加；
- 后续需要判断简化水量平衡是否在全部站点和日期上成立。

因此，数据层面必须保留逐日值和 7 天累计值。是否将其加入模型输出层属于后续模型选择问题。

## 7. 7 天累计水量平衡

### 7.1 固定根区公式

定义 `soil_boundary_outflow_100cm_7d_mm` 为穿过 `100 cm` 边界向下流出的正值。固定控制体的 7 天累计水量平衡为：

```text
delta_soil_storage_0_100cm_7d_mm
= rain_7d_mm
+ irrigation_7d_mm
+ runon_7d_mm
- aet_7d_mm
- runoff_7d_mm
- soil_drainage_0_100cm_7d_mm
- soil_boundary_outflow_100cm_7d_mm
+ balance_error_7d_mm
```

等价的物理一致性残差为：

```text
water_balance_residual_0_100cm_7d_mm
= rain_7d_mm
+ irrigation_7d_mm
+ runon_7d_mm
- aet_7d_mm
- runoff_7d_mm
- soil_drainage_0_100cm_7d_mm
- soil_boundary_outflow_100cm_7d_mm
- delta_soil_storage_0_100cm_7d_mm
```

理想情况下，该残差应接近 0。

### 7.2 动态根区与移动边界项

根深变化时，根区是移动控制体。根区向下增长会把原本不属于根区的含水土层纳入储水统计，该部分不是穿过根区底部的实际水流，必须单独表示为移动边界项：

```text
moving_root_boundary_term_7d_mm
= 10 × integral(theta(R(t), t), dR(t))
```

离散计算时，应按每日新增根深和新纳入土层的含水率累计：

```text
moving_root_boundary_term_7d_mm
approximately equals 10 × sum(theta_new_day × (root_depth_day - root_depth_previous_day))
```

动态根区平衡为：

```text
water_balance_residual_7d_mm
= rain_7d_mm
+ irrigation_7d_mm
+ runon_7d_mm
- aet_7d_mm
- runoff_7d_mm
- root_drainage_7d_mm
- root_boundary_outflow_7d_mm
+ moving_root_boundary_term_7d_mm
- delta_rootzone_storage_7d_mm
```

理想情况下，该残差应接近 0。固定根区是 `moving_root_boundary_term_7d_mm = 0` 的特例。

### 7.3 P1 原始结果的符号复核

审计设置：

```text
站点：P1/code_N1
决策日：16-Jul-2024
正式窗口：2024-07-16 至 2024-07-22
灌溉量：0 mm 和 30 mm
```

使用 SWAP 原生“正向上、负向下”的符号重新核对后：

| 灌溉量 | 根深起点/终点 | 每日瞬时值矩形积分，正向上 | 平衡反推的有符号边界通量 | 两者差值 |
|---:|---:|---:|---:|---:|
| 0 mm | 100/100 cm | +0.082607 mm | -0.0455 mm | +0.128107 mm |
| 30 mm | 100/100 cm | +0.051779 mm | -0.0709 mm | +0.122679 mm |

这表明：

- P1 是固定根区样本，不涉及移动边界项；
- P1 当前日期的根区排水和根区底部通量量级较小；
- 每日一次瞬时采样仍留下约 `0.12 mm` 的边界通量差异；
- 此前将正 `waterflux` 当作向下流出得到的 `-0.0371 mm` 和 `0.0191 mm` 不能作为正确符号下的闭合结果；
- 整个土壤剖面的 `QBottom` 不是动态根区底部通量，不能直接用于动态根区水量平衡。

多站点固定根区和动态根区审计数据见 `three_output_rootzone_water_balance_audit_2026-07-13.md`。

## 8. 残差通量监督标签

### 8.1 标签定义

为兼容部分站点可能存在不可忽略的未显式水分通量，数据中生成残差通量监督标签：

```text
residual_flux_7d_mm
= runoff_7d_mm
+ soil_drainage_0_100cm_7d_mm
+ soil_boundary_outflow_100cm_7d_mm
```

同时保留三个组成项，不能只保存求和结果。

动态根区审计分支另外保留：

```text
moving_root_boundary_term_7d_mm
```

该项属于控制体边界移动引起的储水变化，不属于实际水分流出，不能并入 `residual_flux_7d_mm`。在使用逐日土层含水量显式计算该项前，只能把平衡差值称为“与移动边界项量级一致”，不能把差值直接当作已经测得的移动边界项。

2026-07-14 输出频率诊断已确认：当前每天一次瞬时通量生成的 `residual_flux_7d_mm` 不能作为正式监督标签。老师已正式采用 `NPrintDay=24`；该频率在两个 C2 固定根区样本中将平衡残差降至 `0.16 mm` 内，并在 N2 动态根区样本加入移动边界项后将残差降至 `0.0986 mm`。

### 8.2 预训练监督

老师已确认：如果启用残差通量输出头，预训练阶段直接使用 SWAP 提取的 `residual_flux_7d_mm` 作为监督标签。

对应损失记为：

```text
L_residual_flux
= loss(predicted_residual_flux_7d,
       swap_residual_flux_7d)
```

同时通过 7 天累计水量平衡残差损失约束预测结果：

```text
L_physics = loss(water_balance_residual_0_100cm_7d, 0)
```

直接监督可以防止残差通量头任意抵消其他预测误差，使物理损失失去约束作用。

## 9. 是否启用残差通量输出头

### 9.1 已确定事项

- 原始数据必须保留 `Runoff`、`RootDrainage` 和 `RootBoundaryFlux`；
- 必须同时保留原生有符号 `waterflux`、时间积分方法和转换后的向下净流出量；
- 动态根区审计分支必须单独保留移动边界项；
- 必须生成可审计的 `residual_flux_7d_mm` 候选监督标签；
- 必须报告残差通量相对于降水、灌溉、蒸散发和储水变化的数量级；
- 如果启用残差通量头，必须使用 SWAP 标签直接监督。

### 9.2 正式采用的输出结构

动态根区诊断阶段已采用“直接物理通量监督 + 移动边界项独立计算”的审计结构。2026-07-15 固定 `0–100 cm` 复核通过后，正式代理模型进一步简化为固定控制体：

- `residual_flux_7d_mm` 使用 SWAP 高频输出积分得到的直接物理流出量监督；
- 正式模型的储水量和边界通量均按固定 `0–100 cm` 计算；
- 正式模型不增加 `moving_root_boundary_term_7d_mm` 输出头；
- 动态根区审计数据中，该项仍由逐日根深和剖面含水量独立计算、独立保存；
- `water_balance_residual_0_100cm_7d_mm` 只作为正式模型的物理闭合审计与损失约束，不反向替代直接物理通量标签。

后续多站点、多日期和不同灌溉量的 smoke 用于验证该结构的跨样本稳定性、输出体积和运行成本，不再用于重新选择上述结构。P1 单一样本中根区底部通量小于 `0.1 mm/7d`，不足以代表全部站点，尤其不能代表：

- 沙质土壤站点；
- 强降水日期；
- 高灌溉量日期；
- 根深发生明显变化的生育阶段。

本版本不预先设定可忽略阈值。多站点 smoke 应同时给出绝对量、相对占比和水量平衡误差，用于审计正式输出头的稳定性。

### 9.3 输出频率诊断结果

老师批准的 `C2 30 mm`、`C2 60 mm` 和 `N2 30 mm` 三样本诊断已完成：

| 样本 | NPrintDay=1 | NPrintDay=4 | NPrintDay=24 |
|---|---:|---:|---:|
| C2 / 30 mm 固定根区残差 | 8.5575 mm | -1.2925 mm | -0.1555 mm |
| C2 / 60 mm 固定根区残差 | 3.4882 mm | 3.7574 mm | -0.1393 mm |
| N2 / 30 mm 加移动项后残差 | -11.3325 mm | -3.9328 mm | 0.0986 mm |

数据处理必须遵守：

- `result.vap` 的根区边界瞬时通量按实际子日时间间隔做梯形积分；
- SWAP 原生正向上、负向下符号必须保留到积分完成后；
- 高频 `result.inc` 的午夜行按 `Dcum` 归属模拟日，不能按日历 `Date` 直接分组；
- 动态根区移动边界项单独计算；
- `NPrintDay=4` 不能作为已经收敛的通量频率；
- 正式生产参数采用 `NPrintDay=24`，下一轮先做小规模多站点 smoke，再决定是否进入批量生成。

完整结果见 `three_output_rootzone_flux_frequency_validation_results_2026-07-14.md`。

## 10. 候选级正式数据字段

每个样本对应：

```text
站点 × 决策日期 × 天气情景 × 连续候选灌溉量
```

至少包含以下标签和审计字段：

```text
net_gain_7d

tact_day01_mm ... tact_day07_mm
eact_day01_mm ... eact_day07_mm
interc_day01_mm ... interc_day07_mm
aet_day01_mm ... aet_day07_mm
aet_7d_mm

root_depth_day01_cm ... root_depth_day07_cm
soil_vwc_0_100cm_day01 ... soil_vwc_0_100cm_day07
soil_storage_0_100cm_day01_mm ... soil_storage_0_100cm_day07_mm
predecision_soil_storage_0_100cm_mm
final_soil_storage_0_100cm_mm
delta_soil_storage_0_100cm_7d_mm

runoff_day01_mm ... runoff_day07_mm
soil_drainage_0_100cm_day01_mm ... soil_drainage_0_100cm_day07_mm
soil_boundary_waterflux_100cm_signed_day01_mm ... soil_boundary_waterflux_100cm_signed_day07_mm
soil_boundary_outflow_100cm_day01_mm ... soil_boundary_outflow_100cm_day07_mm
soil_boundary_depth_day01_cm ... soil_boundary_depth_day07_cm

runoff_7d_mm
soil_drainage_0_100cm_7d_mm
soil_boundary_waterflux_100cm_signed_7d_mm
soil_boundary_outflow_100cm_7d_mm
residual_flux_7d_mm
water_balance_residual_0_100cm_7d_mm

control_volume_type = fixed_0_100cm
control_depth_cm = 100
nprintday
flux_integration_method
```

同时保留：

```text
SWAP版本
天气来源
站点
年份
决策日期
窗口起止日期
实际输出天数
候选灌溉量
单位版本
数据处理规范版本
```

## 11. 数据质量检查

每批数据生成完成后必须检查：

1. 每个候选恰好包含 7 个正式预测日期；
2. 时间窗口不存在第 8 天；
3. `aet_day = tact_day + eact_day + interc_day`；
4. 所有水量从 `cm` 转换为 `mm` 时只乘一次 10；
5. 固定土层覆盖厚度精确等于 `100 cm`；
6. `soil_vwc_0_100cm` 处于站点土壤物理合理范围；
7. 逐日根深、固定土层储水量和 `100 cm` 边界均有记录；
8. 7 天累计水量平衡残差有明确数值；
9. `waterflux` 原生符号保持正向上、负向下，向下净流出只在派生字段中转换；
10. `waterflux` 和 `drainage` 的累计值记录输出频率与积分方法；
11. 动态根区审计分支单独计算并记录移动边界项，正式模型字段中该项不存在；
12. 对水量平衡不能闭合的样本保留原始 SWAP 文件路径和失败原因；
13. 预训练、验证和测试年份标记不能混淆。

## 12. 后续执行顺序

1. 修复 restart 数据生成程序的 8 天窗口问题；
2. 在每个候选 SWAP 运行结束后立即提取上述字段，避免下一候选覆盖原始输出；
3. 已完成 C2 固定根区与 N2 动态根区的 `NPrintDay=1、4、24` 诊断；
4. 已确认 `NPrintDay=1` 不足、`NPrintDay=4` 未稳定收敛、`NPrintDay=24` 在三个样本中闭合；
5. 已锁定正式通量频率和直接通量标签；
6. 已确认固定 `0–100 cm` 控制体可行，并已修改正式标签生成器、字段名、验证器和回归测试；
7. 已完成固定 `0–100 cm` 新契约五站点 smoke：40 个候选全部通过，旧 `rootzone` 或移动边界正式字段为 0，最大绝对水量平衡残差为 `0.308267 mm`；
8. 根据老师确认的后续实验范围组织小批量数据准备，不直接启动 2015-2019 全量生成；
9. GEFS 协议锁定后，使用固定 `0–100 cm` 储水与边界通量开展模型训练；移动边界项只用于动态根区审计。

## 13. 当前不处理的内容

以下内容留待数据与预训练链路完成后再处理：

- TTA 阶段土壤湿度标签来源；
- TTA 在线更新时序；
- 事件触发 TTA；
- TTA 学习率、更新步数和模型回退机制。
