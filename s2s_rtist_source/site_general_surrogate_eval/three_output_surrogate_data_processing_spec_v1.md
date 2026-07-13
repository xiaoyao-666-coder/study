# 三输出连续灌溉代理模型数据处理规范 v1

日期：2026-07-13

## 1. 文档目的

本文档固定三输出连续灌溉代理模型现阶段已经确认的数据处理方式，作为后续修改 SWAP 标签生成脚本、构建训练集和验证集、设计物理一致性损失以及整理正式数据报告的依据。

本文档只讨论数据生成与预训练所需标签。TTA 阶段的土壤湿度监督来源和在线更新方式暂不纳入本版本。

## 2. 已确认的模型输出

代理模型对外报告三个输出：

1. 未来 7 天净收益，标量；
2. 未来 7 天累计实际蒸散发，标量；
3. 未来 7 天逐日动态根区平均土壤湿度，长度为 7 的序列。

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

## 5. 动态根区土壤湿度

### 5.1 数据来源

- 每日根深：`result_restart.crp` 中的 `Rootd`；
- 各土层体积含水量：`result_restart.vap` 中的 `wcontent`；
- 土层顶部和底部深度：`result_restart.vap` 中的 `top` 和 `bottom`。

### 5.2 厚度加权平均

对每个日期，根据当日根深确定动态根区 `[0, Rootd]`。对所有与根区相交的土层计算相交厚度：

```text
overlap_i = 第 i 层与 [0, Rootd] 的相交厚度
```

当日动态根区平均体积含水量为：

```text
rootzone_vwc_day
= Σ(wcontent_i × overlap_i) / Σ(overlap_i)
```

如果根区底部只覆盖某一土层的一部分，必须按实际相交厚度加权，不能把整层直接纳入或直接丢弃。

### 5.3 根区储水量

当日动态根区储水深度为：

```text
rootzone_storage_cm
= Σ(wcontent_i × overlap_i)
```

转换为毫米：

```text
rootzone_storage_mm = 10 × rootzone_storage_cm
```

7 天动态根区储水变化为：

```text
delta_rootzone_storage_7d_mm
= day7_end_rootzone_storage_mm
- predecision_rootzone_storage_mm
```

初始储水量使用决策日前一日的 `result_forec.vap` 和 `result_forec.crp` 计算。不能使用第 1 个预测日结束后的土壤湿度代替初始状态。

### 5.4 输出字段

数据中应保留：

```text
root_depth_day01_cm ... root_depth_day07_cm
rootzone_vwc_day01 ... rootzone_vwc_day07
rootzone_storage_day01_mm ... rootzone_storage_day07_mm
predecision_root_depth_cm
predecision_rootzone_vwc
predecision_rootzone_storage_mm
delta_rootzone_storage_7d_mm
```

代理模型正式土壤湿度输出为 `rootzone_vwc_day01 ... day07`。

## 6. RootBoundaryFlux 与 RootDrainage

### 6.1 RootBoundaryFlux 的含义

`result_restart.vap` 中的 `waterflux` 表示土层边界水通量，并作用于对应土层的顶部边界。

对每个日期，根据动态根深选择最接近根区底部的土层边界，该边界的 `waterflux` 记为：

```text
root_boundary_flux_day_cm
```

P1 审计显示，将正的 `RootBoundaryFlux` 作为向下流出根区的水量处理时，动态根区水量平衡误差更接近 0。因此现阶段符号约定为：

```text
RootBoundaryFlux > 0：从根区向下流出
RootBoundaryFlux < 0：从根区下方向上补给
```

转换为毫米并累计：

```text
root_boundary_flux_7d_mm
= 10 × Σ(root_boundary_flux_day_cm)
```

### 6.2 RootDrainage 的含义

`result_restart.vap` 中的 `drainage` 为土层排水量。动态根区排水量按土层与根区的相交比例累计：

```text
root_drainage_day_cm
= Σ(drainage_i × overlap_i / layer_thickness_i)
```

未来 7 天累计值为：

```text
root_drainage_7d_mm
= 10 × Σ(root_drainage_day_cm)
```

### 6.3 为什么需要保留这两个字段

即使某个站点和日期的根区底部通量、根区排水量接近 0，也不能在原始数据生成阶段直接删除。原因包括：

- 不同站点的土壤水力参数不同；
- 不同年份和天气条件下深层渗漏可能显著变化；
- 大灌溉量或强降水后根区底部通量可能增加；
- 后续需要判断简化水量平衡是否在全部站点和日期上成立。

因此，数据层面必须保留逐日值和 7 天累计值。是否将其加入模型输出层属于后续模型选择问题。

## 7. 7 天累计动态根区水量平衡

### 7.1 当前公式

按照正值表示水分流出根区的约定，7 天累计动态根区水量平衡为：

```text
delta_rootzone_storage_7d_mm
= rain_7d_mm
+ irrigation_7d_mm
+ runon_7d_mm
- aet_7d_mm
- runoff_7d_mm
- root_drainage_7d_mm
- root_boundary_flux_7d_mm
+ balance_error_7d_mm
```

等价的物理一致性残差为：

```text
water_balance_residual_7d_mm
= rain_7d_mm
+ irrigation_7d_mm
+ runon_7d_mm
- aet_7d_mm
- runoff_7d_mm
- root_drainage_7d_mm
- root_boundary_flux_7d_mm
- delta_rootzone_storage_7d_mm
```

理想情况下，该残差应接近 0。

### 7.2 P1 审计结果

审计设置：

```text
站点：P1/code_N1
决策日：16-Jul-2024
正式窗口：2024-07-16 至 2024-07-22
灌溉量：0 mm 和 30 mm
```

结果如下：

| 灌溉量 | 根深起点/终点 | 根区底部通量 | 根区排水 | 动态根区平衡误差 |
|---:|---:|---:|---:|---:|
| 0 mm | 100/100 cm | 0.0826 mm | 0 mm | -0.0371 mm |
| 30 mm | 100/100 cm | 0.0518 mm | 0 mm | 0.0191 mm |

这表明：

- 当前动态根区储水量计算能够与 SWAP 通量基本闭合；
- P1 当前日期的根区排水和根区底部通量很小；
- 整个土壤剖面的 `QBottom` 不是动态根区底部通量，不能直接用于动态根区水量平衡。

## 8. 残差通量监督标签

### 8.1 标签定义

为兼容部分站点可能存在不可忽略的未显式水分通量，数据中生成残差通量监督标签：

```text
residual_flux_7d_mm
= runoff_7d_mm
+ root_drainage_7d_mm
+ root_boundary_flux_7d_mm
```

同时保留三个组成项，不能只保存求和结果。

如果多站点审计发现根深变化导致动态控制体边界出现额外储水项，则增加：

```text
moving_root_boundary_adjustment_7d_mm
```

并将其单独报告。在完成变根深样本审计前，不允许将无法闭合的部分无说明地并入普通数值误差。

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
L_physics = loss(water_balance_residual_7d, 0)
```

直接监督可以防止残差通量头任意抵消其他预测误差，使物理损失失去约束作用。

## 9. 是否启用残差通量输出头

### 9.1 已确定事项

- 原始数据必须保留 `Runoff`、`RootDrainage` 和 `RootBoundaryFlux`；
- 必须生成 `residual_flux_7d_mm` 监督标签；
- 必须报告残差通量相对于降水、灌溉、蒸散发和储水变化的数量级；
- 如果启用残差通量头，必须使用 SWAP 标签直接监督。

### 9.2 尚未确定事项

是否在正式代理模型中启用残差通量输出头，需要在多站点、多日期和不同灌溉量的 smoke 审计后决定。

P1 单一样本中根区底部通量小于 `0.1 mm/7d`，不足以代表全部站点，尤其不能代表：

- 沙质土壤站点；
- 强降水日期；
- 高灌溉量日期；
- 根深发生明显变化的生育阶段。

本版本不预先设定可忽略阈值。多站点 smoke 应同时给出绝对量、相对占比和水量平衡误差，再决定是否启用该输出头。

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
rootzone_vwc_day01 ... rootzone_vwc_day07
rootzone_storage_day01_mm ... rootzone_storage_day07_mm
predecision_rootzone_storage_mm
delta_rootzone_storage_7d_mm

runoff_day01_mm ... runoff_day07_mm
root_drainage_day01_mm ... root_drainage_day07_mm
root_boundary_flux_day01_mm ... root_boundary_flux_day07_mm

runoff_7d_mm
root_drainage_7d_mm
root_boundary_flux_7d_mm
residual_flux_7d_mm
water_balance_residual_7d_mm
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
5. 动态根区土层覆盖厚度与当日根深一致；
6. `rootzone_vwc` 处于站点土壤物理合理范围；
7. 逐日根深、根区储水量和根区底部边界深度均有记录；
8. 7 天累计水量平衡残差有明确数值；
9. 对水量平衡不能闭合的样本保留原始 SWAP 文件路径和失败原因；
10. 预训练、验证和测试年份标记不能混淆。

## 12. 后续执行顺序

1. 修复 restart 数据生成程序的 8 天窗口问题；
2. 在每个候选 SWAP 运行结束后立即提取上述字段，避免下一候选覆盖原始输出；
3. 使用 P1 的 0 mm 和 30 mm 审计结果做回归测试；
4. 运行小规模多站点、多日期、多灌溉量 smoke；
5. 统计残差通量数量级和变根深情况下的平衡误差；
6. 决定正式模型是否启用残差通量输出头；
7. smoke 通过后再批量重新生成 2015-2019 年正式三输出数据。

## 13. 当前不处理的内容

以下内容留待数据与预训练链路完成后再处理：

- TTA 阶段土壤湿度标签来源；
- TTA 在线更新时序；
- 事件触发 TTA；
- TTA 学习率、更新步数和模型回退机制。
