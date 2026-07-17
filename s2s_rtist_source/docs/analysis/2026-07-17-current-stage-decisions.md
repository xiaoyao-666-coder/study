# 2026-07-17 当前阶段决策

## 固定 0-100 cm 控制体

正式水量平衡控制体采用固定 `0-100 cm`，土壤水分输出定义为“0-100 cm 土层平均含水率”。玉米配置最大根深为 `RDC=100 cm`，SWAP 在 100 cm 处存在精确土层边界，因此该深度具有作物和空间离散依据。

N2 动态样本使用固定控制体重算后，移动边界项按定义为 0，水量平衡残差为 `-0.1590 mm`。五站点 40 候选 smoke 的最大绝对残差为 `0.308267 mm`。动态根区及其移动边界项继续保留在审计层，不作为代理模型正式输出。

直接证据位于：

- `fixed_0_100cm_teacher_evidence_pack_20260715_v1/`
- `fixed_0_100cm_teacher_evidence_pack_20260715_v1/固定0-100cm关键数值.csv`
- `fixed_0_100cm_teacher_evidence_pack_20260715_v1/五站点固定控制体残差汇总.csv`
- `fixed_0_100cm_teacher_evidence_pack_20260715_v1/N2动态根区频率诊断汇总.csv`
- `fixed_0_100cm_teacher_evidence_pack_20260715_v1/N2逐日移动边界项.csv`

## GEFS 31成员降水验证

5 个 00 UTC 周期、5 个站点、31 个成员共形成 175 个站点/周期/日集合观测。当前样本显示明确的条件性偏差：干日和轻雨偏湿，中雨和强降水偏干；强降水平均低估 `21.1584 mm/day`。

总体平均偏差约为 `+0.0017 mm/day`，但这是不同天气条件下正负偏差相互抵消，不能解释为无偏。7 个强降水观测中仅 2 个落入集合 P10-P90，3 个落入全部 31 成员的最小至最大范围。

因此 GEFS 继续作为决策时可获得的未来天气来源，但原始集合均值不作为唯一输入。后续至少保留集合均值、P10/P50/P90、标准差和不同降水阈值的成员概率，并使用独立验证集比较“集合均值基线”和“集合分布特征方案”。

直接证据位于：

- `docs/analysis/gefs_31member_5cycle_precipitation_bias_20260716_v1/00_结论与证据.md`
- `docs/analysis/gefs_31member_5cycle_precipitation_bias_20260716_v1/02_按降水强度分层指标.csv`
- `docs/analysis/gefs_31member_5cycle_precipitation_bias_20260716_v1/04_强降水事件逐例证据.csv`
- `docs/analysis/gefs_31member_5cycle_precipitation_bias_20260716_v1/05_各站点各周期7天累计降水.csv`

## 结论边界

固定控制体结论已有 N2 动态样本和五站点 smoke 的直接物理证据支持。GEFS 的条件性偏差结论基于有目的选择的 5 个诊断周期，其中强降水观测只有 7 个，足以证明偏差与天气条件相关，但不足以形成整个 2024 生长季的固定订正系数。
