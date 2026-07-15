# GEFS 协议可行性与证据

## 1. 目的

本文件对以下五项拟定规则逐条判断是否可行，并区分：

- NOAA 官方产品事实；
- 项目当前已有字段和代码事实；
- 基于这些事实作出的研究选择；
- 采用后可能产生的影响。

拟判断的规则为：

1. 每个决策日使用决策时刻之前最新且完整可用的 GEFS 00 UTC 周期；
2. 首版模型使用集合成员逐日天气均值，离散度只做审计；
3. 7 天天气窗口按站点当地日期映射为 `D` 至 `D+6`；
4. 累计降水按相邻提前期差分；
5. 周期缺失时最多回退 24 小时，不使用实际天气补齐。

## 2. 官方产品事实

NOAA EMC 对 GEFS v12 的说明为：

- 每天运行 `00、06、12、18 UTC` 四个周期；
- 每个周期包含 31 个成员，即控制成员和 30 个扰动成员；
- 大气产品水平分辨率为 `0.25 degree`；
- 常规预报长度为 16 天。

来源：

- [NOAA EMC GEFS](https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php)
- [NCEP GEFS 产品清单](https://www.nco.ncep.noaa.gov/pmb/products/gens/)

NCEP 产品清单显示，`pgrb2s.0p25` 常用变量产品在 `f003-f240` 为 3 小时间隔。因此，覆盖美国站点当地 `D` 至 `D+6` 时获取到 `f180` 在产品范围内可行。

NCEI 说明 2017 年至今的 GEFS 数据通过 NOAA Open Data Dissemination 提供，但同时注明该渠道不是正式归档，NCEI 自有正式归档在 2020 年分辨率变更处结束。因此，2024 数据当前可获得，但项目必须保存对象路径、校验信息和不可变子集，不能只依赖远程对象长期存在。

来源：[NCEI GEFS 数据说明](https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast)

## 3. 2024 决策日可用性实查

对项目当前 10 个 2024 决策日，检查 NOAA Open Data 中 `00 UTC / p30 / f180` 对象的 `Last-Modified`。结果均存在，发布时间距离初始化约 `5.028-5.079 h`。

| 决策日 | f180 完成时间 UTC | 初始化后延迟 |
|---|---:|---:|
| 2024-07-16 | 05:02:35 | 5.043 h |
| 2024-07-20 | 05:03:06 | 5.052 h |
| 2024-07-24 | 05:03:09 | 5.053 h |
| 2024-07-28 | 05:02:24 | 5.040 h |
| 2024-08-01 | 05:02:53 | 5.048 h |
| 2024-08-05 | 05:03:01 | 5.050 h |
| 2024-08-09 | 05:01:39 | 5.028 h |
| 2024-08-13 | 05:02:11 | 5.036 h |
| 2024-08-17 | 05:03:28 | 5.058 h |
| 2024-08-21 | 05:04:45 | 5.079 h |

完整表见 `gefs_2024_00utc_availability_evidence_2026-07-14.csv`。

另外，对 `2024-07-16 00 UTC` 已核验：

- 控制成员 `c00` 和 `p01-p30` 共 31 个成员的 `f180` 均存在；
- `p01/f180` 包含 `TMP、DPT、RH、TMAX、TMIN、UGRD、VGRD、APCP、DSWRF` 等项目所需变量；
- 该事实证明一个实际项目日期的数据路径和变量组合可获取，但不替代后续对全部日期、全部成员和 `f003-f180` 的正式预检。

示例官方对象索引：

- [2024-07-16 p01 f180](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f180.idx)
- [2024-07-16 p30 f180](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/pgrb2sp25/gep30.t00z.pgrb2s.0p25.f180.idx)

## 4. 五项规则判定

### 4.1 00 UTC 周期

判定：**补充决策时刻后可行。**

项目当前的采样计划只有决策日期和 DOY，没有决策时刻；站点表有经纬度，但没有时区。因此，原句中的“决策时刻之前”当前无法由代码判断。

根据十个决策日约 5 小时的完成时间，采用以下规则具有事实依据：

```text
decision_time_local = 06:00
forecast_cycle = same-date 00 UTC
cycle_complete = 31 members × required variables × f003-f180 all present
```

当地 06:00 为统一的信息截止时刻，明显晚于本次实查的完整发布时间。后续仍必须对每个周期读取实际对象状态，不能把“约 5 小时”写成永远不变的产品保证。

### 4.2 集合成员处理

判定：**集合均值天气可算，但不推荐作为主方案。**

原因不是计算困难，而是非线性：

```text
model(mean(weather_members)) != mean(model(weather_member))
```

平均降水可能把“部分成员暴雨、部分成员无雨”变成一场不存在的中雨。土壤水分、收益和灌溉最优量都对天气具有非线性响应。

项目原始决策代码也先保存每个集合成员的最优结果，再对灌溉建议求平均，见 `Main_win.py` 的成员结果汇总逻辑。

修订方案：

1. 31 个成员作为独立天气情景输入同一个代理模型；
2. 主方法选择使 31 个成员平均预测收益最大的单一灌溉量；
3. 论文对齐基线分别优化成员灌溉量，再对 31 个成员建议求平均；
4. “集合均值天气输入”保留为消融基线；
5. 成员标准差、分位数和成员有效数写入审计表。

该方案会增加约 31 倍推理行数，但代理模型前向计算成本很低，远小于 31 次 SWAP 运行，并保留了集合不确定性。

### 4.3 当地日期映射

判定：**增加站点时区和区间分配规则后可行。**

当前站点表缺少 IANA 时区。GEFS 时间为 UTC，而美国夏令时下当地午夜不一定落在 3 小时输出边界上。因此不能只对 `valid_time.date()` 分组。

修订方案：

- 为每个站点保存 IANA timezone，例如 `America/Chicago`；
- 瞬时变量按有效时刻转换到当地时间后聚合；
- 降水和辐射等区间量按预测区间与当地日的重叠比例分配；
- 目标窗口固定为当地 `D 00:00` 至 `D+7 00:00`；
- 获取到 `f180`，保证西部站点的第七个当地日完整。

影响：跨日的 3 小时区间需要假设区间内均匀分布。该假设必须记录，但比直接用 UTC 日代替当地日更符合 SWAP 日尺度输入定义。

### 4.4 累计降水差分

判定：**原规则不可行。**

项目实际日期 `2024-07-16` 的官方 GRIB 索引为：

| 文件 | APCP 时间区间 |
|---|---|
| f003 | 0-3 h accumulated forecast |
| f006 | 0-6 h accumulated forecast |
| f009 | 6-9 h accumulated forecast |
| f012 | 6-12 h accumulated forecast |

来源：

- [f003 索引](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003.idx)
- [f006 索引](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f006.idx)
- [f009 索引](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f009.idx)
- [f012 索引](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240716/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f012.idx)

因此，直接计算 `f009-f006` 会拿 `6-9 h` 累计减去 `0-6 h` 累计，物理上错误。

正确重建为：

```text
0-3 h  = f003
3-6 h  = f006 - f003
6-9 h  = f009
9-12 h = f012 - f009
```

实现时必须解析 GRIB 的 `startStep/endStep`，按相同累计起点识别嵌套区间，再生成非重叠增量。负值容差应由实际 GRIB 打包精度确定；不能在读取数据前任意规定“微小负值”的数值。

### 4.5 缺失周期

判定：**整套周期回退后可行。**

不能让同一个集合中的部分成员来自当天 00 UTC，另一些成员来自前一天，否则集合成员不再属于同一次初始化。

修订方案：

1. 预检 31 个成员、全部变量和完整提前期；
2. 任一关键部分缺失时，整套回退到前一天 00 UTC；
3. 回退上限为 24 小时；
4. 前一天周期仍不完整时剔除该样本；
5. 禁止使用实际天气或其他周期的零散成员补齐；
6. 记录原周期、实际周期、回退原因、缺失对象和成员数。

## 5. 证据支持的正式协议

```text
decision_time_local: 06:00
forecast_cycle: same-date 00 UTC
required_members: c00, p01-p30
required_leads: f003-f180 at 3-hour intervals
ensemble_primary: maximize mean predicted gain over separate member scenarios
ensemble_paper_baseline: mean of member-optimal irrigation amounts
ensemble_mean_weather: ablation only
daily_window: site-local D 00:00 through D+7 00:00
precipitation: reconstruct intervals from GRIB startStep/endStep
missing_cycle: fallback the entire ensemble to previous 00 UTC, maximum 24 h
actual_weather_fill: forbidden
reproducibility: retain object paths, timestamps, ETags/checksums, variables, leads, and immutable subsets
```

## 6. 尚需通过数据预检验证的内容

当前证据足以判断方案可实施，但正式下载前仍应自动检查：

- 10 个决策日的 31 个成员是否全部覆盖 `f003-f180`；
- 每个提前期是否包含全部必需变量；
- 各站点 IANA 时区是否映射正确；
- APCP 全时段的 `startStep/endStep` 模式是否一致；
- GRIB 打包精度对应的负降水容差；
- 使用字节范围或空间裁剪后的实际下载量和运行时间。

这些属于可自动验证的数据质量问题，不需要凭经验预先假定通过。

## 7. 集合均值偏差实证更新（2026-07-15）

已完成 5 站点、10 个 2024 决策周期、当地 D 至 D+6 的 GEFS `geavg` 与 gridMET 对比。2100 条变量配对无缺失、无重复；600 个 GEFS 提前期下载单元全部完成。

集合均值降水的总体逐日 bias 为 `-0.0303 mm/day`，但这来自方向相反的条件偏差抵消：

- 干天平均高估 `1.0593 mm/day`；
- 小雨平均高估 `1.9853 mm/day`；
- 中雨平均低估 `5.8204 mm/day`；
- 大雨平均低估 `23.1697 mm/day`，12 条配对全部低估；
- 50 个 7 天窗口的降水 MAE 为 `10.4145 mm`，误差范围为 `-31.5333` 至 `+36.3333 mm`。

因此第 4.2 节的判断得到数据支持：`geavg` 可作为消融基线，但不应作为唯一主输入。正式方法保留 31 个成员情景。本诊断使用 gridMET 网格参考，不等同于站点实测真值；相邻 7 天窗口存在重叠，不能把 350 条逐日配对都视为独立事件。

完整证据见 `gefs_gridmet_bias_validation_analysis_2026-07-15.md`。
