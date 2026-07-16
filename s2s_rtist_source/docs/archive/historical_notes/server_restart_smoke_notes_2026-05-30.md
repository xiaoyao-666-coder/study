# 服务器 SWAP 续跑加速 smoke test 记录

日期：2026-05-30

## 目的

验证原论文灌溉决策中，是否可以把每个候选灌溉量的 SWAP 仿真从“整季重跑”改为：

1. 先跑到决策日前一天，保存 `.end` 状态；
2. 每个候选灌溉量从 `.end` 状态续跑 7 天；
3. 比较续跑结果是否保持原始整季重跑的决策排序。

如果可行，后续批量生成代理模型训练数据时，可以大幅减少 SWAP 运行量。

## 已验证环境

服务器路径：

```text
/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
```

Linux 版 `swap_test` 可以运行。服务器缺失的 `libgfortran.so.5` 已通过用户目录运行库解决。运行前需要设置：

```bash
export LD_LIBRARY_PATH="/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/local_libs/gcc_runtime/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```

注意：`swap_test` 正常完成时可能仍返回非零 exit code，例如 `100`。判断是否成功应优先看日志中的：

```text
Swap normal completion!
```

## 单候选 restart probe

测试候选：

```text
16-Jul-2024, 20 mm
```

结果：

```text
mode                end_daynr   dvs   CWDM    CWSO
full                    205    1.31   5912    1189
restart                 205    1.31   5931    1203
restart_minus_full        0    0.00     19      14
```

相对差异：

```text
CWDM: 19 / 5912 = 0.32%
CWSO: 14 / 1189 = 1.18%
DVS: 0
```

结论：`.end` 续跑技术上可运行，单候选误差较小。

## 8 候选 full vs restart 对比

脚本：

```text
D:\study\s2s_rtist_source\decision_restart_8ir_compare.py
```

服务器运行目录：

```text
/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/Maize_restart_8ir_compare
```

输出：

```text
decision_restart_8ir_compare.csv
decision_restart_8ir_full.csv
decision_restart_8ir_restart.csv
```

### full 整季重跑结果

```text
ir   CWDM   CWSO   target_value
0    5496    806       0.0
10   5733   1025      33.4
15   5856   1139      51.0
20   5912   1189      55.2
25   5941   1213      54.0
30   5957   1226      50.2
40   6010   1264      46.8
60   6036   1282      24.0
```

full 最优候选：

```text
20 mm, target_value = 55.2
```

### restart 7 天续跑结果

```text
ir   CWDM   CWSO   target_value
0    5497    807       0.0
10   5743   1031      35.2
15   5864   1144      52.4
20   5931   1203      58.8
25   5965   1231      58.6
30   5986   1247      55.8
40   6010   1264      46.6
60   6036   1282      23.8
```

restart 最优候选：

```text
20 mm, target_value = 58.8
```

### 差异

```text
ir   CWDM_diff   CWSO_diff   DVS_diff   target_diff
0         1          1          0.0         0.0
10       10          6          0.0         1.8
15        8          5          0.0         1.4
20       19         14          0.0         3.6
25       24         18          0.0         4.6
30       29         21          0.0         5.6
40        0          0          0.0        -0.2
60        0          0          0.0        -0.2
```

## 当前判断

1. `.end` 续跑能成功运行。
2. 对 2024-07-16 这个决策日，8 个候选量中 full 和 restart 的最优灌溉量一致，都是 `20 mm`。
3. restart 与 full 并非逐值完全一致，`CWDM` 最大差异为 `29 kg/ha`，`target_value` 最大差异约 `5.6`。
4. 由于 `20 mm` 与 `25 mm` 的目标值本来就非常接近，restart 下二者差距只有 `0.2`，说明后续需要在更多日期和 ensemble 上检查排序稳定性。

## 下一步建议

先不要直接把 restart 当作正式数据生成方案。建议先做一个小规模稳定性测试：

```text
3 个决策日 × 8 个候选量 × full/restart 对比
```

评价：

```text
最优灌溉量是否一致
top-2 是否一致
target regret
CWDM/CWSO/DVS 误差
运行时间节省比例
```

如果多个日期上最优候选和目标函数排序都稳定，再把 restart 方案用于批量代理模型数据生成。
