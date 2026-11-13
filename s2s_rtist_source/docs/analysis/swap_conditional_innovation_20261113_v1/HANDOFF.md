# 条件创新量候选交接

2026-11-13。实现、纯人工验证和交付包已完成。尚未运行真实实验，没有收益改善结论。当前采用仍为 `shared_advantage_gate / swap_a1_advantage_gate_20261019_v1`。

本轮只回答：扣除原专家合法上下文 C 可预测的水分残差后，成熟已执行动作反馈的剩余创新，能否比同 C 的收益基线更稳定地改善后续灌溉净收益排序。2014 供体拟合上下文变换及水分条件岭预测，2015 同供体先拟合并冻结 C-only 收益基线，再拟合反馈头；2016 另一排除站评价。固定 30 原专家、60 方向，五方法 `raw/static/on/reverse/uncentered`。完整公式及固定筛查见 [DESIGN.md](DESIGN.md)。

主比较 `on-static`，同时必须优于 raw、reverse、uncentered，满足逐站、逐 seed、去最大受益周期、留一站及风险门。失败关闭；不选控制组、不扩参、不自动采用。2016 是反复使用的开发池，结果是固定行为回放，不是新策略真实闭环。

## 已完成验证

- 全人工梯度与独立 SLSQP 检验；条件岭拟合与独立增广最小二乘一致；常数上下文和不罚截距；零创新回到 static、反向对照、零剂量锚点及幅度限制。
- 全人工 60 方向、180 阶段流水线；越权读取和封条篡改阻断。
- 未来反馈干预不影响早期预测；人工经济聚合正负例及版本重复不变性。
- 人工分年干预：2015 上下文/水分/收益改变不能改变 2014 nuisance；2014 收益改变不能改变任何最终拟合参数。后一年干预确实改变收益修正头。
- 真实 Linux 人工进程测试：重复启动锁、心跳、成功幂等、失败保留并重试、超时回包。
- 保存资产哈希与身份核验：30 专家、60 方向、64 物理周期、59 成熟周期、768 相关版本；新增 10 个目标 2016 输入/根区状态文件逐文件与来源绑定，键和状态时钟与保存上下文相符。
- ZIP 干净解压后精确成员、CRC、SHA-256、Python 3.10 语法、LF shell、ASCII 成员、无 pycache，且无 2016 labels.csv。

本地真实 forward、拟合、归一化估计、优化及新策略评分调用为 0。真实编码输入与已封存 pre/B、原收益及 probe 的重现留给服务器 predict 阶段核验。

## 交付身份

- 工作树：`C:\Users\LEGION\.codex\worktrees\water-conditional-innovation\study`
- 分支：`codex/water-conditional-innovation`，起点 `fb5085948ad0c1611941c4bf53bebc3d6344fb6f`（创建时 origin/main）。仅本候选目录，不合并旧失败提交。
- ZIP：`D:\study\s2s_rtist_source\swap_conditional_innovation_20261113_v1.zip`
- 大小：1,718,360 bytes；988 成员。
- ZIP SHA-256：`11570bd09f4eb2a336ebe368c9c83b8e24dc6b49457b7b5ae7beef7652b3ecb2`
- manifest SHA-256：`07596325386d583c19f4acc079200193344deb14a658e19d740ca21e684e42b6`
- `prepare.py build` 与 `prepare.py package` 已执行，均为 create-only，不重复执行，不覆盖旧文件。
- `.gitattributes` 禁止本目录换行转换，保持资产、验证收据与 Git 字节一致。

## 上传后启动

```bash
mkdir -p /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy && unzip -n /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/swap_conditional_innovation_20261113_v1.zip -d /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy && bash /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy/swap_conditional_innovation_20261113_v1/start.sh && tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/swap_conditional_innovation_20261113_v1.log
```

8 workers、BLAS1、30 秒心跳；单 worker 1,800 秒、全程 10,800 秒超时；source → 全局封存 → predict → 全局封存 → score。启动器自行转入后台，Ctrl+C 仅退出 tail。首次自动回包 `swap_conditional_innovation_20261113_v1_r001_results.zip`；失败也回包，重试保留旧结果并递增 rNNN。

用户说“已传回”后查找此回包，先审计 manifest、阶段封存、60 个方向完整性、原专家身份、条件岭方程证据与三个收益头收敛，再读 `output/aggregate/screening.json` 和 contrasts/sites/seeds。分别判断工程完成与科学筛查，不能因有结果 ZIP 就判实验成功。

本轮没有更新项目记忆、history 或共享 Word。旧 `water-model-matched` 工作树的失败关闭提交仍仅本地保留；没有推送废弃方法。
