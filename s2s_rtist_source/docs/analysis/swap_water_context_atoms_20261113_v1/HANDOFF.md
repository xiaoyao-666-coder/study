# 2026-11-13 情境响应原子实验交付

当前状态：`swap_water_context_atoms_20261113_v1` 已完成实现、人工验证和交付打包，等待用户在服务器运行并回传。没有本轮真实模型运行或效果证据。上下文恢复以旧对话 `01a1cf0b-f793-7ed2-b026-ad8009f0116e` 为准，共享 Word 与项目记忆中的旧状态不作为当前进度。

## 已结束与当前问题

旧对话确认 H1 停止、R1 关闭；D1 四种反馈对照完成 351 个任务，均未通过预先规定的稳定改善门槛，固定三特征迁移方法族关闭。原有结构审计已完成。

本轮只检验：先由公开早期情境形成响应原子的先验，再用一个已执行早期剂量的成熟七日储水反馈更新权重，能否改善后续未观测剂量之间的累计七日原生蒸腾差。每个外层站点使用其他四站的八个 2014–2015 站点年份原子；残差与目标预测严格绑定同一个排除目标站及捐赠站的基模型。十一项情境、权重公式、反馈尺度和筛选门槛见包内 `CONTRACT.md`。

主比较为 `on` 对 `off`，`raw` 仅作参照。每种行为排除该反馈剂量，两端都来自剩余七剂量，21 对/行为、80 个行为版本、1680 对；只有十个独立后期物理窗口。按行为→窗口→站点汇总，同时报告分年、去掉最大受益窗口和日尺度/符号/非负性检查。不做经济评分，不自动晋升，不重跑 SWAP。

## 工作树整理

- 新分支 `codex/water-context-atoms` 从 `origin/main` 的 `fb5085948ad0c1611941c4bf53bebc3d6344fb6f` 建立。
- 工作树：`C:/Users/LEGION/.codex/worktrees/water-context-atoms/study`。
- 仅迁移当前方法的源码；旧分支的 34 个未推送提交和其他修改保留，没有混入此分支。
- 原候选草稿及整理前补丁保存在 `D:/study/tmp/water_context_atoms_draft_20261113/`。未删除旧成果。
- 分支包含源码、合成测试、检查记录及交接说明；生成的真实数值载荷留在本地交付 ZIP，不入 Git。未合并。

## 交付与验证

ZIP：`D:/study/s2s_rtist_source/swap_water_context_atoms_20261113_v1.zip`，319597 字节、117 个成员。

SHA-256：`6ea0036d1d723e891b4be57a7bfb7e7f155f2eff663052695a0958152e91d94c`。

清单 SHA-256：`15c7b254b9c648c90329c541c87169a096b9c2bde0f7e1b850e831c618d92d57`。

- 认证 R1/B0/D1 保存资产，按固定字段投影；710 个受保护输入哈希保持不变。没有本地真实残差计算、拟合、前向或评分。
- 人工数据跑通 96 个生产 worker；独立核算先验、似然、后验、条件分支及评分。覆盖常量情境、下溢、反馈方向、行为隔离和篡改拒绝。
- WSL Ubuntu 人工进程跑通 96 任务；验证四阶段失败、坏封存、工作进程/总超时及后代终止、重复启动锁、失败尝试保留、启动失败回包和最终证书篡改拒绝。
- 固定契约、精确输入字段、Python 3.10 语法、LF shell、ZIP CRC/精确清单/解压逐字节一致性通过。本地真实 worker 在读数值文件前被路径守卫拒绝。
- 详细凭据：`asset_validation.json`、`synthetic_validation.json`、`runtime_validation.json`、`contract_validation.json`、`release_validation.json`、`build.json`。

## 运行习惯与下一步

用户自行上传、执行和回传；不 SSH、不安装。将 ZIP 上传到服务器项目根目录后，运行以下单行命令。8 workers、BLAS=1、30 秒心跳、单任务 1800 秒/总 10800 秒限制，启动器负责后台执行；Ctrl+C 只退出日志跟随。

```bash
mkdir -p /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy && unzip -n /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/swap_water_context_atoms_20261113_v1.zip -d /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy && bash /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy/swap_water_context_atoms_20261113_v1/start.sh && tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/swap_water_context_atoms_20261113_v1.log
```

等日志出现 `RESULTS_SAVED ..._rNNN_results.zip`，将对应文件传回。成功和运行失败均自动打包。回包后先核对 ZIP、包身份、阶段封存、读取账本及 96 个任务完整性，再审计 `output/score/global/result.json`；明确区分工程完成与方法通过。阴性结果关闭这个固定候选，不针对目标结果调参。

## 复现开发检查

在新的干净目录中执行 `prepare.py --source-root D:/study/s2s_rtist_source`，要求三个历史包的固定 manifest 与实际文件一致。准备脚本只用标准库投影保存字段。随后依次运行 `test_synthetic.py`（需要已有 NumPy）、`test_runtime.py`（Linux 人工进程）、`validate_contract.py`、`package.py --output-dir <交付目录>` 和 `verify_release.py`。脚本为 create-only，保留现有验证记录和交付物，不覆盖重跑。真实计算必须在固定 Linux 部署路径执行。
