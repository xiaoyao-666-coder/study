# 下一步：运行同一专家的站外反馈收益修正

状态：代码、来源身份、人工 60 方向/180 阶段、Linux 生命周期与干净解压校验均通过。尚未服务器真实计算，尚无科学结果。

交付 ZIP：`D:/study/s2s_rtist_source/swap_model_matched_feedback_20261113_v1.zip`，1,668,200 bytes，976 成员。

SHA-256：`0a4858cfac5813577327ef74d0189ab19e501327167f41bf4c632c9100818250`。

固定问题和边界见 [DESIGN.md](DESIGN.md)。30 个三站专家、60 供体→目标方向，2014–2015 来源监督，2016 评价。64 个物理周期，59 个具有成熟支持。反馈只取已执行动作，收益标签在全局预测封存后解锁。不是把已做 crossfit/specificity 改名，也不是 R1 取水回归：此处固定原 GV 专家身份并直接检验决策收益。仅属受限开发变体，无成功保证、无因果归因或新算法声明。

上传 ZIP 至服务器项目根，执行：

```bash
mkdir -p /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy && unzip -n /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/swap_model_matched_feedback_20261113_v1.zip -d /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy && bash /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy/swap_model_matched_feedback_20261113_v1/start.sh && tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/swap_model_matched_feedback_20261113_v1.log
```

Ctrl+C 只退出 tail。成功或失败均尝试生成项目根 `swap_model_matched_feedback_20261113_v1_r001_results.zip`。重启保留旧尝试并递增编号；已经完成会提示 ALREADY_COMPLETE。8 workers/BLAS1/30 秒心跳/worker 1800 秒/总 10800 秒；无额外 nohup。

回传后先验证 delivery、package manifest、result manifest、180 阶段封存和两道全局屏障；再检查原专家重建、供体/目标隔离、反馈 release、优化目标和 PG≤1e−6。复核 output/aggregate/summary.csv、contrasts.csv、sites.csv、seeds.csv 与 screening.json；从保存预测/标签独立算术核对，禁止本地重新 forward/优化。科学失败按设计关闭，不能按目标挑供体、seed、反向或调参。采用指针仍 shared_advantage_gate，不自动合并或晋升。

旧 context-atoms 的失败验收和前提勘误仅在其本地分支保留，未推送新失败提交。新分支从 origin/main 创建，只含本候选；不合并。history、共享 Word 和记忆未更新。
