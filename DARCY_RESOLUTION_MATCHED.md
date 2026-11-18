# Darcy 分辨率匹配训练：本次发布

**实验类型：Darcy observation-density resolution-matched training。**

[完整实验与结果](data/rebuttal/2026-11/E4/resolution_matched_20261117/README.md) · [修改说明](data/rebuttal/2026-11/E4/resolution_matched_20261117/UPDATE.md) · [作者主表](data/rebuttal/2026-11/E4/resolution_matched_20261117/results/tables/rebuttal_main.md) · [受控趋势](data/rebuttal/2026-11/E4/resolution_matched_20261117/results/tables/controlled_common.md) · [全部图](data/rebuttal/2026-11/E4/resolution_matched_20261117/figures/README.md)

本轮新增 141²/211² 从头训练及同协议 85² 内部对照；三组均为 train900/validation100、1000 epochs、有效 batch 4、training seed 0，每个 backbone 有自己的 CA head。完整评价为 22,200 个预测请求和 44,400 行 common/native 指标，sampling seeds 为 0/1/2。

修复内容包括原生分辨率数据/训练适配、同分辨率 CA teacher/head、完整输出评价、逐 seed rAUEC 与区域采样分析，以及 head 验收的原 batch 8 复现方式。区域统计另纠正为原生坐标上的严格物理带宽，保留旧最近邻布尔标签记录为 legacy；训练、实际选点和 44,400 行 field 误差不变。主表以原文 85² 为历史参考，受控趋势另含新的 85²。

原 E4、PI β 校准及远端已有 Airfoil、ShapeNet-Car 工作均保留。本次更新实验分支，不修改 main 或论文主表源文件。
