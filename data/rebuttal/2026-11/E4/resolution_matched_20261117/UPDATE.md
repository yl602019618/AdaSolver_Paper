# 本次发布的实验类型与修改

实验类型：Darcy observation-density resolution-matched training。

- 新增真实 141²、211² 从头训练模型及同协议 85² 内部对照，全部采用同一 train900/validation100 case 划分、1000 epochs、有效 batch 4 和 training seed 0。
- 为三个冻结 backbone 分别训练并绑定独立 CA head。修正 head 验收器的 batch 8 复现方式，保留原差异诊断；已训练参数与策略数学规则未改变。
- 完成 Full、Uniform、PI、CPG、CA 的全部 200-case × 预算 × sampling-seed 评价，公开全部逐 case CSV 和 retained-index hashes。
- 分开报告共同 211² 与 native-grid 的绝对误差、逐 seed rAUEC 和 mean/sample SD，并给出边界、界面、平滑区的实际采样统计。
- 区域统计单独纠正为原生坐标上的严格 0.02 物理带宽；保留原最近邻布尔标签 CSV 为 legacy。该修正不改变模型、选点、field/rAUEC，也不归因于训练分辨率。
- 发布 13 组 PDF/SVG/PNG、绘图源 CSV、训练曲线、实际代码、协议与验收记录。
- 作者主表只放原文 85² 历史参考及新增 141²/211²；受控趋势包含新的 85²。
- 保留原 E4、PI β 校准、Airfoil 和 ShapeNet-Car 补充及远端其他更新。本次不修改 main 或论文主表源文件。

结果按实测报告，不预设高分辨率训练或任一采样策略必然改善。三个 sampling seeds 不称为三个 training seeds。
