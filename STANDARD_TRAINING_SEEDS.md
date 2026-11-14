# Standard benchmarks: three independent training seeds

## 本分支的内容

分支：`rebuttal/20261114-standard-training-seeds`。

在已有 ShapeNet-Car E5 实验基础上，补充 **Elasticity、Pipe、Darcy** 三个标准算例。
每个算例重新训练 3 份完整 Transolver3 骨干及其对应 CA head，共 **9 个骨干、9 个 head**。
每份模型使用同一组 200 个测试案例，评估 Full，以及 Uniform、PI、CPG、CA 的
37.5%、50%、75% 点预算；自适应预算使用 3 个推理采样 seed。

**实验与数值核验已完成：27/27 个执行阶段，66,600 条逐案例记录。**
本补充实验的主要交付是结果表、可机读指标、训练配置、checkpoint 索引和复现脚本。

## 核心结果

以下为 rAUEC 的 **均值 ± 训练 seed 样本标准差**，越低越好；Uniform 为 1。
标准差基于 3 个独立训练运行。推理采样方差在单独的 CSV 中报告。

| Benchmark | PI | CPG | CA |
| --- | ---: | ---: | ---: |
| Elasticity | 0.7264 ± 0.0022 | 0.4670 ± 0.0012 | 0.5601 ± 0.0106 |
| Pipe | 0.8557 ± 0.0127 | 0.6544 ± 0.0101 | 0.6923 ± 0.0094 |
| Darcy | 0.9960 ± 0.0028 | 0.6393 ± 0.0159 | 0.6394 ± 0.0164 |

Elasticity 和 Pipe 中，CPG 在三个训练 seed 下均取得最低 rAUEC。
Darcy 中，CPG 与 CA 的平均值仅差约 0.000122，二者排序随训练 seed 变化。

## 文件索引 / Contents

| 文件 | 内容 |
| --- | --- |
| [完整结果报告](data/rebuttal/2026-11/E5_standard/README.md) | 总表、逐训练 seed 表、各预算绝对误差、实验协议与结果解读 |
| [training_seed_rauec.csv](data/rebuttal/2026-11/E5_standard/training_seed_rauec.csv) | 12 行：3 个算例 × 4 种采样方法，包含 seed 0/1/2、均值及训练 SD |
| [training_seed_summary.csv](data/rebuttal/2026-11/E5_standard/training_seed_summary.csv) | 39 行：各方法、各预算的完整场相对 L2 误差及训练 SD，含 Full 参照 |
| [sampling_variability.csv](data/rebuttal/2026-11/E5_standard/sampling_variability.csv) | 117 行：各训练模型内部的推理采样均值和采样 SD |
| [paired_training_differences.csv](data/rebuttal/2026-11/E5_standard/paired_training_differences.csv) | 54 行：按训练模型配对的方法差值 |
| [metrics_records.csv.gz](data/rebuttal/2026-11/E5_standard/metrics_records.csv.gz) | 66,600 条逐案例观测，包含误差、分子/分母、seed 和 checkpoint 身份 |
| [checkpoint_registry.csv](data/rebuttal/2026-11/E5_standard/checkpoint_registry.csv) | 9 对骨干/head 的身份、训练轮数、验证选择 epoch、训练成本和索引 |
| [训练与采样配置](reproducibility/rebuttal/2026-11/E5_standard/config) | 固定数据划分、模型、训练、head、采样策略和统计口径 |
| [复现说明与脚本](reproducibility/rebuttal/2026-11/E5_standard/README.md) | 实验驱动、统计生成，以及无需模型权重的独立表格复算 |
| [LaTeX 总表](data/rebuttal/2026-11/E5_standard/standard_training_seeds.tex) | 可用于论文或回复材料的 rAUEC 表格源码 |
| [数值核验](data/rebuttal/2026-11/E5_standard/verification.json) | 完整矩阵、checkpoint 配对、采点与统计计算的检查结果 |
| [发布清单](rebuttal/2026-11/E5_standard_RELEASE_MANIFEST.json) | 本补充包的文件分类、大小和内容校验值 |

## 训练与统计口径

- Elasticity 骨干训练 500 轮，head 训练 200 轮；Pipe、Darcy 的骨干与 head 各训练 1000 轮。
- 三个 Elasticity head 均采用 FP32 CPU 特征提取与训练；其完整结构为 `128→4→1`。Pipe 和 Darcy 保留完整的两块 Transolver head，在 GPU 上训练。
- 骨干与 head 的 checkpoint 由与测试集分开的验证样本选择。既有采样策略保持固定；Pipe、Darcy 历史策略选参涉及测试数据的范围已在完整报告中说明。
- 对每个采样 seed 分别计算与 Uniform 配对的误差面积比，在每个训练模型内平均，再对三个独立训练值计算均值与样本 SD。
- 独立复算 66,600 条记录和 51 行汇总表，最大数值差为 `2.22e-16`。
- 大权重和完整采样点数组保留在实验工作区；仓库中的 registry 给出相对索引和身份信息。

从仓库根目录复算已发布的数值表，无需模型权重或 AdaSolver 源码：

```bash
python reproducibility/rebuttal/2026-11/E5_standard/code/recompute_tables.py \
  --data data/rebuttal/2026-11/E5_standard
```

## Related E5 experiment

The companion ShapeNet-Car three-training-seed experiment is documented in
[the prior E5 package](data/rebuttal/2026-11/E5/README.md).
The full English protocol and detailed numeric tables for the new standard
benchmarks are in [the results report](data/rebuttal/2026-11/E5_standard/README.md).
