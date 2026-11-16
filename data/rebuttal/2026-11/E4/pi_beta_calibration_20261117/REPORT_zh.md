# Darcy PI：β校准后的真实结果

已完成固定候选网格的校准和原200-case完整测试。**共享β=1使三个分辨率的平均rAUEC都低于1。** 这是重新执行模型推理与原重建规则得到的结果；原E4 β=0.5记录保留为对照。

## 主要结果

| 观测分辨率 | 原β=0.5 | 校准共享β=1 | 共享β中低于1的sampling seeds | 分辨率校准β | 分辨率校准rAUEC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 85² | 0.996622 ± 0.018057 | 0.998995 ± 0.019993 | 1/3 | 0.625 | 0.993774 ± 0.018484 |
| 141² | 1.000776 ± 0.005778 | 0.971548 ± 0.003287 | 3/3 | 1.5 | 0.970917 ± 0.000803 |
| 211² | 1.003448 ± 0.013816 | 0.969888 ± 0.012662 | 3/3 | 2 | 0.959246 ± 0.010756 |

±表示同一个冻结训练模型下3个sampling seeds的样本标准差，不是training SD。85²共享β1虽然均值为0.998995，但比原β.5的0.996622略差，且仅1/3采样seed低于1；不能写成三个分辨率均有稳定改善。141²/211²共享β1的三个采样seed均低于1。

## 选择和评价如何分开

- 原模型仍是E4的85² full-mesh Transolver3、training seed0，checkpoint/normalizer保持原身份。
- 校准集为真实smooth2 rows400–499，共100例，global IDs1400–1499；与最终200例rows0–199不重叠。
- 在校准集固定评估β=[0,.125,.25,.375,.5,.625,.75,.875,1,1.5,2]，共32700条Full/Uniform/PI记录。
- 共享参数按三个分辨率最差的校准mean rAUEC最小选择；次级参数各自在校准集选择。两者均在测试前写入不可变selection.json。
- 最终测试22200条记录，包含所有选定β、原β.5、匹配Uniform和Full。测试结果不改变主方案或重新选β。
- 每个β使用完全相同的选点顺序；原prior、amp、ell、重建网格和规则均保持不变。这里只改(1/raw prior weight)^β的指数。
- 原checkpoint及旧PI策略有历史test-informed选择记录，因此不将旧200例称为从未查看的holdout。

## 统计与解释

对每个sampling seed先计算200例平均误差曲线的面积，除以相同分辨率、checkpoint、人群和重建下的Uniform面积，再对三个比值求mean/sample SD。积分预算为37.5/50/75%，Full100%不进入积分。

所有校准候选和最终预先指定条件都保留，包括大于1的数值及相对原β.5的退步。平均低于1与每个seed低于1、显著优于Uniform是不同结论。β1.5/2作为预先声明的过补偿候选，不能称为更精确的inclusion-probability修正；weighted without replacement的真实最终包含概率没有在此重新估计。

这两套策略都使用了三个分辨率的校准信息。共享β不随测试分辨率改变，但也不应改称完全未校准的zero-shot分辨率迁移。原E4冻结β.5结果与本校准补充应区分展示。

## 可用配置与文件

- [共享β=1配置](config/primary_shared_beta1.json)
- [按分辨率校准配置](config/secondary_resolution_beta.json)
- [所有β校准结果](results/validation_rauec.csv) · [最终测试结果](results/test_rauec.csv)
- [最终对照图](figures/test_beta_comparison.png) · [同seed配对变化](figures/paired_beta_change.png)
- [校准独立审计](results/independent_validation_audit.json) · [最终统计核验](results/final_statistics.json)
- 原始逐case记录：`records/validation_r*.csv.gz` 和 `records/test_r*.csv.gz`；冻结选择：`selection.json`。

测试分片只改变执行调度，250个已完成case文件原样复用；数值Runtime类AST未变，protocol/selection hash未变。此过程未做新的模型训练或正式时延基准。

## 与原E4其它方法的关系

在211²上，新校准PI（共享β1）的mean rAUEC为0.969888，低于原冻结CPG的0.976785和CA的0.976940；因此若将新PI放入对照表，平均值排序也要据实更新。这里的PI获得额外校准，而CPG/CA保留原冻结配置，需标注协议差异，不能把这组更新写成原冻结策略的结果。各方法的sampling SD仍需同时展示；此处不作显著性结论。85²和141²下，原CPG/CA均值仍更低。

数据：[context_with_frozen_E4.csv](results/context_with_frozen_E4.csv)。

Public reproduction: `python3 verify_published.py --package . --output /tmp/darcy-pi-beta-check.json`. Raw records are lossless gzip; the verifier materializes the original CSV bytes in a temporary directory and uses the unchanged statistics source.
