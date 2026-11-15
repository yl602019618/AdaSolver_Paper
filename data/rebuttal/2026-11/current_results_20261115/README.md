# 当前结果与测试内容 / Current verified results

这是2026年11月15日汇总的已核验结果快照。各实验保留独立的训练 cohort、checkpoint、测试人群、采样方法和重建/计时边界。

- [测试了什么、还没测试什么](TESTED_CONTENTS.md)
- [测试覆盖表](test_coverage.csv)与[去重后的模型/head清单](model_inventory.csv)
- [Elasticity / Pipe / 历史Darcy：三独立训练种子](standard/README.md)
- [Darcy85×85：Transolver3、官方LinearNO point与conv Full-only](darcy/README.md)
- [ShapeNet-Car与Airfoil：完整场精度、重建、A/B/C时延、显存与缓存](reconstruction_airfoil/README.md)
- [既有score/density/实际选点证据](../linearno_followup_20261114/score_visualization/README.md)

三训练种子SD与单模型的sampling SD分开。Full100%不放入rAUEC积分，各模型使用自己的matched Uniform分母。Darcy1000/200历史cohort与900/100新matched cohort不合并；conv为Full-only参考，不当成第三个training seed。

本次恢复了Airfoil完成测量后的统计汇总错误，也补齐了Darcy原先待独立聚合的标量验收。原失败日志、历史test-informed边界和不利结果均保留。阶段profiling不相加冒充request wall time；缓存复用估计也不冒充连续多请求实测。

仍未完成的新matched training seeds1/2、Car LinearNO PI/CA扩展及FZ2未决项单列在测试索引中。补种子调度器的GPU利用率解析错误已修复并恢复，新的运行中任务不作为本次已完成结果。

模型权重、原始CFD场、完整预测NPZ、缓存和原始长日志保留在外部。这里发布标量结果、必要原始标量重复值（gzip CSV）、源记录索引、配置、脚本、图表和核验记录。
