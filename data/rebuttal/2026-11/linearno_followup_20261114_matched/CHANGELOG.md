# 本分支改动清单

本分支以已核验的 `91c1fa7cc81b8c1cb61a43618296c1c892d818c0` 为基线，只新增 `data/rebuttal/2026-11/linearno_followup_20261114_matched/`。

| 类型 | 修改 | 验证 |
| --- | --- | --- |
| Reconstruction implementation | 规则保持的 sparse apply；无插桩主请求；显式 A/B/C 边界；几何与选点相关缓存 | 原有6个契约样例、108组GPU固定预测；新Airfoil完整测试old/new |
| Baseline extension | 官方LinearNO point-temp、conv-temp身份分开；变量点集与补偿位置核对 | 实际类接口、单位权重/复制质量测试；源码版本登记 |
| Matched training | 共同900/100 split；train-only normalizer；seed0匹配schedule；分别训练CA head | 最终优化步、选择epoch、权重/normalizer/head身份及全部head bank记录 |
| Mechanism ablation | CPG beta0/1仅改变补偿；固定coarse/final points | 1980组paired coarse/index/prediction hashes |
| Accuracy evidence | 两个Airfoil主模型完整Full/U/PI/CPG0/CPG1/CA矩阵 | 110case；三个预算；三个sampling seeds；完整原始记录 |
| Protocol correction | Darcy PI二维EDT作为预先声明的新策略；双方统一物理场loss，Darcy含0.1 derivative项 | 已锁定，Darcy精度仍待运行；不复用不匹配旧结果 |
| Statistics and figures | seed-first rAUEC、absolute errors、自动表图、独立CSV重算 | 20行rAUEC、66行absolute汇总；实际渲染QA |
| Persistence fixes | 不可变阶段CSV、原子发布、重复worker锁、partial失败观测保留 | 文件发布和CPU故障注入记录，明确不当作性能数据 |
| Candidate bug fixes | exact-kNN budget guard、CPU副本匹配、k与finite检查 | 有效域CPU复验、非法输入提前拒绝、真实Car fixture |

补充完成：Airfoil官方conv-temp Full参照，单列Full-grid训练协议和TF32设置。

待完成：Darcy新训练和精度；Car全100-case复评；全部正式A/B/C时延与显存矩阵。因此本分支不声称整轮完成，也不报告新端到端加速结论。图表中的Darcy缺项标为N/A/pending。

采样误差条为同一training-seed0模型的三个sampling seeds，不是training SD。Airfoil Cd/Cl相关原始数值是field-target策略的补充诊断，未与其他target的配置拼接。
