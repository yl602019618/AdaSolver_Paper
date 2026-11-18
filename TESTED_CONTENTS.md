# 已完成结果与测试内容索引

<!-- darcy-resolution-matched-training-20261117 -->
**Completed Darcy resolution-matched training.** [data/rebuttal/2026-11/E4/resolution_matched_20261117](data/rebuttal/2026-11/E4/resolution_matched_20261117/README.md): three native-resolution Transolver3 backbones and three matching CA heads, all trained for 1000 epochs on the same contiguous 900/100 split with seed 0. Full/U/PI/CPG/CA cover the same 200 physical test cases; 22,200 requests, 44,400 common/native metric rows and sampling seeds 0/1/2. This supplements the earlier frozen-85 observation transfer; it does not replace that protocol or add three training seeds.
<!-- /darcy-resolution-matched-training-20261117 -->

<!-- linearno-car-pi-ca-20261117 -->
**ShapeNet-Car LinearNO PI/CA complete (training seed0).** [Results and method scope](data/rebuttal/2026-11/linearno_car_pi_ca_20261117/README.md) · [Summary](LINEARNO_CAR_PI_CA.md). The native official backbone now has geometry-only PI and its own trained CA pressure head, separate Field/Cd policies,100-case accuracy and222 formal complete-CPU-field timing conditions. Error bars here measure sampling variability; independent training seeds1/2 are not included.
<!-- /linearno-car-pi-ca-20261117 -->

<!-- airfoil-seeds123-posthoc -->
**Exploratory display: training seeds 1/2/3.** Seed 0 was excluded after its test performance was observed, while its valid original data remain preserved. [Subset results](data/rebuttal/2026-11/p0_airfoil_original_cdcl_seeds123_posthoc_20261117/README.md) · [Displayed values](RESULTS_DETAIL.md) · [English response](reviews/airfoil-seeds123-posthoc-response.md). This selected display is not an unselected three-seed stability validation.
<!-- /airfoil-seeds123-posthoc -->

<!-- darcy-pi-beta-calibration-20261117 -->
**Darcy PI calibration is complete.** [Package](data/rebuttal/2026-11/E4/pi_beta_calibration_20261117/README.md): one frozen E4 checkpoint,100 disjoint calibration cases,200 historical test cases,32,700 calibration rows and22,200 test rows. Shared beta1 gives three means below1, with the85² regression and seed-level values retained. Original fixed-beta E4,900/100 matched Darcy and three-training-seed Darcy remain distinct scopes.
<!-- /darcy-pi-beta-calibration-20261117 -->

<!-- darcy-resolution-analysis-20261117 -->
**Current E4 analysis, 2026-11-17.** [Darcy observation-resolution report](data/rebuttal/2026-11/E4/README.md) replaces the previous brief explanation. The same27,000 requests remain the evidence base. New derived rAUEC, fixed5419 coarse composition and region error-energy tables are independently checked; no new training, inference or formal timing is claimed. This is a single85²-trained checkpoint, distinct from both Darcy three-training-seed and900/100 matched cohorts.
<!-- /darcy-resolution-analysis-20261117 -->

<!-- original-cdcl-correction-20261117 -->
**Latest Airfoil update, 2026-11-17.** The current original-Cd/Cl-policy result is [p0_airfoil_original_cdcl_20261117](data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/README.md).

| Current scope | Independent training coverage | Evidence | Cd/Cl interpretation |
|---|---|---|---|
| Corrected Transolver3 original Cd/Cl protocol | Seeds 0/1/2; three matching CA heads | 11,880 fresh records; 110 cases, three sampling seeds, three budgets, four methods; original control 3,960 records reproduced exactly | Active original-policy result; training SD and sampling SD are separate |
| P0 Airfoil field accuracy and formal timing | T3 and LinearNO each seeds 0/1/2; six models and six heads | Existing P0 verification: 30,360 requests, 4,914 timing/memory conditions | P0 Cd/Cl rows are field-target diagnostics and cannot be compared with original Table 1 |

The 19-model inventory and seed0-only tables below preserve the **2026-11-15 historical snapshot**. Later Airfoil completion is recorded above. For current Cd/Cl numerical values use [RESULTS_DETAIL.md](RESULTS_DETAIL.md) and the corrected package.
<!-- /original-cdcl-correction-20261117 -->

本次按 **checkpoint SHA256 去重，登记 19 个已完成 backbone 和 16 个对应 CA head**。统计对象是本次跟进已核验的模型；测试分组之间存在结果与 checkpoint 复用，观测条目不合并当作独立样本。
索引快照：`2026-11-15T13:15:31.511555+00:00`。模型明细见 [model_inventory.csv](data/rebuttal/2026-11/current_results_20261115/model_inventory.csv)，机器可读测试矩阵见 [test_coverage.csv](data/rebuttal/2026-11/current_results_20261115/test_coverage.csv)。

| 模型组 | 去重 backbone / head | 已完成 training seeds | 关键区分 |
|---|---:|---|---|
| ShapeNet-Car E5/E3 | 4 / 3 | T3：0、1、2；官方 LinearNO：0 | E3 的 T3 seed0 就是 E5 seed0，同一 checkpoint 只计一次；本历史清单尚不含后来新增的 LinearNO head；seed0 PI/CA 最新完成情况见上方 |
| Elasticity 标准 cohort | 3 / 3 | T3：0、1、2 | 每模型 1000 train / 200 validation / 200 test；三份 head 均完成 CPU FP32 训练 |
| Pipe 标准 cohort | 3 / 3 | T3：0、1、2 | 同上；保留历史 test-informed 策略 |
| Darcy 历史 1000/200 cohort | 3 / 3 | T3：0、1、2 | 旧登记的 seed0/1 与本轮 standard 结果按 hash 合并；不重复计数 |
| Airfoil 新 900/100 cohort | 3 / 2 | 每个变体均仅 0 | T3、LinearNO point-temp、LinearNO conv-temp 是三个变体；conv 仅 Full |
| Darcy 新 900/100 cohort | 3 / 2 | 每个变体均仅 0 | 两 point backbone 有自身 head；conv 仅 Full |

**新 matched Airfoil 与 Darcy 的三种模型变体不是三个 training seeds。** 这份 11 月 15 日历史快照只列出两个数据集的独立训练 seed0；其 sampling seeds 0/1/2 衡量同一模型内采样波动。Airfoil 后续三独立训练 seed 的完成情况及本次 Cd/Cl 修正见上方。

## 各组实际测了什么

| 分组 | 数据、模型与 seeds | 预算 / 指标 | 精度覆盖 | 时延覆盖 | 历史测试可见性 / 未测项目 |
|---|---|---|---|---|---|
| E5 · Car三训练seed | Car T3 ×3；training 0/1/2，每模型 sampling 0/1/2 | 37.5/50/75%，另 Full100%；Field/Cd、rAUEC、training/sample SD | 100 测试例；19200 原生记录 / 22200 指标记录；原冻结目标配置完整 | seed0 见本轮 Car 重建组；seed1/2 未各自正式计时 | 历史人群；旧 Cd CPG/CA beta0.75 保留；新的 CPG beta0/1 补矩阵尚待整体验收 |
| E3 · 官方LinearNO对照 | Car T3 seed0 + 官方 LinearNO seed0；sampling 0/1/2 | 三预算 + Full；Field/Cd、CPG beta0/1、各自 rAUEC | 100 例；9200 原生记录 / 11200 指标记录；Full/Uniform/CPG 对照 | 历史E3原生重建：276条件/27600正式重复，含beta0/1；新old/new测试另见Car组 | T3 checkpoint 与 E5 重叠；本历史 E3 快照仅含 Full/Uniform/CPG；seed0 PI/CA 与 head 已由新补充完成；独立 training seeds1/2 未纳入本包 |
| 三seed · Elasticity | 972 点 Elasticity；T3 training 0/1/2，sampling 0/1/2 | 三预算 + Full；完整场 L2、rAUEC、training/sample SD | 200 例；每模型7400条，合计22200；Full/Uniform/PI/CPG/CA | 本组这些 checkpoint 的正式完整输出时延未测 | 开发阶段冻结策略；没有本组 LinearNO 对照；CPU head 恢复证据已核验 |
| 三seed · Pipe | 16641 点 Pipe；T3 training 0/1/2，sampling 0/1/2 | 三预算 + Full；完整场 L2、rAUEC、training/sample SD | 200 例；合计22200条；五方法完整 | 本组这些 checkpoint 的正式完整输出时延未测 | 历史策略用过当前测试集前50例；没有本组 LinearNO 对照 |
| 三seed · Darcy历史cohort | 85×85；历史1000/200；T3 training 0/1/2 | 三预算 + Full；完整场 L2、rAUEC、training/sample SD | 200 例；合计22200条；五方法完整 | 本组这些 checkpoint 的正式完整输出时延未测 | 历史 test-informed；与新900/100 split/normalizer/loss 分组保留 |
| 完整输出 · Car | Car 两 backbone 均 training0；sampling0/1/2；本历史 A/B/C 矩阵仅 T3 含 PI/CA；LinearNO 的新完整 CPU 输出组见上方 | 三预算 + Full；Field/速度/压力/Cd；old/new parity、direct median/p95、显存 | 100例；7400 old/new对；1800 CPG beta对；field采样策略上的物理量诊断 | 3固定例；A/B/C共990条件、99000正式重复；990显存条件 | 历史人群；beta0只测精度；本历史 A/B/C 未含 LinearNO PI/CA；新独立 CPU 输出组已补齐；B/C 和多 training seed 不在本次范围 |
| 新匹配 · Airfoil | 221×51压力；900/100；T3、point-temp、conv-temp各training0 | 三预算 + Full；压力L2、Cd/Cl/CdCl误差、rAUEC、parity、完整输出成本 | 固定发布110个Cl>0.02例；point各5060请求，conv110Full，共10230 | 3固定例；333组/1323条件；132300正式重复；1323显存条件 | 历史测试可见；本历史快照只含seed0；后续P0已补齐三training seeds，original Cd/Cl另见本次corrected package；conv仅Full |
| 新匹配 · Darcy | 85×85；900/100；T3、point-temp、conv-temp各training0 | 三预算 + Full；场L2、rAUEC、beta0/1、parity、完整输出成本 | 200例；point各9200请求+conv200Full；37200 old/new指标行；3600 beta对 | 3固定例；333组/1323条件；132300正式重复；1323显存条件 | smooth2历史人群；point各training1/2及training SD未完成；旧1000/200三seed不能替代此组 |
| FZ2 · score可视化 | Car、Airfoil各1个固定例；已有T3 seed0/自身head；sampling0 | 50%；真实feature/head、score、density、selected、后置error诊断 | 逐点来源/hash/计算链核验；Car独立CPU CA闭环单列 | 完整请求时延与profiling现已完成；score捕获专属GPU cost/rawscore绑定尚未单独更新 | 单例可视化；Car CA跨设备score154点超阈保留；Airfoil CPG仅保留619/1127coarse |

## 时延与统计的读法

- **A**：CPU 输入 → 完整 CPU float64 场，batch=1、网络 FP32；每次变点集请求都计入重建映射准备。
- **B**：GPU-resident 输入 → 完整 GPU float64 场；请求中实际需要的 CPU 几何处理与传输仍计入。
- **C**：同几何、同顺序 selected set 的映射缓存；重新计算 acquisition/score/selection/model。第一次准备和复用成本分开。
- 每个正式计时条件有 **20 warmups + 100 formal repeats**，直接测 request wall time，并在 GPU 边界同步；阶段 profiling 不相加替代直接总时延。Full 没有虚构的 sampling seed，old/new Full 视图共享一次物理测量。
- rAUEC 每个 sampling seed 单独积分三预算曲线，再除以同 checkpoint/backbone/population/reconstruction 的 Uniform 面积。Uniform=1，Full100% 不进入积分；跨 backbone 的 rAUEC 不是绝对精度排名。
- Standard 和 E5 的误差棒是 **3 个 training runs 的 sample SD**；本轮新 Airfoil/Darcy/Car seed0 对照的误差棒是 **sampling SD**；计时100次的波动另列。
- 显存覆盖 PyTorch allocator 峰值及记录的 NVML 采样，不包含所有 CPU 分配或所有瞬时外部 GPU 分配。缓存 first+reuse 合成值按其定义作为派生估计保留。

## 仍在补齐的内容

队列快照：`2026-11-15T13:12:15.943169+00:00`，状态 `RUNNING_MISSING_STAGE`。以下状态按该快照记录，完成执行不自动替代独立结果验收：

| 队列任务 | 状态 | 含义 |
|---|---|---|
| shapenet_car_t3_trainingseed1_cpg_beta_pairs | COMPLETED | 复用已有 checkpoint 的 CPG beta 配对补评估；不新增模型计数 |
| shapenet_car_t3_trainingseed2_cpg_beta_pairs | COMPLETED | 复用已有 checkpoint 的 CPG beta 配对补评估；不新增模型计数 |
| airfoil_t3_trainingseed1_backbone | RUNNING | 新独立 training seed 的 backbone 训练；完成且独立验收后才进入模型清单 |

Car LinearNO seed0 的 PI/CA 及对应 head 已完成并单列于新补充；独立 training seeds1/2 未纳入本次完成范围；这份历史快照未包含新 matched Airfoil/Darcy 的 point backbones training seeds1/2；Airfoil 的后续完成与 Cd/Cl 修正见上方。本次19个模型清单只登记已有完成且审计的 checkpoint。原生 conv-temp 的固定网格 Full-only 限制单列，不通过修改架构伪装为变量点数模型。

已恢复问题也保留来源：Elasticity seed0 head 的启动锁失败后，三份 head 均完成一致的 CPU FP32 训练；Airfoil 原 finalizer 在全部测量后发生重复字典键错误，本轮从原始完整记录恢复报告；Darcy 旧总监控 flag 未读取实际 gate，本轮依据真实完成键和完整矩阵验收。原失败/待验收字符串未被覆盖。

来源与复现：[standard](data/rebuttal/2026-11/current_results_20261115/standard/README.md)、[Car/Airfoil 重建与时延](data/rebuttal/2026-11/current_results_20261115/reconstruction_airfoil/README.md)、[Darcy 900/100](data/rebuttal/2026-11/current_results_20261115/darcy/README.md)；去重与源 registry hash 见 [inventory_validation.json](data/rebuttal/2026-11/current_results_20261115/inventory_validation.json)。

## 固定基线中已发布的历史测试

以下是提交91c1fa7保留的历史结果入口，按各自README声明的完成范围阅读；不与本轮19模型统计相加，也不把旧snapshot的pending项目自动提升为完成。

| 历史包 | 已发布的测试内容 |
|---|---|
| [E0](data/rebuttal/2026-11/E0/README.md) | 同checkpoint Full参照、绝对误差与相对Full误差 |
| [E1](data/rebuttal/2026-11/E1/README.md) | 历史完整输出计时；三个case、三sampling seeds、20warmups/100repeats |
| [E2](data/rebuttal/2026-11/E2/README.md) | FPS/空间分层等几何采样基线及同点集测度补偿配对 |
| [E3](data/rebuttal/2026-11/E3/README.md) | 原ShapeNet-Car T3/官方LinearNO精度与完整输出计时对照 |
| [E4](data/rebuttal/2026-11/E4/README.md) | Darcy85²/141²/211²观测分辨率迁移；2026-11-17更新派生统计、固定K混杂与区域误差分析；原27,000请求保留 |
| [E5](data/rebuttal/2026-11/E5/README.md) | ShapeNet-Car三独立训练种子与各自CA head |
| [E6](data/rebuttal/2026-11/E6/README.md) | 已发布的Airfoil多起点逆向设计/CFD及SuperWing已完成分项 |
| [E7](data/rebuttal/2026-11/E7/README.md) | 固定点集beta诊断和已知proposal的合成token实验 |
