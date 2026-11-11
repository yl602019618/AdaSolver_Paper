# Response: Elasticity coarse-sampling consistency

**Comment, paraphrased.** The CPG method description uses approximately 10%
coarse points, whereas the Elasticity timing table specifies 25% coarse
points and a two-block CA head. Please correct this inconsistency and use the
10% configuration throughout the comparison.

**Response.** Thank you for identifying this inconsistency. The previous
Elasticity latency entries used an earlier sampling configuration. We have
remeasured all Elasticity entries using the same frozen policies as the
accuracy evaluation: both CPG and CA use 97 of the 972 reference points for
coarse acquisition, corresponding to the 10% budget rounded down. The final
budgets contain 364, 486 and 729 points, including those 97 coarse points.

The revised timing experiment also uses the same 128-to-4-to-1 CA stress
head as the accuracy experiment. This head was trained on frozen first-block
features from coarse inputs containing at most 97 points. CPG uses a complete
forward pass of the frozen backbone on the 97-point input; it has no separate
training stage. The backbone's geometry-amortized pretraining range remains
389–972 points, as specified in the training table. We now state the coarse
input count explicitly in the Elasticity CPG and CA descriptions.

The new experiment uses three geometries, three sampling seeds and all three
final budgets, with matched Uniform, PI, CPG and CA measurements and three
full-mesh references. Each configuration is timed for 20 repetitions on one
NVIDIA H20 with batch size one. The corrected latencies are:

| Final point budget | CPG latency (ms) | CA latency (ms) |
| --- | ---: | ---: |
| 37.5% | 3.115 ± 0.015 | 2.392 ± 0.013 |
| 50% | 3.363 ± 0.019 | 2.692 ± 0.022 |
| 75% | 4.012 ± 0.046 | 3.306 ± 0.022 |

The standard deviations are across the nine geometry/seed mean latencies.
The full-mesh reference is 1.256 ± 0.001 ms across three geometries. CA reduces
latency relative to CPG by 23.2%, 20.0% and 17.6% at the three budgets. It
remains slower than full-mesh inference on Elasticity, and the revised
computational-cost discussion retains this result with the corrected values.

All 111 measured configurations reproduce their reference point sets and
full-field predictions exactly. Relative-L2 errors differ from the previously
saved accuracy evaluation by at most 1.104e-7. The original Voronoi weight
calculation, including nearest-neighbor ties, is retained and charged to the
timed request. The accuracy tables therefore continue to use their existing
10% results. We have replaced the latency rows, removed the obsolete 25%
configuration note, and updated the associated text and data.

**Manuscript changes:**

- Appendix C.2, page 22: specify 97 coarse points and the three final point counts for Elasticity CPG.
- Appendix C.3, page 24: specify the CA head's sparse training inputs and 97-point inference input.
- Appendix B.4, page 18: state that the original Voronoi assignment and its transfers are included in latency.
- Appendix D.4 and Table 7, page 31: replace the Elasticity latency entries and update the cost discussion.

Raw measurements, configurations and numerical checks are included in
`data/elasticity_cpg10_timing/`.
