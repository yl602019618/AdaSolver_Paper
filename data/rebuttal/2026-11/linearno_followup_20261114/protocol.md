# Three-training-seed follow-up protocol

This extension uses paper release `91c1fa7cc81b8c1cb61a43618296c1c892d818c0`
and its Elasticity ancestor `487fab2d87642c735ddacfe6d6ce13de7ec4999c`.
The first-stage evidence audit is `reuse_audit.json`. The existing seed0
workflows retain ownership of their checkpoints, raw records and timing jobs.

## Cohorts and selection

ShapeNet-Car retains the E5 split (710 train, 79 validation, 100 test), frozen
normalizer, variable-point schedule, 400 epochs, global batch2 with DDP2,
142000 updates and validation-selected checkpoints. All three E5 Transolver3
models and their heads are reused. E3 LinearNO seed0 is reused. LinearNO seeds1
and2 use the same E3 official model, schedule, optimizer and selection criterion.
Training seed is separate from sampling seed. New heads must consume their own
backbone's features and select only on the validation population.

Darcy85×85 and Airfoil221×51 extend the running followup's shared900/100 split
to training seeds1 and2. Seed0 remains owned by the existing workflows.
Normalizer bytes and case IDs are reused, and both point-model backbones retain
the respective seed0 point-count schedules, loss, optimizer, precision and budget.
Darcy uses1000epochs/225000updates/batch4; Airfoil uses1000epochs/113000updates/
batch8. The official fixed-grid convolution variant remains a separate Full-only
reference and is outside the requested variable-point comparison.

Historical test visibility remains disclosed. Darcy's other1000/200 training
cohort uses a different loss and normalizer and cannot supply the missing seeds
for this900/100 comparison. Airfoil evaluates pressure channel3 on the published
110 positive-lift IDs (reference Cl>0.02), retaining the original221×51 evaluation
grid. Test errors never select epochs, heads, beta, or reconstruction.

Exact executable locks, source snapshots, seeds and environment receipts accompany
each isolated runner. Training cannot start until the source, split and normalizer
checks pass. A failed compatibility check is recorded without substituting a
different architecture. GPU device admission is recorded at launch, after the
existing owners have finished their reservations.

## Inference and statistics

Each frozen training model uses Full100% and Uniform, PI, CPG(beta1), CA at
budgets0.375/0.50/0.75 and sampling seeds0/1/2. CPG beta0/1 is a separate paired
comparison with identical coarse predictions and selected indices. E5 Cd CPG beta0.75 is retained as legacy evidence and cannot fill the requested
CPG beta1 cell. E5 CA retains its recorded frozen beta, including0.75 for Cd;
the user did not require every CA configuration to use beta1.

For each training model and sampling seed, average errors over the fixed case
population at each budget, integrate the three-point curve using the trapezoid
rule, then divide by the matched Uniform area for the same checkpoint,
population, metric and reconstruction. Full lies outside this integral.
Average these three ratios within each model, then report the mean and sample
SD(ddof1) across the three independent training models. Sampling variability,
training variability and timing-repeat variability use separate columns.
Absolute errors provide comparisons across backbone models; their independently
normalized rAUEC values do not define an absolute accuracy ranking.

## Reconstruction and timing

Reuse the accepted reconstruction API v1.1 where its source hash agrees with
the accepted receipt. Its fixed-prediction parity gate is already verified;
each new checkpoint still requires end-to-end validation. Preserve original
coordinates, k, weights, epsilon, ties, duplicates, boundaries, channels and
float64 output containers. Use atol1e-6 and rtol1e-5 elementwise, reporting
maximum absolute error, relative norm and downstream metric differences.

Main timing is CPU input to complete CPU field, batch1 and FP32 network.
GPU-resident to complete GPU field and geometry/point-set map reuse are separate
boundaries. At least3 fixed cases,3 sampling seeds,20 warmups and100 direct wall
time repetitions are required per relevant condition; Full has no sampling seed.
Synchronize at request boundaries. Stage profiling does not replace direct total
time. Measure preparation and cache application separately and bind every timing
record to checkpoint, point set, reconstruction, hardware and scope.

Existing E1 Car uses a different checkpoint from E5. Existing756-condition GPU
map-application diagnostics use30 repetitions and cannot be presented as the
new100-repeat complete-output request matrix. No old error is paired with a new
time without the required parity and identity checks.

## Release

Publish verified summary data and reproducibility material on
`rebuttal/20261114-linearno-seeds`. Record partial and queued work explicitly.
Raw weights, long logs, private conversation text and local path inventories stay
internal. No main update, force push or destructive workspace operation is used.
