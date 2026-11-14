**Airfoil seed0 rAUEC and table.** Pressure rAUEC for one independently trained model per backbone on the published110 positive-lift cases. Mean and sample SD are across sampling seeds0/1/2. Each sampling seed integrates budgets37.5%,50%,75% and divides by its matched same-backbone Uniform area. CPG uses beta1. Full100% is excluded from the integral. The pressure task uses the official LinearNO point-temp variant with its corresponding head.

**Airfoil seed0 absolute error.** Pressure relative L2 on the221×51 evaluation grid. Curves show110-case means and sample SD across sampling seeds0/1/2. Dotted lines show same-checkpoint full-point errors. All curves use the same accepted linear interpolation and nearest fallback rule, with sparse GPU application and float64 complete outputs.

Both figures use training seed0 only. The historical positive-lift population is fixed; no never-touched-test claim is made. Full end-to-end latency and training seeds1/2 are separate pending measurements.
