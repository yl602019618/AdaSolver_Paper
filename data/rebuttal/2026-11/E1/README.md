# E1: Complete-output latency: completed groups

The CSV contains only completed, accepted groups, with three fixed geometries, three sampling seeds, 20 warmups and 100 timing repetitions per configuration. Latency includes the requested complete field output, sampling, reconstruction and required transfers. Timing populations are smaller than the full accuracy-test populations. The overview figure shows the four tasks named in overview_figure_scope.json; the CSV may cover additional completed tasks. Repeat-level data explicitly mark warmups, which are excluded from formal latency statistics.

Implementations are explicit: native Elasticity and the exact accelerated implementation are separate groups. Field-target timing must not be assigned to a differently tuned Cd/CdCl policy. The two 18.75% Airfoil groups retain their distinct Field and Cd/Cl configurations. DrivAerML's complete formal matrix and isolated memory measurements are still in progress at this snapshot. Stage sums are not substituted for direct request time.

## Files

- [latency_summary.csv](latency_summary.csv)
- [E6_low_budget_metric_alignment.csv](E6_low_budget_metric_alignment.csv)
- [latency_repeats.csv](latency_repeats.csv)
- [stages.csv](stages.csv)
- [accuracy.csv](accuracy.csv)
- [overview_figure_scope.json](overview_figure_scope.json)
