# E6: Inverse design: completed components

Airfoil static evaluation covers the locked 110-case positive-lift population at 18.75%, with 2,420 records. The static Field and Cd/Cl strategies differ and must not inherit the original fixed-sampling design's per-step time.

The multistart design has three preselected initial geometries and Full, unit-weight Uniform, same-point Uniform+coverage, and CA conditions. All 90 optimization calls, 30 selected endpoints and 33 independent CFD geometries passed verification; CFD residuals meet 1e-7. Three timing repetitions select the median-time representative, and their endpoints are identical. Full has no artificial sampling-seed variability. The low-camber initial geometry has CFD Cl≈0.03239, below the final Cl≥0.05 constraint; all 30 final designs satisfy that constraint.

Ordinary Uniform's mean CFD Cd/Cl exceeds corresponding Full by 3.36–5.00%; same-point coverage reduces the gap to 0.61–0.68%. CA means are lower by 0.16–0.25%, a small effect across these three starts. Design-only speed ratios exclude setup and independent CFD; setup-inclusive columns are separate.

All 72 SuperWing optimization trajectories and Full-surrogate audits are complete. These are surrogate results, not CFD trajectories. At 90 seconds, CA beats Uniform in only 2/6 case means and PI in 1/6. Equal-call views count candidate objective evaluations and use matched case/seed prefixes. The 78-geometry independent Wing CFD matrix remains in progress; it is not included as a completed CFD result set.

## Files

- [static_18p75_table.csv](static_18p75_table.csv)
- [static_18p75.csv](static_18p75.csv)
- [wing_surrogate_design_summary.csv](wing_surrogate_design_summary.csv)
- [wing_surrogate_equal_calls.csv](wing_surrogate_equal_calls.csv)
- [wing_method_summary.csv](wing_method_summary.csv)
- [wing_curve_data.csv](wing_curve_data.csv)
- [airfoil_curve_data.csv](airfoil_curve_data.csv)
- [airfoil_cfd_plot_data.csv](airfoil_cfd_plot_data.csv)
- [airfoil_design_summary.csv](airfoil_design_summary.csv)
- [airfoil_design_equal_calls.csv](airfoil_design_equal_calls.csv)
- [airfoil_design_equal_wall.csv](airfoil_design_equal_wall.csv)
- [airfoil_method_summary.csv](airfoil_method_summary.csv)
- [airfoil_cfd_endpoints.csv](airfoil_cfd_endpoints.csv)
