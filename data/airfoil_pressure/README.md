# Airfoil pressure sampling results

The field comparison uses the same 112 positive-lift test geometries and sampling seeds 0, 1, and 2 at 37.5%, 50%, and 75% point budgets. Each seed's method error curve is integrated and divided by its paired Uniform integral; the reported rAUEC is the mean and sample standard deviation of those three ratios.

`test_records.csv` contains per-case field errors. `test_summary.json` contains the budget curves and per-seed rAUECs. `confirmation_records.csv` covers 48 additional geometries and seeds 6, 7, and 8. Configuration selection uses the separate screening and selection sets described in `protocol.json`.

`configs.json` records the four evaluated policies. CPG uses a squared normalized pressure-gradient weight with a 0.1 floor. CA uses the same gradient exponent with a 0.01 floor and retains its coarse points. Both use 10% coarse inputs, 16 gradient neighbors, and unit-exponent Voronoi compensation. The pressure head has architecture 128→8→32→1 and 1,353 parameters.

These records supply the Airfoil **field** column and pressure-budget curves. The separately selected aerodynamic-quantity and inverse-design configurations are documented in the manuscript.
