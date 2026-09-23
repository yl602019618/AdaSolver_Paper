# Appendix figures

[Density visualization PDF](density_visualization.pdf) · [PNG](density_visualization.png) · [SVG](density_visualization.svg). This overview uses the same eight cases at a 37.5% budget and seed zero. It shows actual local selected/native point fractions on a shared scale; the refinement-feature rows in the case pages retain their original definitions.

One compact budget-accuracy figure and eight separate case pages. All 21 physical quantities are retained. Field examples use a 37.5% point budget and sampling seed zero. Curve values are the supplied means and sample standard deviations over three seeds, normalized by the matching full-mesh mean error.

[Budget curves PDF](budget_error_curves.pdf) · [PNG](budget_error_curves.png) · [SVG](budget_error_curves.svg)

| Case | Fields | PDF | PNG | SVG |
| --- | ---: | --- | --- | --- |
| Airfoil2D | 1 | [PDF](airfoil2d_fields.pdf) | [PNG](airfoil2d_fields.png) | [SVG](airfoil2d_fields.svg) |
| Pipe | 1 | [PDF](pipe_fields.pdf) | [PNG](pipe_fields.png) | [SVG](pipe_fields.svg) |
| Darcy | 1 | [PDF](darcy_fields.pdf) | [PNG](darcy_fields.png) | [SVG](darcy_fields.svg) |
| Elasticity | 1 | [PDF](elasticity_fields.pdf) | [PNG](elasticity_fields.png) | [SVG](elasticity_fields.svg) |
| Plasticity | 3 | [PDF](plasticity2d_fields.pdf) | [PNG](plasticity2d_fields.png) | [SVG](plasticity2d_fields.svg) |
| ShapeNet-Car | 5 | [PDF](shapenet_car_fields.pdf) | [PNG](shapenet_car_fields.png) | [SVG](shapenet_car_fields.svg) |
| DrivAerML | 5 | [PDF](driverml_fields.pdf) | [PNG](driverml_fields.png) | [SVG](driverml_fields.svg) |
| SuperWing | 4 | [PDF](superwing_fields.pdf) | [PNG](superwing_fields.png) | [SVG](superwing_fields.svg) |

The density-feature row shows normalized refinement scores. Component errors are signed residuals; magnitude errors are vector norms. Plasticity uses all 20 time steps and averages absolute component errors or vector norms before display. Shared colour ranges and clipping arrows are retained within each field.

The Transolver curve includes its measured 100% endpoint. Labels are shortened to Uniform, PI, CPG, CA, Transolver, and Full mesh; protocol details are in the appendix text. The 48 reference-model points are in `fullmesh_training_curve_points.csv`.
