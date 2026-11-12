# E2: Geometry sampling baselines

Darcy, Airfoil and ShapeNet-Car use the locked operators, populations and reconstruction rules. Geometric FPS and spatial stratification are each measured with and without their corresponding coverage/design correction, on identical point sets within each correction pair. All 14,760 requests and 7,380 pairs were verified.

Darcy FPS plus coverage has rAUEC 0.644516, close to CPG 0.641872 and CA 0.640025. These three sampling seeds do not establish a significant advantage over this geometry baseline. Airfoil corrected-FPS Cd/Cl is variable (1.137722 ± 1.421608). Zero coverage masses can occur for coincident reference representatives; these are nonnegative quadrature weights, not strictly positive inverse-inclusion probabilities. Single-request evaluator timings are distinct from E1 repeated latency.

## Files

- [baseline_summary.csv](baseline_summary.csv)
- [rauec.csv](rauec.csv)
- [metrics_records.csv](metrics_records.csv)
