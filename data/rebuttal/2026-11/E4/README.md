# E4: Darcy observation-resolution transfer

The same 200 physical samples are observed at 85², 141² and 211² using true 421² source data. Evaluation is on the common 211² grid; a separate comparison fixes 5,419 retained points. There are 27,000 successful requests and 135 groups of 200 cases.

The backbone and normalizer were trained at 85². Full error rises from 0.009845 to 0.014560 and 0.018342 as observation density increases. Adaptive gains relative to Uniform narrow; denser observations do not automatically improve this frozen model. Historical test-informed development is retained as a scope limitation. This frozen comparison preserves the original PI distance routine; its flattened-mask EDT differs from a physical two-dimensional interface distance. A separate correction must not be silently substituted into these results. CPG's original index-space derivative convention is also retained.

## Files

- [resolution_budget.csv](resolution_budget.csv)
- [fixed_points.csv](fixed_points.csv)
- [region_summary.csv](region_summary.csv)
- [metrics_records.csv](metrics_records.csv)
- [example_array_source_manifest.json](example_array_source_manifest.json)
