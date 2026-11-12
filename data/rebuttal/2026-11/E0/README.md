# E0: Same-checkpoint references

Eight benchmarks and twelve metrics use the same operator checkpoint within each sampling comparison. The table separates absolute error from error divided by that checkpoint's full-mesh error. Full is an observed reference, not a mathematical error bound. At 37.5% and 50%, all 32 Field CPG/CA conditions have higher mean error than the matching Full reference. Only two of all 48 low-budget metric conditions have slightly lower means, both DrivAerML drag CPG; this is not a significance claim.

Four cached endpoints for separately trained DrivAerML/SuperWing references lack their original raw files. They are not included as verified endpoint evidence in this snapshot. Asset paths are portable identifiers; the checkpoint and normalizer identities are retained.

## Files

- [same_checkpoint_reference.csv](same_checkpoint_reference.csv)
- [checkpoint_map.csv](checkpoint_map.csv)
- [model_asset_registry.csv](model_asset_registry.csv)
- [cpg_ca_low_budget_comparison.csv](cpg_ca_low_budget_comparison.csv)
