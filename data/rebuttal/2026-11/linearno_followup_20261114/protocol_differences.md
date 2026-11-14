# Reuse boundaries

| Evidence | Existing protocol | Requested reuse decision |
|---|---|---|
| E5 Car Transolver3 | Three training seeds, three matching heads; field CPG/CA beta1 | Reuse models, heads and verified field evidence; no retraining |
| E5 Car Cd | CPG/CA beta0.75 | Preserve recorded beta; evaluate missing CPG beta1 cells separately; CA need not adopt beta1 |
| E3 Car LinearNO | Official native model, seed0; Full/Uniform/CPG beta0/1 | Reuse seed0 and accepted rows; missing seeds1/2 and own CA heads remain explicit |
| Standard-seeds Darcy T3 |1000 train/200 validation; normalized target loss | Keep alternative; cannot pool with new900/100 physical-loss cohort |
| Current Darcy/Airfoil followup | Shared900/100 split, training seed0 only | Wait for seed0 owners; add isolated seeds1/2 with same protocol |
| Airfoil historical YAML | Older default budgets/population than released110 Cl>0.02 IDs | Published case IDs and221×51 grid take precedence |
| E1 Car timing | Checkpoint820e7fe8… | Cannot attach to E5 seed0 checkpointb12ae86b… |
| Reconstruction APIv1.1 |108 real fixed-prediction pairs and6 contract cases | Reuse implementation gate; retain per-new-checkpoint request validation |
| GPU sparse-application diagnostics |30 repeats; fixed map; map preparation separately3 repeats | Legacy diagnostic; not100-repeat complete request evidence |

The source directories are preserved. Differences do not invalidate their original
scopes, but they prevent substituting these results into a different protocol.
