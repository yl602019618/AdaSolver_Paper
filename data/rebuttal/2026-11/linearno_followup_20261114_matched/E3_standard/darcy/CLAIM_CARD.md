# Darcy follow-up claim card

| Claim | Current status | Evidence or missing work |
| --- | --- | --- |
| Both selected point models support the requested budgets | Supported at implementation level | Actual official/native class interface checks at 7,225, 2,709, 3,612 and 5,419 points. |
| Official convolution LinearNO supports arbitrary retained sets | Unsupported | Native reshape fails for 2,709 points; keep its valid Full-only reference. |
| LinearNO correction is applied at its point aggregation | Supported at implementation level | Key-logit `log(c)` addition after temperature division; unit and fixed-input multiplicity checks. See `LINEARNO_ADAPTER.md`. |
| New Darcy comparison has matched training cohorts | Plan locked; execution pending | Shared 900/100 IDs, normalizer and training schedule; new checkpoints not yet trained. |
| New checkpoint reconstruction preserves accepted field/metric tolerances | Pending | Geometry contract and initialized-class integration passed; trained validation/test checks have not run. |
| β=1 improves on β=0 | Not yet supported | Requires frozen-checkpoint paired coarse prediction / selected indices and final errors. Theoretical implementation identity alone does not establish an optimal β. |
| LinearNO is faster or slower than Transolver3 on this cohort | Not yet supported | Formal same-GPU Full and sampled CPU→CPU timing is pending. Point and convolution variants remain separate. |
| Reconstruction optimization speeds up complete inference | Not yet supported | Pending separate reconstruction and direct request measurements. Any adverse result will remain in the matrix. |
| The smooth2 test set is an untouched holdout | Unsupported | The historical population had already been viewed. New model selection uses only the declared internal validation set. |
| Full local experiment and publication package is complete | Pending | Training, trained-model accuracy, formal timing, independent rAUEC review and figures remain. Live queues and prepared implementation are not final results. |

No accuracy, rAUEC or speed value has been filled from an old table or from initialized-model preflight output. Missing scientific metrics remain N/A with the corresponding execution dependency.
