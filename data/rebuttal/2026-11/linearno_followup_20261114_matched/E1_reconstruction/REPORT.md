# Reconstruction follow-up: execution in progress

Shared reconstruction API v1.1 is accepted after6 numerical contract cases and108 fixed-prediction real GPU comparisons. Maximum real-case field difference is5.684341886080802e-14. The gate preserves native Car k4 Euclidean IDW and native standard-grid linear plus nearest fallback rules.

Twelve real CPU validation fixtures and1,100 CPU stage observations are now saved. The staged native reconstruction exactly reproduces the native output (maximum difference0). These are diagnostic records, not formal GPU latency or new-cohort standard-task accuracy. Darcy and Airfoil still spend most CPU reconstruction work in QHull construction and simplex queries.

Sixteen further CPU Car A/C request comparisons passed after moving geometry-dependent sampler setup inside each request; all previously saved pilot metrics and selected-index hashes remain unchanged.

The persistent Car workflow is active: CPU preparation is complete, and GPU pilot is waiting for an idle GPU6. External training occupies the card. The remaining matrix is7400 paired old/new Car predictions,330 formal conditions per A/B/C scope with20 warmups and100 repeats, plus isolated memory. Missing final results are not replaced with old timing or synthetic numbers.

See runtime/car_workflow.json and read.me for current execution, config/car_protocol.json for the locked pre-execution boundary amendment, and results/car_matrix_coverage.json for actual/expected counts. Accepted API source and gate receipts are recorded in accepted_protocols.json. The code patch is implementation.patch. Standard-task owners use the same API and perform their own new-checkpoint validation and final accuracy.

This local package makes no remote push, main-branch update, manuscript overwrite or Feishu change.

## Candidate and recovery updates

The separate exact GPU-kNN candidate is now version0.2-input-guards. It passes the unchanged12-case independent CPU oracle suite,29 pre-search invalid-input rejections,2 additional valid-boundary checks, and one real Car velocity/surface-pressure fixture. The real fixed-field maximum difference is7.105427357601002e-15. These remain CPU checks; the candidate has not been selected for the accepted GPU protocol. Original candidate/audit sources are preserved. See results/gpu_knn_candidate_guard_audit.md and results/gpu_knn_candidate_real_car_cpu.json.

Worker recovery now preserves independent attempt receipts, prior GPU telemetry and partial observations on Python exceptions. Completed-source resume and duplicate-child rejection were checked with synthetic CPU fault injection. The running controller, CarRequest, timing_helper and frozen API are unchanged. See results/worker_resume_audit.md.
