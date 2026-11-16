# P0 evidence scope

<!-- original-cdcl-correction-20261117 -->
**Cd/Cl scope correction, 2026-11-17.** The Cd/Cl rows and panels in this P0 package are **field-policy force diagnostics**. They cannot be compared with the original Table 1 Cd/Cl metric-specific policies. The active Transolver3 three-training-seed Cd/Cl result is the [corrected original-policy package](../p0_airfoil_original_cdcl_20261117/README.md). Current Cd/Cl summaries and the reviewer response cite that package. The P0 raw numerical files, field accuracy, reconstruction parity and timing remain preserved under their original policies. The correction does not establish an original-policy Cd/Cl result for LinearNO.
<!-- /original-cdcl-correction-20261117 -->

- Publication status: VERIFIED_COMPLETE; accepted rows and their identities are listed in verification.json.
- Seed0's existing point formal matrix is already complete and reused. Its beta0 extension is separately counted (324/324).
- Training completion alone is not a completed accuracy/latency result. Missing cells remain in pending.json.
- Across-training sampleSD requires the independent checkpoint count shown in each table; sampling runs are not extra training models.
- rAUEC normalizes against each checkpoint's matched Uniform curve and does not provide an absolute cross-backbone ranking.
- The dataset retains historical test visibility. Complete-output timing and fixed-map reuse remain distinct boundaries.

Analysis code commit: `d6847602ea8fed642b49c399b53ceaa38a9af936`.
