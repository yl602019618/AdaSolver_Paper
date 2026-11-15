# P0 evidence scope

- Publication status: VERIFIED_COMPLETE; accepted rows and their identities are listed in verification.json.
- Seed0's existing point formal matrix is already complete and reused. Its beta0 extension is separately counted (324/324).
- Training completion alone is not a completed accuracy/latency result. Missing cells remain in pending.json.
- Across-training sampleSD requires the independent checkpoint count shown in each table; sampling runs are not extra training models.
- rAUEC normalizes against each checkpoint's matched Uniform curve and does not provide an absolute cross-backbone ranking.
- The dataset retains historical test visibility. Complete-output timing and fixed-map reuse remain distinct boundaries.

Analysis code commit: `d6847602ea8fed642b49c399b53ceaa38a9af936`.
