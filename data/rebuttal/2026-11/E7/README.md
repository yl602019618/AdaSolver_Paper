# E7: Measure-correction diagnostics: verified subset

This snapshot includes the completed production-beta versus beta=1 table from the fixed-point sweep and the known-proposal synthetic token experiment. rAUEC divides by matched Uniform area; relative-to-beta0 quantities have a different denominator. The sweep is not used to select a new production beta, and it does not establish how every historical hyperparameter was originally chosen.

The synthetic study uses 4,096 fixed reference points, six fixed feature functions and eight fixed slices. It draws categorical samples with replacement under a known positive proposal, with 1,000 repetitions per condition. All beta variants share the same draws. Bias is empirical mean-token discrepancy; variance uses the population convention across the Monte Carlo draws. The reference arrays and exact finite-reference token are supplied. This tests fixed-feature quadrature, not whole-network unbiasedness or beta=1 universal optimality.

The historical selection audit and additional real-case correction diagnostics are still being consolidated and are outside this verified subset.

## Files

- [beta_default_comparison.csv](beta_default_comparison.csv)
- [token_synthetic.csv](token_synthetic.csv)
- [synthetic_protocol.json](synthetic_protocol.json)
- [synthetic_reference.npz](synthetic_reference.npz)
