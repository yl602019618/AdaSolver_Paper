# Verified experiment snapshot — November 2026

This branch publishes derived experiment tables, compact per-case metrics, scientific figures and their scope descriptions. It is a staged evidence release; it does not represent a completed new manuscript, a submitted response, or a completed result for every ongoing experiment.

| Experiment | Contents | Snapshot status |
|---|---|---|
| [E0](../../data/rebuttal/2026-11/E0/README.md) | Same-checkpoint references | verified completed component |
| [E1](../../data/rebuttal/2026-11/E1/README.md) | Complete-output latency: completed groups | completed groups; full task ongoing |
| [E2](../../data/rebuttal/2026-11/E2/README.md) | Geometry sampling baselines | verified completed component |
| [E3](../../data/rebuttal/2026-11/E3/README.md) | Matched Transolver3 and LinearNO | verified completed component |
| [E4](../../data/rebuttal/2026-11/E4/README.md) | Darcy observation-resolution transfer | verified completed component |
| [E5](../../data/rebuttal/2026-11/E5/README.md) | Independent training seeds | verified completed component |
| [E6](../../data/rebuttal/2026-11/E6/README.md) | Inverse design: completed components | Airfoil CFD and Wing surrogate complete; Wing CFD ongoing |
| [E7](../../data/rebuttal/2026-11/E7/README.md) | Measure-correction diagnostics: verified subset | verified subset; historical selection audit ongoing |

All reported unfavorable outcomes are retained. Sampling seeds, training seeds, optimizer seeds and timing repetitions are distinct. Same-checkpoint full-mesh inference is separated from independently trained reference models. Full-mesh error is an observed reference, not an error bound. Read each experiment's scope before combining tables or interpreting latency.

## Download and provenance

- [Results and figures index](RESULTS_INDEX.md)
- [Release manifest](RELEASE_MANIFEST.json)
- [Validation summary](VALIDATION.json)

The manifest records the bytes published here and the source-artifact fingerprints used to prepare them. Host-specific filesystem locations are replaced by portable identifiers; numeric CSV fields are unchanged. Raw physical fields, checkpoints, internal review material and source code subject to separate release approval are outside this snapshot. This does not claim a complete executable reproduction package.

The inherited paper files remain at the branch's existing manuscript baseline. The new experiment snapshot is organized under `data/rebuttal/2026-11`, `figures/rebuttal/2026-11` and this index.
