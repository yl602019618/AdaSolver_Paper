# Completed training inventory

This index contains 9 completed backbones and 5 matching trained heads. It separates training completion from test-accuracy and complete-output latency acceptance.

| Dataset / protocol | Backbone | Training seeds | Backbone training | Matching heads | Accuracy / latency |
|---|---|---|---|---|---|
| ShapeNet-Car, E5 | Transolver3 | 0, 1, 2 | 400 epochs each; 142,000 updates | 3 completed | E5 accuracy verified; full follow-up timing pending |
| ShapeNet-Car, E3 | Official LinearNO | 0 | 400 epochs; 142,000 updates | Pending | Full/Uniform/CPG accuracy and original timing verified; PI/CA extension pending |
| Airfoil, matched 900/100 | Transolver3 | 0 | 1,000 epochs; 113,000 updates | Completed | Full accuracy matrix verified; formal latency pending |
| Airfoil, matched 900/100 | Official LinearNO point-temp | 0 | 1,000 epochs; 113,000 updates | Completed | Full accuracy matrix verified; formal latency pending |
| Darcy, historical 1000/200 alternative | Transolver3 | 0, 1 | 1,000 epochs each; 250,000 updates | Pending | Training verified only |
| Airfoil, native fixed-grid reference | Official LinearNO conv-temp | 0 | 1,000 epochs; 113,000 updates | Not applicable | Full-only training verified; evaluation acceptance pending |

The two Darcy backbones are the newly indexed training results. Their full 1,000-epoch histories, validation-selected checkpoints, persisted optimizer/scheduler steps, normalizers and model loading were independently checked. The validation errors are 0.0108434390 and 0.0108681973. Recorded training wall times are 10,744.79 s and 10,762.20 s.

The Darcy alternative uses 1,000 training cases, 200 validation cases, its recorded normalizer and normalized-target loss. The new LinearNO comparison uses a different 900/100 split and physical-plus-derivative loss. These two completed models cannot supply seeds for that matched comparison. Historical test visibility is retained; this is not a newly sealed holdout.

Browse [model_index.csv](model_index.csv), [model configurations](model_configurations.json), [training evidence](training_evidence.json) and [external source-file hashes](source_files.json). Previously published accuracy and figures remain in the [parent result package](../README.md).

Checkpoints, normalizers and original long logs remain in external storage. Resolve workspace-relative paths in source_files.json against the project working root to locate them. No weight payload is included in Git. Training source Git revisions and GPU busy time remain explicitly unrecorded where the original runner did not save them.

Incomplete runs and pending heads/evaluations are listed in [pending.json](pending.json). The live experiment and resource queues are preserved.

The convolutional LinearNO reference retains its native fixed-grid architecture and full-grid training schedule. It is Full-only and does not enter the variable-point rAUEC comparison.
