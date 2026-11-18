# Figure index

All scientific result plots are generated from accepted CSV data. PDF and SVG retain vector text; PNG exports use 300 dpi. Color identifies methods and marker/line style provides a redundant encoding.

## A_rauec_resolution

Controlled resolution trend for three models trained from scratch on the same physical split. Left: common 211² output; right: native output. Each model/grid uses its own Uniform area. Error bars are sample SD across three sampling seeds for training seed 0. The historical paper 85² is excluded.

[PDF](A_rauec_resolution.pdf) · [SVG](A_rauec_resolution.svg) · [PNG](A_rauec_resolution.png)

## B_absolute_error_budget

Absolute full-field relative L2 on common (top) and native (bottom) output grids. Each row uses a common y axis across resolutions. The dashed black line is each checkpoint Full error; Full is evaluated once per case. Error bars show sampling SD across three seeds after averaging the same 200 cases.

[PDF](B_absolute_error_budget.pdf) · [SVG](B_absolute_error_budget.svg) · [PNG](B_absolute_error_budget.png)

## C_regional_retention

Actual regional retained fraction divided by the actual global retained fraction. Values above 1 indicate relative refinement; below 1 indicate lower sampling. Curves average physical-case regional rates before aggregating three seeds; error bars are sampling SD. Boundary and interface distances are evaluated directly at native coordinates with width 0.02 and the same common211 coefficient-edge geometry. This corrects nearest-boolean region classification without changing any retained point. These curves measure retained indices, not target densities.

[PDF](C_regional_retention.pdf) · [SVG](C_regional_retention.svg) · [PNG](C_regional_retention.png)

## D_sampling_case1000_pi

Fixed case 1000, PI, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the PI nominal proposal, not an exact inclusion probability. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1000_pi.pdf) · [SVG](D_sampling_case1000_pi.svg) · [PNG](D_sampling_case1000_pi.png)

## D_sampling_case1000_cpg

Fixed case 1000, CPG, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the pre-FPS final patch allocation divided by native available nodes and global budget. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1000_cpg.pdf) · [SVG](D_sampling_case1000_cpg.svg) · [PNG](D_sampling_case1000_cpg.png)

## D_sampling_case1000_ca

Fixed case 1000, CA, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the pre-FPS final patch allocation divided by native available nodes and global budget. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1000_ca.pdf) · [SVG](D_sampling_case1000_ca.svg) · [PNG](D_sampling_case1000_ca.png)

## D_sampling_case1099_pi

Fixed case 1099, PI, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the PI nominal proposal, not an exact inclusion probability. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1099_pi.pdf) · [SVG](D_sampling_case1099_pi.svg) · [PNG](D_sampling_case1099_pi.png)

## D_sampling_case1099_cpg

Fixed case 1099, CPG, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the pre-FPS final patch allocation divided by native available nodes and global budget. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1099_cpg.pdf) · [SVG](D_sampling_case1099_cpg.svg) · [PNG](D_sampling_case1099_cpg.png)

## D_sampling_case1099_ca

Fixed case 1099, CA, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the pre-FPS final patch allocation divided by native available nodes and global budget. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1099_ca.pdf) · [SVG](D_sampling_case1099_ca.svg) · [PNG](D_sampling_case1099_ca.png)

## D_sampling_case1199_pi

Fixed case 1199, PI, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the PI nominal proposal, not an exact inclusion probability. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1199_pi.pdf) · [SVG](D_sampling_case1199_pi.svg) · [PNG](D_sampling_case1199_pi.png)

## D_sampling_case1199_cpg

Fixed case 1199, CPG, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the pre-FPS final patch allocation divided by native available nodes and global budget. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1199_cpg.pdf) · [SVG](D_sampling_case1199_cpg.svg) · [PNG](D_sampling_case1199_cpg.png)

## D_sampling_case1199_ca

Fixed case 1199, CA, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is the pre-FPS final patch allocation divided by native available nodes and global budget. Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.

[PDF](D_sampling_case1199_ca.pdf) · [SVG](D_sampling_case1199_ca.svg) · [PNG](D_sampling_case1199_ca.png)

## E_training_head_curves

Full 1000-epoch training curves. Backbone panels show physical field train and validation relative L2; head panels use separate labelled axes for validation vector norm and training Huber loss. Checkpoint selection uses validation only. Exact original epoch fields are retained in the source CSV.

[PDF](E_training_head_curves.pdf) · [SVG](E_training_head_curves.svg) · [PNG](E_training_head_curves.png)
