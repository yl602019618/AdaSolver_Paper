# CPG coarse prediction and score trace

This companion exposes the actual CPU coarse-forward predictions already stored in the original capture. No model forward, GPU task or sampling experiment was repeated. The companion cpg_coarse_trace.csv contains the original3,218 ordered coarse points, native xyz and surface mask, and four generated physical prediction channels. These channels are predictions, not CFD truth.

The figure shows the355 coarse surface-pressure predictions used by acquisition, followed by raw gradient-energy score and q95-normalized score from the same CPU computation. Both score panels read the existing data/cpg_perpoint.csv fields raw_score_cpu and normalized_score_cpu. No GPU raw score is claimed or substituted.

Before plotting, the script independently recreates the native k4 pressure-difference gradient energy, nearest transfer to the surface, exponential surface-distance envelope, and q95/floor normalization from the exported predictions. Both full-cloud score arrays must match their existing CSV values exactly. Coordinates and coarse ordering are checked as well.

Run from this directory:

    python plot.py --coarse-csv ../cpg_coarse_trace.csv --perpoint-csv ../data/cpg_perpoint.csv --output-dir .

The three-dimensional views use the same fixed native coordinates and orthographic camera as the companion Car figures. The raw/normalized score panels show the original surface and the same central volume slab. Hashes and verification are in source_manifest.json and validation.json.
