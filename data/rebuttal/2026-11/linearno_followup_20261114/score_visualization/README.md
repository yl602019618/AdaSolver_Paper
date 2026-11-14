# Score, density and actual-point evidence

This package covers one fixed 3D ShapeNet-Car case and one fixed 2D Airfoil case,
both with Transolver3 training seed0, budget0.5 and sampling seed0. Each includes
Uniform, PI, CPG and its own trained CA head. The independent CPU Car CA chain
is a separate record; it does not overwrite the archived GPU score.

Browse score_visualization_manifest.csv, score_fields_index.csv, score_summary.csv,
score_visualization_qa.json, and each dataset's README. Compact per-point NPZs
contain only scalar score/selection data and coordinates, with explicit field
mappings and hashes. Complete feature tensors, native CFD truth and original
prediction caches remain external. Full scalar CSVs support offline figure redraw.

Raw prior/signal, normalized score, designed target density, actual quota measure,
proposal probability, selected coverage density, compensation, physical prediction
and diagnostic error have separate meanings and columns. A target/proposal is not
a claim of independent point-inclusion probabilities for FPS or sampling without
replacement. Real selected points are reconstructed from their stored order, not
sampled again for the figures. Uniform has no model score. Car PI uses a hard-shell
two-stage prior, not a synthetic continuous model score.

The score computation does not consume test truth. Airfoil denies truth-file and
target access during capture. The Car loader possesses truth, but acquisition
does not read it; error computation is a later separate branch. Error fields are
diagnostics and are never fed into score generation or point selection.

All data-alignment and numerical source checks passed. The complete FZ2 acceptance
status remains PARTIAL: Airfoil CPG retains619/1127 coarse points, contrary to the
pinned paper's generic nested-coarse statement; its score timing is a CPU observer
measurement. Car CPU/GPU CA scores have154 elementwise tolerance violations,
preserved in the audit. A separately verified CPU CA chain has its own score,
selection, prediction and error. New pre-execution source-hash replay verifies
current Car reproducibility without inventing a missing old-run receipt.

Darcy85×85 legacy display fields are indexed in twod_reuse_audit.json. They were
historically selected and display-rescaled, so they do not complete current score
evidence. This release satisfies the available one-2D plus one-3D coverage stage;
the current matched Darcy score capture is pending alongside its frozen models.

The FZ2 evidence ledger deliberately does not state that all requested manuscript
revisions have been completed. See FZ2_score_evidence.md. Formal complete-request
timing remains in the original experiment queue; CPU observation costs are not
substituted into the GPU accuracy-latency tables.

The Car coarse trace adds an explicit frozen-model coarse pressure → raw gradient energy → normalized score view from3218 ordered coarse predictions. Its CSV-only renderer independently reproduces all32186 raw/normalized values exactly. The complete package has9 figure families:6 Airfoil and3 Car. All visualized backbones are Transolver3; no LinearNO score capture is claimed.
