# Visual contract

The figures answer the observation-density reviewer question using three models trained at their own 85²/141²/211² resolution. Numerical conclusions wait for accepted final checkpoints and the complete evaluation matrix.

| Artifact | Evidence / sources | Display |
|---|---|---|
| A: rAUEC resolution | `results/rauec_summary.csv`; matched three models | Common-211 and native panels; mean ± sampling SD; own-Uniform denominator |
| B: Absolute error–budget | `results/absolute_errors.csv` | Common/native rows, three model columns, same y range within each metric row, own Full reference |
| C: Regional retention | `results/regional_summary.csv` | Boundary/interface/smooth columns; relative actual retained rate versus budget; resolution rows |
| D: Actual sampling | Fixed physical cases 1000/1099/1199, 50%, sampling seed 0; actual saved predictions/indices and planned patch counts | Three resolutions, five columns: physical regions, score, target density, observed bin retention, retained nodes; one plate per method/case |

The historical paper 85² appears only in the labelled author-facing table, not in the controlled resolution trend. Mean ± SD means three sampling seeds for one training seed. Color limits in D are shared across resolutions and methods for each physical case, with no percentile clipping. Score display normalization, PI nominal density, pre-FPS patch allocation and actual selection are explicitly distinguished. Raw support scores remain in the source CSVs.

All quantitative plots use Matplotlib from actual CSV data. Method colors use gray Uniform, orange PI, blue CPG and green CA, with distinct markers and line styles. Scalar maps use viridis. Score maps show `log(1 + score / support mean)` to preserve visible variation across large raw ranges, with shared limits and no clipping; untransformed scores remain in CSV. This is a display transform only. Outputs are vector PDF, editable-text SVG and 300 dpi PNG. Captions and source hashes are generated automatically. Render QA will inspect actual final PNGs; no fixture chart is labelled as an experiment result.
