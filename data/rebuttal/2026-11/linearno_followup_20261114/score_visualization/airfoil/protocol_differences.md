# Airfoil score diagnostic scope

The score-source replay passes: all four native selected-index orders and compensation arrays match exactly. The paper-consistency status is PARTIAL.

At case1001, budget50%, sampling seed0, the frozen CPG configuration uses `sampler=weighted` and forces only the121 airfoil-surface points. It retains619 of1127 coarse points. The frozen CA configuration uses `sampler=weighted_nested` and retains1127 of1127 coarse points. The diagnostic preserves both original selectors; no coarse points were added after sampling.

Filled orange markers identify retained coarse points; orange crosses identify omitted coarse points. This CPG diagnostic cannot support the general statement that CPG and CA both retain all S0. It is not a manuscript revision or a claim that the mismatch has been repaired.

Raw native GPU score arrays were absent. The new score values are observed on CPU from the frozen checkpoint, own head and actual source functions. Exact native selected-set and compensation replay is verified; raw CPU/GPU score-value parity is not asserted.
