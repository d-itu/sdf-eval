# SDF Reconstruction Scoring Rubric

This rubric applies to one task with a ground-truth signed GLSL SDF and an aligned generated GLSL SDF. The verifier receives `--reference` and `--output` directories (each containing `sdf.glsl`) and writes JSON to `--score`; it cannot access the reconstruction images in `input/`. Scores measure geometry in the reference coordinate frame, not visual appearance or the agent's process. The four capability areas in the verifier guide are represented by observable geometry and executable-output evidence; input understanding and tool use cannot be scored independently from these files.

## Scoring

| Item | Metric and computation | Maximum |
| --- | --- | ---: |
| Volumetric IoU | `50 * intersection / union` over occupied voxel centers (`sdf(p) < 0`) in the shared box. | 50 |
| Surface F-score | `30 * F`, where `F = 2PR/(P+R)` (0 when both are 0), and P/R are the fractions of prediction/reference surface samples whose nearest opposite-surface sample is at most `0.02 L` away. `L` is the longest side of the box. | 30 |
| Betti number similarity | `20 * (1 - D / T)`, where `D = sum_i |beta_i(ref) - beta_i(pred)|` and `T = sum_i (beta_i(ref) + beta_i(pred))`, for `i = 0, 1, 2`. A valid reference has at least one occupied component, so `T > 0`. The score ranges from 0 to 20 and depends on relative, not absolute, Betti error. | 20 |

Total score is the sum of item scores, in `[0, 100]`, without rounding. IoU uses 256 cells per axis. Surface extraction uses 128 cells per axis, marching tetrahedra, 100,000 independent area-weighted samples on each surface, seed 42, and a single threshold of `0.02 L`. Topology uses 128 cells per axis, with 26-connected occupied voxels and a 6-connected complement. The topology item records the reference and prediction Betti vectors and their three absolute differences in `value`. Its score is unchanged if both Betti vectors are multiplied by the same factor; a perfect match gets 20, and disjoint nonzero Betti components get 0. Neither Betti similarity nor equal Betti numbers prove geometric similarity or homeomorphism. Scores reflect discrete sampling, not exact continuous geometry. Chamfer Distance, rendering quality, and human semantic judgments are not scored. The execution/output requirement is a hard validity condition rather than extra points: a self-contained GLSL shader defining `float sdf(vec3 p)` must compile and evaluate with Naga/wgpu.

## Invalid results and failures

- Missing generated file, prediction compile failure, non-finite prediction values, or occupied prediction boundary in any metric: invalid submission; all three items score 0 and the script writes a JSON report.
- A prediction with valid sampled volume but no extractable zero surface receives its IoU points and 0 surface points (`partial`). An empty prediction topology grid receives 0 topology points and retains other valid metric points (`partial`). An empty/unsigned prediction is never treated as a perfect match.
- Missing/invalid reference, empty or boundary-occupied reference (including its 128-cell topology grid), missing evaluator, GPU/driver failure, timeout, malformed evaluator output, or any unclassified evaluation error: evaluation failure, not a zero-scored submission. The script exits nonzero and does not claim a score. Correct the task materials or environment and rerun.
- Reports are JSON (top-level numeric `total_score`, per-item `value` and `earned_points`, `settings` and `status`). Success, including a scored invalid submission, exits 0 and writes the file specified by the required `--score` option.

The default `--bounds` value is `-2,-2,-2,2,2,2`. Bounds and tool version must be pinned for a published task. Compare results only within the same reference and settings. Review the JSON score after every scoring-policy change.
