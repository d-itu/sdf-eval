# Surface metrics

The `surface` command evaluates Chamfer Distance and F-score together from the
same two sampled surfaces.

```sh
./target/debug/sdf-eval surface \
  --reference dataset/Sphere/sdf.glsl \
  --prediction predictions/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 \
  --resolution 128 \
  --samples 100000 \
  --seed 42 \
  --thresholds 0.01,0.02,0.05 \
  --output results/Sphere-surface.json
```

`--thresholds` is required and accepts positive finite distances.
Surface-specific defaults are 100,000 points per surface and seed 42.

## Extraction and sampling

1. Evaluate each shader independently on the GPU at `(N+1)^3` cell corners,
   including the bounding-box faces. This differs from IoU's voxel centers.
2. Read back the FP32 scalar field in bounded batches. On the CPU, split every
   cell into the same six tetrahedra around its body diagonal. Interpolate the
   zero crossing of edges whose endpoints have different inside classifications
   (`value < 0`). This marching-tetrahedra construction avoids ambiguous cube
   cases and produces more triangles than marching cubes.
3. Sample triangles proportionally to their areas, then sample each triangle
   uniformly using square-root barycentric coordinates. Mesh interpolation,
   area computations, sampled coordinates, and nearest-neighbor distances use
   FP64 after evaluating the lattice in FP32.
4. Build a CPU KD-tree for each cloud and query exact Euclidean nearest
   neighbors in both directions. Both metrics reuse those distances.

The temporary mesh and point clouds are not exported. Points lie on the
extracted mesh; there is no projection back onto the shader's zero set.
Reference sampling uses `ChaCha8Rng(seed)` and prediction sampling uses
`ChaCha8Rng(seed XOR 0x9e3779b97f4a7c15)`. Different streams avoid artificially
perfect scores from identical random samples of identical meshes. The streams,
meshing method, projection setting, point count, and seed are recorded in JSON.
Changing batch size does not change the lattice order or random streams.

GPU memory is bounded by the batch size. CPU memory holds one full lattice
at a time, its temporary triangles and area distribution, sampled points, and
KD-trees. A 256-cell lattice alone occupies about 65 MiB. CPU mesh storage
scales with surface complexity; increasing resolution has cubic field cost.

## Definitions

Let `P` be the prediction cloud and `R` the reference cloud:

```text
d_PR(p) = min over r in R of ||p-r||_2
d_RP(r) = min over p in P of ||r-p||_2

chamfer_l2_mean = (mean(d_PR) + mean(d_RP)) / 2
chamfer_l2_squared_mean = (mean(d_PR^2) + mean(d_RP^2)) / 2

precision(t) = fraction of P with d_PR <= t
recall(t) = fraction of R with d_RP <= t
f_score(t) = 2 * precision(t) * recall(t) / (precision(t) + recall(t))
```

F-score is zero when both precision and recall are zero. Threshold comparisons
are inclusive. The JSON `metrics` object contains both Chamfer variants, both
directional mean distances, three HD95 values, and `f_scores` entries with
threshold, precision, recall, and F-score. The directed HD95 values are the
95th percentiles of the directed Euclidean distances; `hd95_symmetric` is the
95th percentile after concatenating both directions. Percentiles use linear
interpolation at `0.95 * (count - 1)`. The squared variant has squared
coordinate units and is not the square of `chamfer_l2_mean`.

Independently sampled identical shapes normally have positive CD and can have F-score below
one at small thresholds. Increase point count to reduce this sampling floor.
Increase grid resolution independently to reduce geometric discretization error.
Neither refinement can recover a surface absent from the signed field.

## Failures and boundaries

Reports add mesh triangle counts and areas, lattice size, meshing and sampling
parameters, and surface metrics. Surface-specific failures include:

- `no_surface_extracted`: no nondegenerate triangles were found for one input.
  This can mean an empty field, geometry outside the box, a box entirely inside
  a solid, an unsigned surface, or features too thin for the grid. Both empty
  surfaces are still a failure, not a perfect score.
- `non_finite_values`: a sampled shader output was NaN or infinity. The command
  stops on the first affected batch rather than silently discarding cells.
- `invalid_settings`: no points or a nonpositive/nonfinite threshold.

For each mesh, `boundary_occupied` is true if any box-face lattice node has
value <= 0. The result also includes the OR of both flags. This is a truncation
diagnostic, not a guarantee of containment. It does not invalidate the score.
Boundary edges are not capped, so the metric uses only the extracted surface
portion inside the box, without introducing artificial closing surfaces.
A field completely negative throughout the box still has no extractable surface.

Unsigned shaders such as the dataset's Triangle do not provide sign-changing
zero crossings. The implementation deliberately does not add thickness or
change the isovalue. Even for a renderable distance estimator, its sampled zero
surface may differ from its ray-marched appearance.

## Validation

Tests cover KD-tree distances against brute force, exact metric definitions and
threshold inclusion, area-weighted sampling and centroid statistics, planar
mesh area without caps, sphere mesh convergence, analytic concentric spheres,
independent self-sampling, batch reproducibility, clipped boundaries, invalid
parameters, NaN/Inf, missing surfaces, and compilation failures.

Development-adapter checks with a unit sphere and concentric radius-0.5 sphere
used 10,000 points each. The analytic surface CD is 0.5:

| Cells per axis | Measured CD |
|---|---|
| 16 | 0.508021 |
| 32 | 0.502097 |
| 64 | 0.500749 |

For identical unit spheres at resolution 32, increasing samples per surface
from 1,000 to 4,000 to 16,000 reduced CD from 0.056498 to 0.028240 to 0.014022.
At the defaults (256 cells, 100,000 points), self-CD was about 0.005600 and
F-score at 0.01 was about 0.9195. That complete run took about 14.3 seconds in
the default dev build on the development machine; timings vary by geometry
and hardware. These values illustrate finite-sampling effects, not shader error.

Sphere, Cube, Torus, and Jetfighter dataset smoke checks succeeded at resolution
32 with 2,000 points. Triangle returned `no_surface_extracted` as expected.
This does not establish surface-metric suitability for every dataset model.
