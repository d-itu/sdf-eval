# Volumetric IoU

```sh
./target/debug/sdf-eval iou \
  --reference dataset/Sphere/sdf.glsl \
  --prediction predictions/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 \
  --resolution 256 \
  --output results/Sphere.json
```

## Definition

```text
IoU = intersection_count / union_count
```

Volume occupancy is sampled at voxel centers. Distance magnitude is unused;
intersection and union are exact integer counts of computed classifications.
The result estimates volume overlap within the supplied box.

Additionally, `N * N` points are sampled at cell centers on each of the six
box faces. These contribute only to boundary diagnostics and non-finite counts,
never to volume intersection or union. `boundary_occupied` is a separate
boolean and does not invalidate the score. Enlarge bounds when full-object
IoU is intended.

## GPU pipeline

Each batch executes:

1. Reference classification into a GPU `u32` buffer (outside, inside, invalid).
2. Prediction classification and intersection/union reduction within workgroups.
3. Reduction of workgroup totals into eight counters.

The CPU reads only eight `u32` counters per batch and accumulates them in
`u64`. No voxel grid is read back or exported. Each workgroup has 256 threads;
the default reference buffer uses about 4 MiB.

Naga 30 lowers GLSL `barrier()` to all barrier flags, unnecessarily requesting
subgroup capabilities. The compiler wrapper corrects these to workgroup
barriers before wgpu validation, matching GLSL compute shared-memory semantics.

## Results

JSON includes voxel size, effective batch size, occupancy and invalid-value
counts, and IoU. Non-finite counts include volume and boundary samples.

| Status | IoU |
| --- | --- |
| `ok` | Number in [0, 1] |
| `empty_union` | `null` |
| `non_finite_values` | `null` |
| `reference_compile_failed` / `prediction_compile_failed` | `null` |
| `invalid_settings`, input read, initialization, or GPU failure | `null` |
| `evaluation_panicked` | `null` |

## Validation evidence

GPU tests cover identical/disjoint/nested spheres, analytic cube overlap,
independent CPU counts in an asymmetric box, batch and partial-workgroup
boundaries, all six boundary faces, empty volume, NaN/Inf, and compile errors.

On the development Vulkan adapter, all 63 reference shaders compiled and ran
at resolution 32 within `[-2,2]^3`. 59 returned self-IoU 1; four had empty
unions. At resolution 256, Bezier and Mandelbulb had occupied samples, while
Serpinski and Triangle still returned `empty_union`. These are compatibility
checks, not an endorsement of a common box or of each shader's signed-volume
semantics. Mandelbulb's escape estimator in particular does not label its
entire visually rendered interior with negative values.

Nested unit/half-radius spheres have analytic IoU 0.125. Measured values at
128, 256, and 512 were approximately 0.125611, 0.124985, and 0.125131. Grid
estimates need not improve monotonically with resolution.
