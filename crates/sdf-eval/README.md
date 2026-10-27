# sdf-eval

Volumetric IoU, Chamfer Distance, and F-score for two aligned GLSL snippets
defining `float sdf(vec3 p)`. Shader evaluation uses Rust, wgpu 30, and Naga,
with Vulkan preferred on Linux; surface meshing and nearest-neighbor queries
run on the CPU.
It does not translate, rotate, or rescale either input. There is no shader
hash calculation or validation.

`sdf-eval topology` computes solid Betti numbers, their absolute differences,
and an exact Betti match flag from occupied voxel cubes. See
[the topology protocol](TOPOLOGY.md) for usage, connectivity, and validity rules.

## Build and run

From the repository root, in the development environment:

```sh
cargo build --locked -p sdf-eval
./target/debug/sdf-eval iou \
  --reference dataset/Sphere/sdf.glsl \
  --prediction predictions/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 \
  --resolution 256 \
  --output results/Sphere.json
```

`--bounds` is required: `xmin,ymin,zmin,xmax,ymax,zmax`. Negative bounds can
be supplied with the `--bounds=...` form shown above. `--resolution` defaults
to 256 cells per axis; non-cubic boxes are supported. `--batch-size` defaults
to 1,048,576 positions and is clamped to device limits. Without `--output`,
JSON goes to stdout. Diagnostics go to stderr. `WGPU_BACKEND` can override
the backend choice.

Input snippets must be self-contained, Naga-compatible GLSL. Do not supply
`#version`, `main`, resources, or identifiers beginning with `sdf_eval_`.
The tool supplies a GLSL 450 compute entry point and its resources. A snippet
that works with a fragment renderer may use features unavailable in compute.

## Surface metrics

Use `sdf-eval surface` to compute both Chamfer Distance and F-score from
independent area-weighted samples of extracted zero-isosurfaces:

```sh
./target/debug/sdf-eval surface \
  --reference dataset/Sphere/sdf.glsl \
  --prediction predictions/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 \
  --resolution 128 --samples 100000 --seed 42 \
  --thresholds 0.01,0.02,0.05 \
  --output results/Sphere-surface.json
```

Thresholds are required in input coordinate units. Unlike IoU, this mode reads
back scalar fields and uses CPU marching tetrahedra and KD-trees. There is no
surface projection. Unsigned fields may yield `no_surface_extracted`.
See [the surface protocol](SURFACE.md) for definitions, sampling, resource costs,
failure semantics, and convergence checks.

## Manifest batch evaluation

The single-comparison commands remain available. Use `batch` to evaluate a JSON
manifest with one GPU device shared across jobs:

```sh
./target/debug/sdf-eval batch --input jobs.json --output results/batch.jsonl
```

For a `jobs.json` at the repository root:

```json
{
  "schema_version": 1,
  "defaults": {
    "metrics": ["iou", "surface", "topology"],
    "resolution": 128,
    "batch_size": 1048576,
    "surface": {
      "samples": 100000,
      "seed": 42,
      "thresholds": [0.01, 0.02, 0.05]
    }
  },
  "jobs": [
    {
      "id": "sphere-attempt-1",
      "reference": "dataset/Sphere/sdf.glsl",
      "prediction": "predictions/attempt-1/Sphere.glsl",
      "bounds": [-2, -2, -2, 2, 2, 2]
    },
    {
      "id": "sphere-attempt-2",
      "reference": "dataset/Sphere/sdf.glsl",
      "prediction": "predictions/attempt-2/Sphere.glsl",
      "bounds": [-2, -2, -2, 2, 2, 2],
      "surface": {"thresholds": [0.02, 0.05]}
    }
  ]
}
```

`schema_version` must be 1 and `jobs` must be nonempty. Each job requires a
nonempty unique `id`, input paths, and explicit shared `bounds`; there is no
automatic fitting or alignment. Shader paths are relative to the manifest
directory, not the current working directory. Absolute paths are accepted.
Unknown fields, duplicate metrics, and invalid settings reject the entire
manifest before opening the output. Existing inputs must not be used as output.

Jobs can override `metrics`, `resolution`, `batch_size`, and `surface`. Surface
options inherit individually; a thresholds array replaces the default array
rather than extending it. Built-in defaults are `["iou"]`, resolution 256,
batch size 1,048,576, 100,000 surface samples, and seed 42. Surface jobs require
a nonempty array of positive finite thresholds in input coordinate units.
`batch_size` still means positions per GPU dispatch batch, not job count.

Execution is serial, grouped by actual reference source contents, bounds,
resolution, batch size, surface sample count, and seed. Identical source at
different paths shares a group. Threshold changes do not invalidate the cache.
Reference contents are snapshotted before evaluation. Within a group, IoU
reuses the compiled reference pipeline and GPU classification buffers; surface
metrics reuse reference points and their KD-tree; topology reuses reference
Betti numbers. IoU and topology use voxel centers, while surface meshing uses
lattice corners; these sampling conventions are not combined. The reduction
pipeline is shared across groups. Group caches are released before the next
group, with no disk cache or concurrent scheduling.

Caching trades memory for less repeated work. IoU reference buffers retain
roughly `4 * (N^3 + 6*N^2)` bytes per active group, in addition to working
buffers. Surface points and their index, and any retained topology result,
remain live until the group finishes. Input source strings are retained for
the run. With one prediction per reference the main benefit is shared GPU
initialization, not repeated-reference preprocessing.

Output is JSONL: one flushed record per job and metric, with the same result
fields and validity rules as the corresponding single command, plus `id`,
`metric`, `requested_batch_size`, and `reference_cache_hit`. Cache hits describe
reused preprocessing for that metric, not an already cached prediction score.
Join results by `id` and `metric`: grouping can reorder jobs; unreadable
references are reported after readable groups. Metrics retain their requested
order within each job. Without `--output`, JSONL goes to stdout; progress and
diagnostics go to stderr. Per-record elapsed time includes only work performed
for that record, not manifest parsing, grouping, input reads, or prior cache
construction. It is not a standalone per-comparison timing benchmark.

Input read, compile, and metric failures produce records and do not stop later
work. A caught panic drops potentially poisoned GPU state before continuing.
The process exits 0 only when every record has status `ok`, otherwise 1.
Manifest and output I/O failures are fatal and reported to stderr; output I/O
failure can leave a partial JSONL file. Empty unions and invalid topology
remain failures, not perfect matches. There is no per-job timeout; use an
external subprocess deadline for untrusted shaders or stalled drivers.

## IoU metric

For a resolution N, volume samples are taken at voxel centers:

```text
p = lower + (index + 0.5) * (upper - lower) / N
inside = sdf(p) < 0
IoU = intersection_count / union_count
```

The distance magnitude is never used. Zero is outside. Calculations use FP32;
intersection and union are exact integer counts for the computed classifications.
This is a sampled estimate of volume overlap within the supplied box, not a
continuous analytic volume calculation. Larger boxes at a fixed resolution
have coarser spacing. Input coordinates must already be aligned.

Additionally, N by N points are sampled at cell centers on each of the six box
faces. These samples contribute only to boundary diagnostics and non-finite
counts, never to the volume intersection or union. Occupied boundary samples
indicate possible truncation. Unoccupied samples do not prove that no geometry
exists outside the box.

Both functions must encode the intended interior with negative values.
Unsigned distance estimates, open surfaces, and thin structures can render
successfully while providing no reliable occupied volume for this metric.
The evaluator does not thicken surfaces or infer interiors.

## IoU GPU pipeline

Each input is compiled independently, avoiding collisions between helper
functions and global variables. Each batch executes:

1. Reference classification into a GPU `u32` buffer (outside, inside, invalid).
2. Prediction classification and intersection/union reduction within workgroups.
3. Reduction of workgroup totals into eight counters.

The CPU reads only eight `u32` counters per batch and accumulates them in
`u64`. No voxel grid is read back or exported. Each workgroup has 256 threads;
the default reference buffer uses about 4 MiB.

Naga 30 lowers GLSL `barrier()` to all barrier flags, unnecessarily requesting
subgroup capabilities. The compiler wrapper corrects these to workgroup
barriers before wgpu validation, matching GLSL compute shared-memory semantics.

## IoU results and failure behavior

JSON includes the schema/tool versions, input paths, bounds, resolution, voxel
size, effective batch size, adapter/driver information, counts, IoU, and total
elapsed wall time (including compilation and initialization).

| Status | IoU | Exit code |
|---|---|---|
| `ok` | Number in [0, 1] | 0 |
| `empty_union` | `null` | 1 |
| `non_finite_values` | `null` | 1 |
| `reference_compile_failed` / `prediction_compile_failed` | `null` | 1 |
| `invalid_settings`, input read, initialization, or GPU failure | `null` | 1 |
| `evaluation_panicked` | `null` | 1 |

`boundary_occupied` is a separate boolean. A successful result with this flag
still measures overlap within the box; enlarge the bounds when full-object
IoU is intended. Non-finite counts include volume and boundary samples.
CLI syntax errors exit through clap; output-file errors are reported to stderr.

There is no built-in process timeout. Batch callers should run the executable
in a subprocess with a deadline; a GPU driver stall or nonterminating shader
cannot be reliably interrupted by a normal in-process timer.

## Validation

```sh
cargo test --locked                         # Tests without a GPU
cargo test --locked -- --include-ignored    # Includes GPU integration tests
```

Manifest tests cover validation, relative paths, input/output alias protection,
and per-metric failure records without a GPU. The opt-in batch GPU test checks
all three metrics against single commands, content-based cache reuse, parameter
invalidation, threshold overrides, and recovery after an invalid prediction.

GPU tests require a compute adapter, including a software Vulkan adapter if
available. They cover identical/disjoint/nested spheres, analytic cube overlap,
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
