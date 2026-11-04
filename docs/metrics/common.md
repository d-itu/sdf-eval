# Shared metric conventions

## Geometry inputs and execution

The `sdf-eval` commands evaluate self-contained GLSL snippets defining
`float sdf(vec3 p)`. Shader evaluation uses Rust, wgpu 30, and Naga, with Vulkan
preferred on Linux. `WGPU_BACKEND` can override the backend choice. CPU analysis
is used for surface meshing, nearest-neighbor queries, and topology.

Do not supply `#version`, `main`, resources, or identifiers beginning with
`sdf_eval_`. The evaluator supplies a GLSL 450 compute entry point and its
resources. Rendering compatibility alone does not establish compute compatibility.
Reference and prediction compile independently, avoiding helper-name collisions.

Shaders remain in their supplied coordinates. There is no translation,
rotation, scaling, normalization, camera fitting, or shader hash calculation
or validation. Reference-based metrics require aligned inputs; `self-check`
evaluates a single shader against its own zero set.

## Bounds and sampling

Geometry commands require explicit finite bounds in the form
`xmin,ymin,zmin,xmax,ymax,zmax`, with each upper bound greater than its lower
bound. Non-cubic boxes are supported. Use `--bounds=...` for negative values.
Distances and thresholds use shader coordinate units. Scores across differently
scaled objects need an explicit aggregation policy.

For IoU, surface, and topology, `--resolution` defaults to 256 cells per axis.
`--batch-size` defaults to 1,048,576 sample positions and is clamped to device
limits; it controls GPU dispatch size, not the number of comparison jobs.
Self-check uses separate probe and ray density settings.

With resolution `N`, voxel centers are
`lower + (index + 0.5) * (upper - lower) / N`; lattice corners are
`lower + index * (upper - lower) / N`. IoU and topology use voxel centers;
surface extraction uses `(N+1)^3` lattice corners. GPU field values use FP32.
The signed interior classification is strictly `sdf(p) < 0`; zero is outside.
Surface boundary diagnostics additionally flag zero-valued boundary nodes.

These are finite-sampling estimates. Increasing bounds at fixed resolution
coarsens spacing; thin structures, small features, unsigned fields, and
discontinuities can invalidate geometric interpretations. The evaluator does
not infer interiors, thicken surfaces, or change the zero isovalue. Compare
multiple resolutions when checking convergence. No sampled boundary test proves
that all geometry is contained in the box.

## Geometry reports and failures

Standalone commands write JSON to `--output`, or stdout when it is omitted;
output parent directories are created as needed. Diagnostics go to stderr.
Reports record tool/schema versions, inputs, effective settings, GPU adapter
information, status, and elapsed time, with metric-specific diagnostic fields.

Successful standalone geometry evaluation has status `ok` and exit code 0.
Input read errors, invalid settings, shader compilation failures, non-finite
samples, initialization failures, and GPU failures exit nonzero. Invalid
metrics remain null; empty or failed results never imply a perfect match.
Surface and topology reports retain a processing stage when available;
self-check may retain partial diagnostics. CLI syntax errors are handled by
clap, and output-file errors are reported to stderr.

Boundary validity depends on the metric: IoU and surface keep scores for the
portion inside the box, while topology rejects occupied outer voxel layers.
The combined reconstruction scorer applies a stricter submission policy.

The Rust evaluator has no built-in process or per-job timeout. Use an external
subprocess deadline for untrusted shaders or driver stalls.

## Paired image evaluation

Choose cameras from the reference alone and reuse them for every prediction.
Both inputs must use the same renderer version/backend, image names, camera
parameters, and shared render settings. There is no resizing, cropping, image
alignment, or camera fitting during comparison. Invalid or mismatched views
fail the entire evaluation without publishing a new report.

Per-model image scores are unweighted means over views. Dataset aggregation
uses equal model weights. Image similarity does not measure hidden internal
geometry. SSIM and LPIPS are reported separately from the geometry score.
