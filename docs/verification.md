# Verification and packaging

`verification/scripts/evaluate.py` combines volumetric IoU, surface F-score, and Betti
number similarity into a geometry score from 0 to 100. Optional paired renders
add SSIM and LPIPS reports separately from that score.

The task author owns metric implementation, score calculation, reference
materials, and the rubric. Verifier prepares an isolated environment, discovers
the single `verification/scripts/evaluate.py`, invokes it without interaction,
and copies its result files unchanged. It does not calculate or validate scores.
The entry point must perform the complete evaluation using the selected
reference materials and submission. It cannot rely on a task's `input/` directory.

## Local geometry scoring

Build `sdf-eval`:

```sh
cargo build --locked -p sdf-eval
python3 verification/scripts/evaluate.py \
  --reference dataset/Sphere \
  --output predictions/Sphere \
  --score results/Sphere/score.json \
  --evaluator target/debug/sdf-eval \
  --bounds=-2,-2,-2,2,2,2
```

`--reference` and `--output` must be directories containing `sdf.glsl`.
Both arguments are required single directory paths declared with `type=Path`;
files and multiple paths are rejected. Paths may be absolute or relative to the
current working directory. `--output` is an existing submission to read.
`--score` is a single JSON file path, defaulting to `score.json` in the current
working directory. It must be outside both input directories and cannot be a
directory or overwrite an evaluation resource. The default bounds are
`-2,-2,-2,2,2,2`; pin bounds and tool version for published tasks. Inputs must
already share the reference coordinate frame. The evaluator does not align
or normalize shaders. Each geometry subprocess has a 300-second timeout.

## Score output

The score file contains the allowed capability area
`Physical, Geometric, and Structural Capability`, its
`Geometric Form and Topological Structure` subcapability array, and the required
numeric top-level `total_score`. Every scoring item has exactly `name`, `score`,
and `weight`. Item scores range from 0 to 100; weights are 0.5 for volumetric
IoU, 0.3 for surface F-score, and 0.2 for Betti number similarity. The total is
the weighted sum, also from 0 to 100.

```json
{
  "Physical, Geometric, and Structural Capability": {
    "Geometric Form and Topological Structure": [
      {"name": "Volumetric IoU", "score": 100.0, "weight": 0.5},
      {"name": "Surface F-score", "score": 100.0, "weight": 0.3},
      {"name": "Betti number similarity", "score": 100.0, "weight": 0.2}
    ]
  },
  "total_score": 100.0
}
```

Use `--report results/Sphere/details.json` for an optional supplementary JSON
containing `status`, `settings`, raw per-item `value`, `earned_points`, and image
metrics when enabled. Its `total_score` agrees with the score file. The two
result paths must differ. The score file contains only the production scoring
structure; diagnostic fields are written to the supplementary report.

Missing or invalid submission shaders receive a scored zero result and exit 0.
Unavailable reference files, invalid configuration, or evaluation environment
failures exit nonzero without writing a new score. A prediction without an
extractable surface or occupied topology can retain points from other valid
metrics. Submission boundary occupancy invalidates the complete geometry score.

## Combined geometry and image evaluation

Install the locked Python environment with
`uv sync --project verification --locked`. Pass `--image-config` to enable
SSIM/LPIPS. The configuration is a JSON object with a required positive
`distance`, chosen using only the reference:

```json
{
  "distance": 4,
  "target": [0, 0, 0],
  "azimuths": 8,
  "elevations": [-30, 0, 30],
  "top": true,
  "bottom": true,
  "width": 512,
  "height": 512,
  "fov": 45,
  "antialiasing": 4
}
```

These are the supported keys. With only `distance` specified, the default is
26 views: 8 azimuths at elevations -30/0/30 plus top and bottom, with 4x
antialiasing. Both shaders use the same camera and render settings.

```sh
verification/.venv/bin/python verification/scripts/evaluate.py \
  --reference dataset/Sphere \
  --output predictions/Sphere \
  --score results/Sphere/score.json \
  --report results/Sphere/details.json \
  --evaluator target/debug/sdf-eval \
  --renderer "$(command -v sdf-view)" \
  --image-config results/Sphere/image-config.json \
  --artifacts results/Sphere/images \
  --weights results/.cache/torch
```

Save the configuration before running this command. If `--image-config` is
omitted, `image-config.json` in the selected reference directory enables image
evaluation automatically when present. Otherwise only geometry is evaluated.
The artifact directory
must not exist. `--weights` is a Torch cache containing the AlexNet checkpoint
under `checkpoints/`; prepare it before offline use. Image evaluation uses CPU.
The report's `image_metrics` field is separate from `total_score`. Invalid
submissions retain their invalid status and skip prediction image evaluation.
Rendering or environment failures exit nonzero without claiming a score.

## Production bundle

The `Build verification` GitHub Actions workflow
tests and packages a standalone evaluator for Ubuntu 24.04 x86_64 and CPython
3.11. To build on that platform:

```sh
python3 scripts/build_verification.py --reference dataset/Sphere
```

`--reference` is required and selects the task's reference shader. Only
`sdf.glsl` is copied from that directory. Pass `--image-config` with a fixed
reference-derived camera JSON to include it and enable image evaluation for
that task. The output directory defaults to `dist/verification` and must not exist.
`--output` can choose another new directory inside the repository. The builder
includes `sdf-eval`, the pinned `sdf-view` binary, Python entry points, the
scoring rubric, locked CPU Python wheels, and checked AlexNet weights. It runs
the combined scoring smoke check before publishing the directory. CI uploads
an uncompressed tar archive to preserve executable permissions.

The generated task materials have this layout:

```text
verification/
  reference/
    sdf.glsl
    image-config.json  # Present only when requested at build time
  rubric/
    scoring.md
    build.json
  scripts/
    evaluate.py
    runtime.py
    render_views.py
    evaluate_images.py
    requirements.lock
    bin/
    wheels/
    weights/
```

There is exactly one `evaluate.py`, under `scripts/`. The reference and scripts
directories contain only evaluation inputs and runtime resources. Development
tests, project manifests, preview images, and dataset metadata are excluded.
The rubric source is copied unchanged; build provenance is stored beside it.
CI packages the Sphere reference as an example; build with the actual task
reference before uploading materials for another model.

From a task directory containing `verification/` and the submitted `output/`:

```sh
python3 verification/scripts/evaluate.py \
  --reference verification/reference --output output --score score.json \
  --report details.json
```

Omitting `--score` writes `score.json` in that working directory.
The bundled binaries and weights are default paths relative to that entry
point. On image evaluation, the runtime bootstrap installs the bundled wheels
offline into a locked `.runtime/` environment using the host's CPython 3.11.
The host must provide the Vulkan runtime and a working compute adapter.

## Validation

```sh
verification/.venv/bin/python -m unittest discover -s verification
```

For an end-to-end check with real shaders, a compute adapter, the renderer,
and prepared AlexNet weights:

```sh
verification/.venv/bin/python scripts/smoke_verification.py \
  --entry verification/scripts/evaluate.py \
  --evaluator target/debug/sdf-eval \
  --renderer "$(command -v sdf-view)" \
  --weights results/.cache/torch
```

The unit tests cover the directory layout, CLI directory and result-path
validation, default score path, weighted capability schema, optional diagnostics,
and offline runtime bootstrap. The smoke check verifies identical, changed, and
invalid shaders through the combined geometry and image entry point and checks
that production scores match diagnostic totals.
