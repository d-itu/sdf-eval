# Verification and packaging

`verification/evaluate.py` combines volumetric IoU, surface F-score, and Betti
number similarity into a geometry score from 0 to 100. Optional paired renders
add SSIM and LPIPS reports separately from that score.

## Local geometry scoring

Build `sdf-eval`:

```sh
cargo build --locked -p sdf-eval
python3 verification/evaluate.py \
  --reference dataset/Sphere \
  --output predictions/Sphere \
  --score results/Sphere/score.json \
  --evaluator target/debug/sdf-eval \
  --bounds=-2,-2,-2,2,2,2
```

`--reference` and `--output` must be directories containing `sdf.glsl`.
`--score` is the required JSON output path. The default bounds are
`-2,-2,-2,2,2,2`; pin bounds and tool version for published tasks. Inputs must
already share the reference coordinate frame. The evaluator does not align
or normalize shaders. Each geometry subprocess has a 300-second timeout.

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
verification/.venv/bin/python verification/evaluate.py \
  --reference dataset/Sphere \
  --output predictions/Sphere \
  --score results/Sphere/score.json \
  --evaluator target/debug/sdf-eval \
  --renderer "$(command -v sdf-view)" \
  --image-config results/Sphere/image-config.json \
  --artifacts results/Sphere/images \
  --weights results/.cache/torch
```

Save the configuration before running this command. The artifact directory
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
python3 scripts/build_verification.py
```

The output directory defaults to `dist/verification` and must not exist.
`--output` can choose another new directory inside the repository. The builder
includes `sdf-eval`, the pinned `sdf-view` binary, Python entry points, the
scoring rubric, locked CPU Python wheels, and checked AlexNet weights. It runs
the combined scoring smoke check before publishing the directory. CI uploads
an uncompressed tar archive to preserve executable permissions.

Run the bundle's `evaluate.py` with the same directory inputs and `--score`.
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
  --entry verification/evaluate.py \
  --evaluator target/debug/sdf-eval \
  --renderer "$(command -v sdf-view)" \
  --weights results/.cache/torch
```

The smoke check verifies identical, changed, and invalid shaders through the
combined geometry and image entry point.
