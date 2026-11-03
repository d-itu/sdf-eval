# Verification bundle

`evaluate.py` scores aligned reference and prediction GLSL shaders using
volumetric IoU, surface F-score, and Betti number similarity. Optional multi-view
SSIM and LPIPS are reported separately and do not affect the geometry score.
See [scoring.md](scoring.md) for the scoring policy and invalid-result rules.

## Local development

From the repository root, enter the Nix development shell and synchronize the
shared Python 3.11 environment with uv:

```sh
nix develop
uv sync --project verification --locked
cargo build --locked -p sdf-eval
verification/.venv/bin/python -m unittest discover -s verification
```

Python 3.11 must already be available. The Nix shell disables automatic Python
downloads and looks for uv-managed interpreters in `.work/python`. The same
environment runs `scripts/evaluate_images.py`. CPU PyTorch dependencies are
locked in `uv.lock`; `requirements.lock` supplies hashes for bundle installation.
Regenerate both lock files when changing dependencies:

```sh
uv lock --project verification
uv export --project verification --locked --no-dev --no-emit-project \
  --emit-index-url --format requirements-txt \
  --output-file verification/requirements.lock
```

For local geometry evaluation, supply the development binary explicitly:

```sh
verification/.venv/bin/python verification/evaluate.py \
  --gt dataset/Sphere/sdf.glsl --gen predictions/Sphere/sdf.glsl \
  --evaluator target/debug/sdf-eval --bounds=-2,-2,-2,2,2,2 \
  --json results/score.json --md results/report.md
```

Bounds above are examples. Choose bounds from the reference, keep the complete
reference strictly inside them, and reuse those bounds for every submission.
Neither shader is automatically aligned, rescaled, or repaired. References
must have valid signed interiors and extractable zero surfaces.

## CI artifact and production runtime

The `Build verification` workflow builds on Ubuntu 24.04 x86_64, runs Rust tests,
and produces a tar archive preserving executable permissions. The archive
contains the `verification/` directory with these files:

```text
evaluate.py, runtime.py, render_views.py, evaluate_images.py
README.md, scoring.md, pyproject.toml, uv.lock, requirements.lock
bin/sdf-eval, bin/sdf-view
wheels/
weights/checkpoints/alexnet-owt-7be5be79.pth
manifest.json
```

The builder is `python3 scripts/build_verification.py` and requires Ubuntu
24.04 x86_64, uv, Rust, and renderer build dependencies. It downloads locked
CPU Python wheels and pretrained weights, builds both Rust executables, and
tests the staged bundle with identical, changed, and invalid shaders before
publishing it to `dist/verification/`. The builder requires network access.

Upload the complete bundle and provide task-specific reference shaders
separately. The runtime host needs CPython 3.11 with `venv`/`ensurepip`, compatible
system libraries, a Vulkan loader, and a working Vulkan compute driver. These
system dependencies are not bundled. Mesa software Vulkan is supported; select
the appropriate ICD through `VK_DRIVER_FILES` when needed. No Rust compiler or
network access is needed to evaluate the complete bundle.

After extracting the archive, run from any working directory:

```sh
python3 verification/evaluate.py \
  --gt reference.glsl --gen prediction.glsl --bounds=-2,-2,-2,2,2,2 \
  --json score.json --md report.md
```

The entry point locates `bin/sdf-eval` relative to itself. Override it with
`--evaluator` if necessary. Check executable permissions and test on the actual
runtime host before publishing a task.

## Optional image metrics

Add `--image-config camera.json --artifacts rendered-views` to enable SSIM and
LPIPS. The artifact directory must not exist. Choose cameras using only the
reference and reuse the configuration for all predictions. For example:

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

`distance` is required. The default orbit has eight azimuths, three elevations,
and top/bottom views, with 4x antialiasing. Image rendering uses the bundled
`bin/sdf-view`; `--renderer` overrides it. `--weights` overrides the bundled
`weights/` cache. Local development must prepare AlexNet weights in that cache
before offline use.

When image metrics are requested in a bundle, `runtime.py` creates a writable
`.runtime/<requirements-hash>/` environment beside the entry point using only
the bundled wheels, verifies package hashes, and reuses the environment on
later runs. `verification/.venv` is only the local development environment and
is not shipped. Geometry-only evaluation does not install Python packages.

Reports include `image_metrics` alongside the unchanged geometry score.
Rendering or dependency failures are evaluation errors, not numeric scores.
