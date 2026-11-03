# SDF reconstruction benchmark

Each directory under `dataset/` contains one model's final `sdf.glsl`,
`render.png`, and `metadata.json`. Upstream provenance and shader adaptations
are recorded in the metadata. Shader copyright and license notices are retained.

## Environment

Run commands from the repository root. `flake.lock` pins Nixpkgs and the
`sdf-view` source. The renderer's upstream `Cargo.lock` pins Rust dependencies.
The shell provides the pinned `sdf-view` command, Rust, Cargo, Python with
Pillow, and the Vulkan loader. Vulkan uses the pinned Nix Mesa ICDs instead
of the host's Mesa libraries; the default excludes the Lavapipe software ICD.
The host still supplies the GPU device and kernel driver.

```sh
nix develop path:.
sdf-view dataset/Sphere/sdf.glsl -o target/sphere-preview.png
# Explicit CPU fallback when no compatible GPU is available:
sdf-view-software dataset/Sphere/sdf.glsl -o target/sphere-preview-software.png
python3 scripts/render_dataset.py
python3 scripts/validate_dataset.py
```

`sdf-view-software` uses the pinned Lavapipe Vulkan driver and can be much
slower than hardware rendering. Neither command changes existing dataset
artifacts unless its output path points to them.

Pass model names to render selected samples, for example
`python3 scripts/render_dataset.py Snail Elephant`. Rendering uses the settings
stored in each model's metadata. `python3 scripts/fit_cameras.py` can refit
cameras from foreground bounds. Open `preview.html` to inspect all outputs.
Regenerate the gallery after rendering with `scripts/validate_dataset.py`.

## Multi-view rendering

For the combined geometry and image entry point, locked Python 3.11 development
environment, and Ubuntu 24.04 production upload builder, see
[verification/README.md](verification/README.md). Production bundles contain
both Rust tools, CPU Python wheels, and AlexNet weights; SSIM/LPIPS are reported
separately from the existing geometry score.

Render an arbitrary SDF into a new directory outside `dataset/`:

```sh
python3 scripts/render_views.py dataset/Sphere/sdf.glsl \
  --output results/Sphere/views --distance 2 \
  --azimuths 8 --elevations=-30,0,30 --top --bottom
```

`--distance` is required in shader coordinates; `--target` defaults to `0,0,0`.
The default orbit has 8 azimuths and elevations of -30, 0, and 30 degrees.
Y is up, and azimuth zero looks from +Z. The top and bottom views use -Z and
+Z as camera up respectively. Images are named by angle (for example,
`az045_el+30.png`) or `top.png` / `bottom.png`. `views.json` contains the image
names, camera parameters, and shared render settings. For software rendering,
pass `--renderer sdf-view-software`; `--width`, `--height`, `--fov`, and
`--timeout` are also configurable. The output directory must not exist; if any
view fails, the script prints the error and leaves no output directory.

## Image evaluation

`scripts/evaluate_images.py` compares paired multi-view renders with RGB SSIM
(higher is better) and AlexNet LPIPS v0.1 (lower is better). It shares the
locked Python 3.11 environment in `verification/.venv` with the verification
tools, including NumPy, Pillow, scikit-image, CPU PyTorch, torchvision, and
LPIPS. The Nix shell supplies the native runtime libraries; re-enter the shell
after updating it. From the repository root, with uv and Python 3.11 available:

```sh
uv sync --project verification --locked
verification/.venv/bin/python scripts/evaluate_images.py \
  --reference results/Sphere/reference_views/views.json \
  --prediction results/Sphere/prediction_views/views.json \
  --output results/Sphere/image_metrics.json
```

The output must be a new file. CPU is the default; pass `--device cuda` for
CUDA. Pretrained AlexNet weights download on first use to
`results/.cache/torch` (override with `--cache-dir`); prepare this cache before
offline evaluation. Dependencies and pretrained weights must be available for
scoring. Reports record package versions, metric parameters, cameras, shared
render settings, each view's scores, and the unweighted mean over all views.
For dataset aggregation, average the per-model means with equal model weights.
Do not assume LPIPS is bounded by 1 or convert it to a percentage.

### `views.json` contract

The evaluator accepts the manifest written by `render_views.py`:

```json
{
  "schema_version": 1,
  "shader": "/path/to/sdf.glsl",
  "renderer": "sdf-view",
  "distance": 3,
  "render": {
    "width": 512,
    "height": 512,
    "antialiasing": 4,
    "background": "transparent",
    "light_direction": [-0.5, 0.8, 1],
    "light_color": "white",
    "object_color": "rgb(137,196,237)",
    "light_intensity": 0.85,
    "ambient": 0.15
  },
  "views": [
    {
      "image": "az000_el+00.png",
      "azimuth": 0,
      "elevation": 0,
      "camera_position": [0, 0, 3],
      "camera_target": [0, 0, 0],
      "camera_up": [0, 1, 0],
      "fov": 45
    }
  ]
}
```

`schema_version` is 1; legacy manifests without this field are treated as v1.
All shown `render` fields are required. `views` must be nonempty, with unique
relative image paths contained within the manifest directory (no `..`). Each
view requires `image`, three finite 3D camera vectors, and `fov` in (0, 180).
`azimuth` and `elevation` are optional labels; `shader`, `renderer`, and
`distance` are provenance only and do not control matching. Use the same
renderer version/backend for both inputs; the manifest cannot verify the
renderer binary or the actual settings used to produce an image.

Views are paired by image name, independently of list order. Both manifests
must have exactly the same image names, shared render object, and per-view
camera parameters. Images must be RGBA PNGs matching the declared dimensions
(at least 64 pixels on each axis for AlexNet LPIPS). Missing, invalid, or mismatched inputs fail
the whole evaluation; views are never silently skipped and no new report is
published on failure.

RGBA is composited onto white using `rgb * alpha + (1 - alpha)` in normalized
sRGB values, without linearization. SSIM uses `data_range=1`, Gaussian weights,
sigma 1.5, and population covariance. LPIPS receives RGB tensors in [-1, 1].
There is no resizing, crop, alignment, or camera fitting during evaluation.
Choose cameras using only the reference and reuse them for every prediction.
For example, fix 8 azimuths, elevations -30/0/30, top and bottom, and 4x
antialiasing for 26 views. Image similarity does not measure hidden internal
geometry; retain the geometric metrics when that matters.

Run lightweight manifest tests with
`verification/.venv/bin/python -m unittest discover -s scripts -p 'test_*views.py'` and
`verification/.venv/bin/python -m unittest discover -s scripts -p 'test_evaluate_images.py'`.
To additionally test actual SSIM/LPIPS identity, transparency, and geometry
changes, run the latter command with the metric environment's Python and
`RUN_LPIPS_TESTS=1`; this may download pretrained weights.

The build uses the default Cargo dev profile and writes build outputs to
`target/` inside this repository. The shell preserves the user's `CARGO_HOME`
and `XDG_CACHE_HOME` settings. Nix manages its own store. `.envrc` supports
optional direnv use.

## GPU volumetric IoU

The evaluator also supports `sdf-eval topology` for Betti numbers and per-pair
Betti matching. See [the topology protocol](crates/sdf-eval/TOPOLOGY.md).

The root Cargo workspace contains `crates/sdf-eval`. In the development shell:

```sh
cargo build --locked -p sdf-eval
./target/debug/sdf-eval iou \
  --reference dataset/Sphere/sdf.glsl \
  --prediction predictions/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 \
  --resolution 256 \
  --output results/Sphere.json
```

Inputs must already be aligned. Bounds are required; the default resolution is
256 voxel centers per axis. The GPU classifies `sdf(p) < 0` and reduces counts,
returning JSON without exporting voxel data or calculating shader hashes.
Omit `--output` to print JSON to stdout. Empty unions and NaN/Inf produce null
IoU and a nonzero exit code. Boundary occupancy is reported separately because
the score applies only within the supplied box.

Run `cargo test --locked -- --include-ignored` for GPU integration tests.
See [the evaluator documentation](crates/sdf-eval/README.md) for sampling,
CLI options, reporting, and shader limitations. Renderable surfaces are not necessarily signed solids suitable for volume IoU.
Surface metrics operate on extracted zero-isosurfaces and have their own
resolution and sampling error; see [the evaluator documentation](crates/sdf-eval/README.md).

## Source

- Dataset: https://github.com/tovacinni/sdf-explorer
- Dataset revision: `c8b1b39796d18d2ad3e7442639ef51bf986eb187`
- Renderer used for dataset images: https://github.com/d-itu/sdf-view
- Dataset renderer revision: `51ff7f4f9e026d852520263374056a6af0740bbc`
- Development shell renderer revision: `cfb01c7535db99cec2351408672d069101789596` (0.2.0)

The dataset contains 63 GLSL models, each rendered to a 512 × 512 RGBA PNG.
All models passed compilation, PNG decoding, shader hash, nonempty foreground,
and image-border checks. These checks do not establish visual or geometric
correctness: visual review remains pending in every model's metadata.

Compatibility changes include local G3D constants, time fixed at zero, removal
of unused texture helpers, explicit array sizes, and a workaround for Naga's
scalar `inout` component handling. MantaRay additionally caps an upstream
unbounded zero-radius tail at local Z in `[-3, 3]`; this geometry truncation is
recorded explicitly in its metadata. Cameras are fitted per model without
changing shader coordinates. Source hashes refer to the downloaded upstream
files; `shader_sha256` refers to the final adapted shader.

Individual shaders have their own license
notices; the repository-level license is not a substitute for those notices.
