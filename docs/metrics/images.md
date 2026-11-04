# Image evaluation

`scripts/evaluate_images.py` compares paired multi-view renders with RGB SSIM
(higher is better) and AlexNet LPIPS v0.1 (lower is better). Set up the locked
Python 3.11 environment, then run:

```sh
uv sync --project verification --locked
verification/.venv/bin/python scripts/evaluate_images.py \
  --reference results/Sphere/reference_views/views.json \
  --prediction results/Sphere/prediction_views/views.json \
  --output results/Sphere/image_metrics.json
```

Generate input manifests with `scripts/render_views.py`.
The output must be a new file. CPU is the default; pass `--device cuda` for
CUDA. Pretrained AlexNet weights download on first use to
`results/.cache/torch` (override with `--cache-dir`); prepare this cache before
offline evaluation. Dependencies and pretrained weights must be available for
scoring. Reports record package versions, metric parameters, cameras, shared
render settings, and each view's scores.
Do not assume LPIPS is bounded by 1 or convert it to a percentage.

## `views.json` contract

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
`distance` are provenance only and do not control matching. The manifest cannot verify the
renderer binary or the actual settings used to produce an image.

Views are paired by image name, independently of list order. Both manifests
must have exactly the same image names, shared render object, and per-view
camera parameters. Images must be RGBA PNGs matching the declared dimensions
(at least 64 pixels on each axis for AlexNet LPIPS). Views are never silently skipped.

## Metric protocol

RGBA is composited onto white using `rgb * alpha + (1 - alpha)` in normalized
sRGB values, without linearization. SSIM uses `data_range=1`, Gaussian weights,
sigma 1.5, and population covariance. LPIPS receives RGB tensors in [-1, 1].
A benchmark can fix 8 azimuths, elevations -30/0/30, top and bottom, and 4x
antialiasing for 26 views.

## Validation

```sh
verification/.venv/bin/python -m unittest discover -s scripts -p 'test_*views.py'
verification/.venv/bin/python -m unittest discover -s scripts -p 'test_evaluate_images.py'
```

To additionally test actual SSIM/LPIPS identity, transparency, and geometry
changes, run the latter command with `RUN_LPIPS_TESTS=1`; this may download
pretrained weights.
