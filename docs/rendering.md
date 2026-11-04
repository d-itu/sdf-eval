# Rendering workflows

Use Python with Pillow available. Rust performs rendering; Python scripts coordinate
subprocesses, camera fitting, and artifact validation.

## Dataset rendering and camera fitting

Build `sdf-view` at revision `51ff7f4f9e026d852520263374056a6af0740bbc`
and place its executable at `target/debug/sdf-view` before running the
preparation scripts. They use this fixed path and record that revision in metadata.

```sh
python3 scripts/render_dataset.py Snail Elephant
python3 scripts/fit_cameras.py Snail Elephant
python3 scripts/validate_dataset.py
```

Omit model arguments to render or fit the whole dataset. Rendering uses each
model's metadata settings. The default timeout is 90 seconds per model;
`render_dataset.py --timeout` overrides it. Render errors and statuses are
written to metadata.

Edit a model's `render` settings for deliberate camera changes, then render
that model again. Camera fitting is a heuristic based on foreground bounds,
not a semantic choice of the best viewpoint. It does not change shader coordinates.

After editing a shader, update its adaptation notes, rerender, and run the
validator. Keep the PNG and shader hash consistent with the final source.
Rendering resets `visual_review` to `pending`. The validator regenerates
`preview.html`, which links every image, shader, and metadata file for offline
review. Scratch logs and build artifacts are not dataset deliverables.

## Multi-view rendering

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
`az045_el+30.png`) or `top.png` / `bottom.png`. `views.json` contains image
names, camera parameters, and shared render settings.

For software rendering, pass `--renderer sdf-view-software`. `--width`,
`--height`, `--fov`, `--antialiasing`, and `--timeout` are also configurable.
The output directory must not exist; if any view fails, the script prints
the error and leaves no output directory.

Choose cameras using only the reference and reuse them for all predictions.
