# SDF benchmark dataset

Each directory under `dataset/` contains one model's final `sdf.glsl`,
`render.png`, and `metadata.json`. Upstream provenance and shader adaptations
are recorded in the metadata. Shader copyright and license notices are retained.

## Environment

Run commands from the repository root. `flake.lock` pins Nixpkgs and the
`sdf-view` source. The renderer's upstream `Cargo.lock` pins Rust dependencies.
The shell provides Rust, Cargo, Python with Pillow, and the Vulkan loader.
It uses the host graphics driver; GPU results may vary across drivers.

```sh
nix develop path:.
cargo build --locked --manifest-path "$SDF_VIEW_SOURCE/Cargo.toml"
./target/debug/sdf-view dataset/Sphere/sdf.glsl -o target/sphere-preview.png
python3 scripts/render_dataset.py
python3 scripts/validate_dataset.py
```

Pass model names to render selected samples, for example
`python3 scripts/render_dataset.py Snail Elephant`. Rendering uses the settings
stored in each model's metadata. `python3 scripts/fit_cameras.py` can refit
cameras from foreground bounds. Open `preview.html` to inspect all outputs.
Regenerate the gallery after rendering with `scripts/validate_dataset.py`.

The build uses the default Cargo dev profile and writes build outputs to
`target/` inside this repository. The shell preserves the user's `CARGO_HOME`
and `XDG_CACHE_HOME` settings. Nix manages its own store. `.envrc` supports
optional direnv use.

## Source

- Dataset: https://github.com/tovacinni/sdf-explorer
- Dataset revision: `c8b1b39796d18d2ad3e7442639ef51bf986eb187`
- Renderer: https://github.com/d-itu/sdf-view
- Renderer revision: `51ff7f4f9e026d852520263374056a6af0740bbc`

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
