# Project handoff

## Scope

This repository contains prepared image inputs and GPU volumetric IoU plus
surface metrics (Chamfer Distance and F-score) for a benchmark of LLM-generated
SDF shaders. There is no LLM API integration or benchmark orchestration runner.

The dataset comes from `tovacinni/sdf-explorer`. Rendering uses the Rust
`sdf-view` renderer. Python scripts handle batch orchestration, camera fitting,
and artifact validation; they do not implement the renderer.

## Dataset layout

There are 63 model directories directly under `dataset/`. The upstream category
hierarchy is flattened; categories are retained in metadata.

```text
dataset/<ModelName>/
  sdf.glsl
  render.png
  metadata.json
```

Each model directory contains exactly these three files. `sdf.glsl` is the
final standalone shader, including any compatibility edits. Do not add a second
original shader file. Retrieve upstream source through the URL and revision in
metadata when a comparison is needed.

- Dataset repository: https://github.com/tovacinni/sdf-explorer
- Dataset revision: `c8b1b39796d18d2ad3e7442639ef51bf986eb187`
- Upstream model location: `data-files/sdf/<Category>/<ModelName>.glsl`
- Renderer repository: https://github.com/d-itu/sdf-view
- Renderer revision: `51ff7f4f9e026d852520263374056a6af0740bbc`

All downloads were checked against upstream Git blob hashes before adaptation.
Preserve shader copyright and license notices. Individual shaders may have
licenses different from the upstream repository-level license.

## Shader contract and adaptations

The renderer accepts a GLSL snippet defining `float sdf(vec3 p)`, plus helper
functions. It supplies the shader entry point and resource setup. Do not add
`#version`, `main`, resource bindings, or names starting with `sdf_view_`.
The renderer uses wgpu and Naga's GLSL frontend, which supports a subset of GLSL.

Twenty-six models required adaptation:

- Define missing G3D `pi` constants and squared-length helpers locally.
- Freeze `g3d_SceneTime` at zero for animated models.
- Replace the `frac` alias with GLSL `fract` in Mountain.
- Remove redundant `const` input qualifiers in Gear.
- Remove unused texture helpers in Girl and unused terrain noise and sampler
  arguments in Tree. These removed paths do not contribute to the active SDF.
- Give Lamborghini's wheel parameter arrays explicit lengths.
- Replace Jetfighter's scalar `inout` component calls with returned values and
  an inline mirror operation to avoid a Naga SPIR-V backend panic.

MantaRay has an intentional geometry change: its upstream zero-radius tail
extends indefinitely. The adapted shader caps that tail at local Z in
`[-3, 3]`. This truncation is documented in its metadata and should receive
particular attention during visual review.

Consult each model's `modifications` list for its actual changes. Keep new
adaptations narrow, explain them in metadata, and distinguish geometry changes
from compatibility fixes.

## Rendering and metadata

Current images are 512 by 512 RGBA PNGs with a blue surface and transparent
background. Lighting is shared across models; cameras are fitted individually
without changing shader coordinates. Dynamic models use time zero.

`metadata.json` records:

- Model name, upstream category, source URL, revision, and original hashes.
- Copyright and license notices extracted from source comments.
- Adaptation notes and fixed scene time where applicable.
- The final shader's `shader_sha256`.
- Camera, light, resolution, and renderer revision.
- Render status, elapsed time, foreground bounds, coverage, and border checks.
- Visual review status.

`source.sha256` and `source.git_blob_sha1` identify the downloaded upstream file;
`shader_sha256` identifies the adapted file used for the current render. Do not
replace upstream hashes with hashes of edited shaders.

## Working commands

Run commands from the repository root with the renderer available at
`target/debug/sdf-view` and Python with Pillow available.

```sh
# Render only the models being changed.
python3 scripts/render_dataset.py Snail Elephant

# Refit selected cameras using foreground bounds when necessary.
python3 scripts/fit_cameras.py Snail Elephant

# Validate all artifacts and regenerate the offline gallery.
python3 scripts/validate_dataset.py
```

Omit model arguments to render or fit the whole dataset. Batch rendering runs
the Rust renderer in a subprocess with a default 90-second timeout per model;
`render_dataset.py --timeout` can override this limit. It writes render errors
and statuses to metadata.

Edit the `render` settings in a model's metadata for deliberate camera changes,
then render that model again. Camera fitting is a heuristic based on alpha
bounds, not a semantic choice of the best viewpoint.

After editing a shader, update its adaptation notes, render it again, and run
the validator. Do not leave a stale PNG or shader hash paired with new source.
The render script resets `visual_review` to `pending`, including after a
previously reviewed sample is rendered again.

`preview.html` links every image, shader, and metadata file and serves as the
visual review gallery. Scratch logs and build artifacts are not dataset
deliverables.

## Project structure

The temporary `tools/sdf-view` checkout and old rendering logs were removed;
there is no vendored renderer copy. Build the renderer before rerendering;
the preparation scripts expect it at `target/debug/sdf-view`. The three dataset
preparation scripts support
focused rerendering, camera fitting, and validation with gallery generation.


## IoU evaluation

`crates/sdf-eval/` implements GPU volumetric IoU with wgpu 30 and Naga in the
root Cargo workspace. Build with `cargo build --locked -p sdf-eval`.
Reference and prediction shaders must already be aligned; never translate,
rotate, or rescale them automatically. `--bounds=xmin,ymin,zmin,xmax,ymax,zmax`
is required. The default is 256 voxel centers per axis, with `sdf(p) < 0`
defining the interior. The evaluator does not calculate or validate hashes.

Separate reference and prediction compute pipelines avoid helper-name
collisions. GPU passes reduce intersection, union, invalid-value, and boundary
counts; only eight counters per batch are read back, accumulated in `u64`.
Empty unions and non-finite values return null IoU and a nonzero exit code.
Boundary occupancy is a separate flag: the score still measures overlap within
the supplied box. Six faces are sampled separately and do not enter IoU counts.

Naga 30's GLSL barrier lowering requires a correction in `gpu.rs`: use
workgroup barriers instead of all barrier flags. Keep this workaround scoped
to GLSL compute semantics and reconsider it when updating Naga.

Run `cargo test --locked -- --include-ignored` with a working compute adapter
for the complete integration tests. GPU tests are ignored by default. All 63
dataset shaders have compiled and executed, but some shaders have thin,
unsigned, or otherwise unsuitable interiors for volume IoU. Do not reinterpret
an empty union as a perfect match or assume a renderable shader has a valid
signed solid. See `crates/sdf-eval/README.md` for protocol and known limitations.

Keep generated predictions and evaluation results outside the reference
dataset, in the ignored root-level `predictions/` and `results/` directories.

## Topology evaluation

`sdf-eval topology` reuses GPU field sampling at voxel centers and computes
solid Betti numbers on the CPU for the union of occupied closed cubes.
Foreground is 26-connected; complement components are 6-connected.
Euler cell counts and bounded complement components determine the three ranks.
Empty occupancy and occupied outer voxel layers retain raw topology but return
null metrics and a nonzero exit. They must not count as exact matches.
See `crates/sdf-eval/TOPOLOGY.md` for the protocol and memory limitations.

## Validation status and remaining work

At handoff, all 63 models rendered successfully and passed the current
validator: exact per-model file layout, final shader hashes, PNG decoding,
image dimensions, nonempty alpha bounds, and foreground not touching borders.
`validate_dataset.py` currently expects exactly 63 model directories; update
that expectation deliberately if the dataset scope changes.

Visual review is still pending for every model. Image attachments could not be
viewed during preparation. Automated checks do not establish geometric fidelity,
good semantic viewpoints, absence of rendering artifacts, or visual similarity
to upstream references. Do not describe the dataset as visually verified.

The next collaborator should inspect `preview.html`, review complex and adapted
models against their source references, adjust viewpoints or shaders where
needed, and record actual review results in metadata. Re-render and revalidate
after any change affecting the image.
