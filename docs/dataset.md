# Reference dataset

There are 63 model directories directly under `dataset/`. The upstream category
hierarchy is flattened; categories are retained in metadata.

```text
dataset/<ModelName>/
  sdf.glsl
  render.png
  metadata.json
```

Each model directory contains exactly these three files. `sdf.glsl` is the final
standalone shader, including compatibility edits. Do not add a second original
shader file. Retrieve upstream source through the URL and revision in metadata
when a comparison is needed.

## Provenance and licenses

- Dataset repository: https://github.com/tovacinni/sdf-explorer
- Dataset revision: `c8b1b39796d18d2ad3e7442639ef51bf986eb187`
- Upstream model location: `data-files/sdf/<Category>/<ModelName>.glsl`
- Renderer repository: https://github.com/d-itu/sdf-view
- Dataset renderer revision: `51ff7f4f9e026d852520263374056a6af0740bbc`

All downloads were checked against upstream Git blob hashes before adaptation.
Preserve shader copyright and license notices. Individual shaders may have
licenses different from the upstream repository-level license.

Source hashes refer to the downloaded upstream files; `shader_sha256` refers to
the final adapted shader used for rendering. Do not replace upstream hashes with
hashes of edited shaders. Consult each model's metadata for its provenance,
adaptations, camera and render settings, and visual review status.

## Shader contract and adaptations

The renderer accepts a GLSL snippet defining `float sdf(vec3 p)`, plus helper
functions. It supplies the shader entry point and resource setup. Do not add
`#version`, `main`, resource bindings, or names starting with `sdf_view_`.
The renderer uses wgpu and Naga's GLSL frontend, which supports a subset of GLSL.
The evaluator has its own reserved `sdf_eval_` prefix.

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
`[-3, 3]`. This truncation is documented in its metadata and needs particular
attention during visual review.

Consult each model's `modifications` list for its actual changes. Keep new
adaptations narrow, explain them in metadata, and distinguish geometry changes
from compatibility fixes. Rerender and validate after changing shaders or cameras.

## Validation status

Each model has a 512 × 512 RGBA PNG with a blue surface and transparent
background. Lighting is shared across models; cameras are fitted individually
without changing shader coordinates. Dynamic models use time zero.

All 63 models passed compilation and the current automated validator: exact
per-model file layout, final shader hashes, PNG decoding, image dimensions,
nonempty alpha bounds, and foreground not touching borders. The validator
expects exactly 63 model directories; update that expectation deliberately
if the dataset scope changes.

Visual review remains pending in every model's metadata. Automated checks do
not establish geometric fidelity, good semantic viewpoints, absence of
rendering artifacts, or visual similarity to upstream references. Inspect
`preview.html`, compare complex and adapted models with their
source references, and record actual review results in metadata.
