# Documentation

All command examples assume the repository root as the working directory.

| Module | Contents |
| --- | --- |
| `environment.md` | Nix shell, renderer versions, Vulkan, Python dependencies |
| `dataset.md` | Layout, provenance, shader contract, adaptations, validation status |
| `rendering.md` | Dataset rendering, camera fitting, gallery, multi-view output |
| `evaluation.md` | Build, batch manifests, caching, execution, tests |
| `metrics/common.md` | Shared input, sampling, output, failure, and comparison conventions |
| `metrics/iou.md` | Volumetric overlap, boundary diagnostics, GPU reduction |
| `metrics/images.md` | SSIM, LPIPS, manifest contract, tests |
| `metrics/surface.md` | Chamfer Distance, F-score, extraction, sampling, limits |
| `metrics/topology.md` | Voxel connectivity, Betti numbers, validity rules |
| `metrics/self-check.md` | Reference-free field probes and sphere-tracing diagnostics |
| `verification.md` | Combined scoring entry point and production packaging |

## Repository map

- `dataset/`: 63 reference models, each with a shader, render, and metadata.
- `crates/sdf-eval/`: Rust GPU field evaluation and CPU surface/topology analysis.
- `scripts/`: rendering, camera fitting, dataset validation, image comparison,
  and verification build/smoke tools.
- `verification/`: scoring entry point, runtime bootstrap, and locked Python environment.
- `.github/workflows/`: Ubuntu verification bundle build.
- `preview.html`: offline dataset review gallery.
- `predictions/`, `results/`, `target/`: ignored predictions, evaluation artifacts,
  and build outputs.
