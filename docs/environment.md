# Development environment

Run commands from the repository root. `flake.lock` pins Nixpkgs and the
`sdf-view` source. The renderer's upstream `Cargo.lock` pins Rust dependencies.
The shell provides the pinned `sdf-view` command and Vulkan runtime libraries.

```sh
nix develop path:.
sdf-view dataset/Sphere/sdf.glsl -o target/sphere-preview.png
# Explicit CPU fallback when no compatible GPU is available:
sdf-view-software dataset/Sphere/sdf.glsl -o target/sphere-preview-software.png
```

`sdf-view-software` uses the pinned Lavapipe Vulkan driver and can be much slower
than hardware rendering. Output paths determine where images are written.
There is no vendored renderer checkout. Multi-view rendering uses the shell's
renderer by default. The dataset preparation scripts require a separately built
`target/debug/sdf-view` executable at the dataset renderer revision listed below.

## Python dependencies

Dataset preparation needs Python with Pillow. Image evaluation and verification
share the locked Python 3.11 environment in `verification/.venv`, including
NumPy, Pillow, scikit-image, CPU PyTorch, torchvision, and LPIPS:

```sh
uv sync --project verification --locked
```

The Nix shell supplies native runtime libraries; re-enter it after updating
the shell.

## Renderer revisions

- Renderer repository: https://github.com/d-itu/sdf-view
- Dataset image revision: `51ff7f4f9e026d852520263374056a6af0740bbc`

The existing dataset images and the current development renderer use different
revisions. Use the same renderer version and backend for paired image evaluation.
