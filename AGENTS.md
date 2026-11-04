# Project guide

- Run commands from the repository root. Rust implements rendering and geometry
  evaluation; Python handles orchestration and validation.
- Keep predictions and evaluation outputs in `predictions/` and `results/`.
- Run checks appropriate to the change. GPU tests require a compute adapter.
