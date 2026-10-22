# Voxel topology protocol

```sh
cargo build --locked -p sdf-eval
./target/debug/sdf-eval topology \
  --reference dataset/Sphere/sdf.glsl \
  --prediction predictions/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 --resolution 256 \
  --output results/Sphere-topology.json
```

Inputs remain in their supplied coordinates. The command samples N^3 voxel
centers on the GPU, using the same positions and strict `sdf(p) < 0` interior
rule as IoU. Each occupied voxel contributes a closed cube and all its faces,
edges, and vertices to a cubical complex. This measures solid topology, not
the topology of a triangle surface. Zero is outside.

The CPU calculates Betti numbers in order `[beta_0, beta_1, beta_2]`:

- `beta_0`: occupied components with 26-connectivity, including edge and corner
  contact between closed cubes.
- `beta_2`: bounded components of the complement, using 6-connectivity for empty
  voxels. All empty components reaching the grid boundary connect to the exterior.
- `beta_1`: `beta_0 + beta_2 - chi`, where `chi = V - E + F - C` counts each unique
  vertex, edge, face, and cube. Alexander duality and the Euler identity give
  these ranks for the finite cubical complex in R^3; no manifold assumption or
  triangle meshing is required.

`reference_topology` and `prediction_topology` contain the Betti arrays, Euler
characteristic, occupied voxel count, and boundary flag. `metrics` contains
`betti_absolute_difference` and `betti_exact_match`. Equal Betti numbers do not
prove homeomorphism or geometric similarity.

If either input has no occupied voxel, status is `empty_occupancy`. This may
mean an empty or unsigned field, out-of-bounds geometry, or missed thin features;
the evaluator cannot distinguish these from samples alone. If an occupied
voxel touches the box boundary, status is `boundary_occupied`. Raw topology
is retained for diagnosis, but these cases have `metrics: null` and a nonzero
exit code. This conservative boundary policy prevents reporting a complete
shape match for potentially clipped solids. It examines outer voxel layers,
not the six additional face samples used by the IoU command.

Non-finite samples, input read errors, shader compilation failures, and GPU
errors also fail explicitly. Reports include the processing stage and adapter.
No shader hashes are computed or checked. Reference and prediction are processed
sequentially; GPU batches limit transfer sizes, but the CPU stores the complete
scalar field and then occupancy/visited grids and a flood-fill queue. Memory
and computation grow with N^3.

Topology is resolution-dependent. Check multiple resolutions when thin bridges
or small holes matter. Finite sampling cannot certify continuous SDF topology
or rule out geometry beyond the bounds.

For dataset aggregation, count `betti_exact_match: true` among reports with
`status: "ok"`, and divide by the number of such valid reports. Report the
valid-report fraction separately. Never treat failed reports as matching zeros.

Run `cargo test --locked -- --include-ignored` with a working Vulkan adapter
for GPU coverage of spheres, tori, shells, batching, and failure reporting.
