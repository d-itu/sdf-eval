# Evaluation tools

## Build and run

From the repository root, in the development environment:

```sh
cargo build --locked -p sdf-eval
./target/debug/sdf-eval iou \
  --reference dataset/Sphere/sdf.glsl \
  --prediction predictions/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 \
  --resolution 256 \
  --output results/Sphere.json
```

## Manifest batch evaluation

The single-comparison commands remain available. Use `batch` to evaluate a JSON
manifest with one GPU device shared across jobs:

```sh
./target/debug/sdf-eval batch --input jobs.json --output results/batch.jsonl
./target/debug/sdf-eval batch --output results/batch.jsonl < jobs.json
```

For a `jobs.json` at the repository root:

```json
{
  "schema_version": 1,
  "defaults": {
    "metrics": ["iou", "surface", "topology"],
    "resolution": 128,
    "batch_size": 1048576,
    "surface": {
      "samples": 100000,
      "seed": 42,
      "thresholds": [0.01, 0.02, 0.05]
    }
  },
  "jobs": [
    {
      "id": "sphere-attempt-1",
      "reference": "dataset/Sphere/sdf.glsl",
      "prediction": "predictions/attempt-1/Sphere.glsl",
      "bounds": [-2, -2, -2, 2, 2, 2]
    },
    {
      "id": "sphere-attempt-2",
      "reference": "dataset/Sphere/sdf.glsl",
      "prediction": "predictions/attempt-2/Sphere.glsl",
      "bounds": [-2, -2, -2, 2, 2, 2],
      "surface": {"thresholds": [0.02, 0.05]}
    }
  ]
}
```

`schema_version` must be 1 and `jobs` must be nonempty. Each job requires a
nonempty unique `id`, input paths, and explicit shared `bounds`.
With `--input`, shader paths are relative to the
manifest directory. Without `--input`, the manifest is read from stdin and
shader paths are relative to the current working directory. Absolute paths are accepted.
Unknown fields, duplicate metrics, and invalid settings reject the entire
manifest before opening the output. Existing inputs must not be used as output.

Jobs can override `metrics`, `resolution`, `batch_size`, and `surface`. Surface
options inherit individually; a thresholds array replaces the default array
rather than extending it. The default metric selection is `["iou"]`. Surface jobs require
a nonempty array of positive finite thresholds in input coordinate units.
`batch_size` still means positions per GPU dispatch batch, not job count.

Execution is serial, grouped by actual reference source contents, bounds,
resolution, batch size, surface sample count, and seed. Identical source at
different paths shares a group. Threshold changes do not invalidate the cache.
Reference contents are snapshotted before evaluation. Within a group, IoU
reuses the compiled reference pipeline and GPU classification buffers; surface
metrics reuse reference points and their KD-tree; topology reuses reference
Betti numbers. IoU and topology use voxel centers, while surface meshing uses
lattice corners; these sampling conventions are not combined. The reduction
pipeline is shared across groups. Group caches are released before the next
group, with no disk cache or concurrent scheduling.

Caching trades memory for less repeated work. IoU reference buffers retain
roughly `4 * (N^3 + 6*N^2)` bytes per active group, in addition to working
buffers. Surface points and their index, and any retained topology result,
remain live until the group finishes. Input source strings are retained for
the run. With one prediction per reference the main benefit is shared GPU
initialization, not repeated-reference preprocessing.

Output is JSONL: one flushed record per job and metric, with the same result
fields and validity rules as the corresponding single command, plus `id`,
`metric`, `requested_batch_size`, and `reference_cache_hit`. Cache hits describe
reused preprocessing for that metric, not an already cached prediction score.
Join results by `id` and `metric`: grouping can reorder jobs; unreadable
references are reported after readable groups. Metrics retain their requested
order within each job. Without `--output`, JSONL goes to stdout; progress and
diagnostics go to stderr. Per-record elapsed time includes only work performed
for that record, not manifest parsing, grouping, input reads, or prior cache
construction. It is not a standalone per-comparison timing benchmark.

Input read, compile, and metric failures produce records and do not stop later
work. A caught panic drops potentially poisoned GPU state before continuing.
The process exits 0 only when every record has status `ok`, otherwise 1.
Manifest and output I/O failures are fatal and reported to stderr; output I/O
failure can leave a partial JSONL file. Empty unions and invalid topology
remain failures, not perfect matches.

## Validation

```sh
cargo test --locked                         # Tests without a GPU
cargo test --locked -- --include-ignored    # Includes GPU integration tests
```

Manifest tests cover validation, relative paths, input/output alias protection,
and per-metric failure records without a GPU. The opt-in batch GPU test checks
all three metrics against single commands, content-based cache reuse, parameter
invalidation, threshold overrides, and recovery after an invalid prediction.

GPU tests require a compute adapter, including a software Vulkan adapter if
available.
