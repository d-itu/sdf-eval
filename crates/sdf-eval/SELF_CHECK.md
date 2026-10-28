# Reference-free SDF self-check

`self-check` evaluates one GLSL shader against its **own** zero set. It does not
compare geometry with a reference shader or use reconstruction images.

Batch jobs can request `self_check` alone without a `reference`, or combine it
with reference-based metrics when `reference` is present. The evaluated shader
is always `prediction`. Parameters may be specified in `defaults.self_check`
or overridden by a job's `self_check` object:

```json
{
  "schema_version": 1,
  "defaults": {
    "metrics": ["self_check"],
    "self_check": {"probes": 16, "rays": 32, "oracle_steps": 512,
                   "max_steps": 256, "epsilon": 0.0004}
  },
  "jobs": [
    {"id": "candidate", "prediction": "candidate.glsl",
     "bounds": [-2, -2, -2, 2, 2, 2]}
  ]
}
```

```sh
./target/debug/sdf-eval batch --input jobs.json --output results.jsonl
```

A batch emits one JSONL record per job and metric, including errors; the
`self_check` record has the same `input`, `metrics`, and status fields as the
standalone command. The effective evaluation settings appear in `input`:
`field.probes_per_axis` controls the field grid, while
`rays.rays_per_face_side`, `rays.oracle_steps`, and `rays.max_steps` control
ray tests. The bounds and effective epsilon are shared by both and live at
`input.bounds` and `input.epsilon`. No input settings appear in `metrics`.
Job-level `self_check` options replace the default option object.

The standalone command remains available:

```sh
cargo build --locked -p sdf-eval
./target/debug/sdf-eval self-check \
  --shader dataset/Sphere/sdf.glsl \
  --bounds=-2,-2,-2,2,2,2 \
  --probes 16 --rays 32 --oracle-steps 512 --max-steps 256 \
  --output results/Sphere-self-check.json
```

The bounds must contain the whole object, with positive field values at the
six box faces. The evaluator does not fit or normalize the shader. `--probes`
is the number of cell-center field probes per axis (default 16). `--rays` is
the number of rays per side on each box face (default 32); rays travel along
all six axis directions at pixel centers. `--oracle-steps` is the number of
uniform intervals per ray (default 512). `--max-steps` caps tracing iterations
(default 256). `--epsilon` is the value-space hit and root tolerance (default:
longest box side / 10000). A nonzero exit and a JSON status other than `ok`
indicate invalid samples or no observed first crossings. Reports retain
partial diagnostic counts in these cases. There is no in-process timeout for
untrusted shaders; run the evaluator under a subprocess deadline.

## Field probes

Each point is evaluated on the GPU. Central differences at `h = L/4096`, where
`L` is the longest box side, estimate gradient magnitude. The probe follows
`-sign(f) * normalize(gradient)` with bounded adaptive steps until it brackets
an opposite-sign value; 16 bisections locate a zero candidate. A zero at the
starting point is also accepted. Root searches can fail at stationary points,
discontinuities, box edges, or after the 128-step limit. Only outside probes
with a bracketed root enter the confirmed-overshoot denominator.

The distance to this candidate is an **upper bound** on the distance to the
nearest zero, provided the sign change corresponds to a continuous zero set.
When `f(p) - |p-q| > 2*epsilon` on an outside probe, the result counts as a
confirmed overshoot. `overshoot_fraction` divides these by
`outside_bracketed_roots`; `overshoot_relative_max` is the largest
`(f(p) - |p-q|) / max(|p-q|, epsilon)` among overshoots. The candidate distance
need not be the nearest distance; `candidate_relative_discrepancy_p95` is the
P95 of `||f(p)| - |p-q|| / max(|p-q|, epsilon)` over bracketed roots, **not** a
true-distance error. It can grow near the surface because of the epsilon
floor. No overshoot observed is not a proof of safety. The root residual is
required to be at most epsilon, but cannot certify continuity of arbitrary
GLSL functions. `gradient_magnitude_error_p95` uses `abs(norm - 1)` over
finite probes; `gradient_above_one_fraction` uses the exploratory tolerance
`norm > 1.02`. Gradient norm above one does not by itself prove unsafe
stepping. `finite_fraction` uses all probes as its denominator. Probe positions
are uniform in volume, so thin or small features may be underrepresented;
raise `--probes` or tighten the box to investigate.

## Rays

An oracle scans each fixed ray at uniform intervals for the first positive-to-
nonpositive sample transition, then bisects that interval 16 times using the
shader. Sphere tracing independently advances by the unmodified positive
`f(p)` until `f <= epsilon`, it leaves the box, or reaches `max_steps`.
The report keeps `verified_first_crossings` as an observed coverage count;
the total ray count is determined by `input.rays.rays_per_face_side` as
`6 * rays_per_face_side^2`. Likewise, the probe count is determined by
`input.field.probes_per_axis` as `probes_per_axis^3`. Input-derived totals are not
included in `metrics`. `correct_hit_fraction`, `missed_fraction`,
`early_hit_fraction`, and
`late_hit_fraction` divide by confirmed first crossings; early and late mean
ray-position error greater than `4*epsilon`. These four outcomes are mutually
exclusive. `pre_crossing_overestimate_fraction` uses the same denominator and
counts rays on which a sampled positive step value exceeded the remaining
ray distance to the observed first crossing. This is **not** a count of steps
that actually crossed the surface. `invalid_fraction` and
`step_limit_fraction` divide by all rays; `unverified_hits` counts hits without
an observed oracle crossing. They are **not** classified as false hits: the
uniform oracle can miss thin geometry or tangencies. All fractions lie in
`[0,1]`, and fractions with no eligible denominator are `null`, not zero.
The iteration median/P95 only use hits within the positional tolerance, in
raw step counts; they are `null` with no correct hits. An early hit near a
grazing ray can result from the value-space epsilon even for a conservative
field.

These rays are an axis-aligned, orthographic diagnostic, not the `sdf-view`
renderer camera or its internal termination policy. Increase ray density and
oracle intervals when studying small features; stable results across settings
increase confidence but cannot prove safety for every ray. Reports are
reference-free and should not be interpreted as geometric reconstruction
scores. Tests with a unit sphere, twice its distance, half its distance,
non-finite output, and empty output require a compute adapter:

```sh
cargo test --locked -p sdf-eval --test self_check -- --include-ignored
```
