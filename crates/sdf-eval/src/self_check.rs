//! Reference-free field diagnostics and fixed-ray sphere-tracing checks.
use std::{
    array, fs,
    io::{self, Write},
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};

use crate::{gpu::Gpu, iou::Settings};

#[derive(ClapArgs)]
pub struct Args {
    /// Standalone GLSL snippet defining float sdf(vec3 p).
    #[arg(long)]
    pub(crate) shader: PathBuf,
    /// xmin,ymin,zmin,xmax,ymax,zmax; no transformation is applied.
    #[arg(long, value_parser = super::parse_bounds, allow_hyphen_values = true)]
    pub(crate) bounds: [f32; 6],
    /// Number of field probe positions per axis.
    #[arg(long, default_value_t = 16)]
    pub(crate) probes: u32,
    /// Rays per side on each of the six box faces.
    #[arg(long, default_value_t = 32)]
    pub(crate) rays: u32,
    /// Uniform oracle intervals along each ray. Thin features may be missed.
    #[arg(long, default_value_t = 512)]
    pub(crate) oracle_steps: u32,
    /// Maximum sphere-tracing steps per ray.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_steps: u32,
    /// Hit and zero-root residual tolerance in shader coordinates.
    #[arg(long)]
    pub(crate) epsilon: Option<f32>,
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
}

const HEADER: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std140) uniform sdf_eval_Parameters {
    vec4 sdf_eval_lower;
    vec4 sdf_eval_upper;
    uvec4 sdf_eval_grid;
};
layout(set = 0, binding = 1, std430) buffer sdf_eval_Values { float sdf_eval_values[]; };
"#;

const PROBES: &str = r#"
float sdf_eval_length_scale() { return max(max(sdf_eval_upper.x-sdf_eval_lower.x, sdf_eval_upper.y-sdf_eval_lower.y), sdf_eval_upper.z-sdf_eval_lower.z); }
vec3 sdf_eval_gradient(vec3 p, float h) {
    return vec3(
        sdf(p + vec3(h,0,0)) - sdf(p - vec3(h,0,0)),
        sdf(p + vec3(0,h,0)) - sdf(p - vec3(0,h,0)),
        sdf(p + vec3(0,0,h)) - sdf(p - vec3(0,0,h))) / (2.0*h);
}
void main() {
    uint local = gl_GlobalInvocationID.x;
    if (local >= sdf_eval_grid.z) return;
    uint side = sdf_eval_grid.x;
    uint i = sdf_eval_grid.y + local;
    vec3 cell = vec3(i % side, (i / side) % side, i / (side*side));
    vec3 p = mix(sdf_eval_lower.xyz, sdf_eval_upper.xyz, (cell + 0.5) / float(side));
    float scale = sdf_eval_length_scale();
    float h = scale / 4096.0;
    float eps = sdf_eval_lower.w;
    float value = sdf(p);
    float grad = length(sdf_eval_gradient(p, h));
    vec3 q = p;
    float current = value;
    float verified = 0.0;
    float distance_to_root = 0.0;
    float residual = 0.0;
    float steps = 0.0;
    for (int step = 0; step < 128; ++step) {
        if (current == 0.0) { verified = 1.0; break; }
        vec3 direction = sdf_eval_gradient(q, h);
        float norm = length(direction);
        if (!(norm > 1e-6) || isnan(norm) || isinf(norm)) break;
        vec3 next = q - sign(current) * direction / norm * min(max(abs(current)*0.25, eps*2.0), scale/16.0);
        if (any(lessThan(next, sdf_eval_lower.xyz)) || any(greaterThan(next, sdf_eval_upper.xyz))) break;
        float v = sdf(next);
        if (isnan(v) || isinf(v)) break;
        steps += 1.0;
        if (v * current < 0.0) {
            vec3 a = q;
            vec3 b = next;
            for (int k = 0; k < 16; ++k) {
                vec3 mid = (a+b)*0.5;
                if (sdf(mid)*current > 0.0) a = mid; else b = mid;
            }
            q = (a+b)*0.5;
            current = sdf(q);
            verified = abs(current) <= eps ? 2.0 : 0.0;
            break;
        }
        q = next;
        current = v;
    }
    if (verified > 0.5) { distance_to_root = length(q-p); residual = abs(current); }
    uint o = local*8u;
    sdf_eval_values[o] = value;
    sdf_eval_values[o+1u] = grad;
    sdf_eval_values[o+2u] = distance_to_root;
    sdf_eval_values[o+3u] = verified;
    sdf_eval_values[o+4u] = residual;
    sdf_eval_values[o+5u] = steps;
    sdf_eval_values[o+6u] = 0.0;
    sdf_eval_values[o+7u] = 0.0;
}
"#;

const RAYS: &str = r#"
void main() {
    uint local = gl_GlobalInvocationID.x;
    if (local >= sdf_eval_grid.z) return;
    uint side = sdf_eval_grid.x;
    uint index = sdf_eval_grid.y + local;
    uint face = index / (side*side);
    uint u = index % side;
    uint v = (index / side) % side;
    uint axis = face / 2u;
    uint a = (axis+1u)%3u;
    uint b = (axis+2u)%3u;
    vec3 lower = sdf_eval_lower.xyz;
    vec3 upper = sdf_eval_upper.xyz;
    vec3 origin = (lower+upper)*0.5;
    origin[a] = mix(lower[a], upper[a], (float(u)+0.5)/float(side));
    origin[b] = mix(lower[b], upper[b], (float(v)+0.5)/float(side));
    origin[axis] = face%2u == 0u ? lower[axis] : upper[axis];
    vec3 direction = vec3(0.0);
    direction[axis] = face%2u == 0u ? 1.0 : -1.0;
    float limit = upper[axis]-lower[axis];
    float eps = sdf_eval_lower.w;
    float oracle = -1.0;
    float previous = sdf(origin);
    bool invalid = isnan(previous) || isinf(previous) || previous <= 0.0;
    for (uint k = 1u; k <= sdf_eval_grid.w && !invalid; ++k) {
        float t = limit * float(k) / float(sdf_eval_grid.w);
        float value = sdf(origin + t*direction);
        if (isnan(value) || isinf(value)) { invalid = true; break; }
        if (previous > 0.0 && value <= 0.0) {
            float lo = limit*float(k-1u)/float(sdf_eval_grid.w);
            float hi = t;
            for (int j = 0; j < 16; ++j) {
                float mid = (lo+hi)*0.5;
                if (sdf(origin+mid*direction) > 0.0) lo = mid; else hi = mid;
            }
            oracle = (lo+hi)*0.5;
            break;
        }
        previous = value;
    }
    float t = 0.0;
    float hit = 0.0;
    float iterations = 0.0;
    float crossing = 0.0;
    float limit_exhausted = 0.0;
    if (!invalid) {
        for (uint k = 0u; k < uint(sdf_eval_upper.w); ++k) {
            if (t > limit) break;
            float value = sdf(origin+t*direction);
            if (isnan(value) || isinf(value)) { invalid = true; break; }
            iterations += 1.0;
            if (value <= eps) { hit = 1.0; break; }
            if (oracle >= 0.0 && t < oracle - eps && value > oracle-t+eps) crossing = 1.0;
            t += value;
        }
    }
    if (!invalid && hit < 0.5 && t <= limit && iterations >= sdf_eval_upper.w) limit_exhausted = 1.0;
    uint o = local*8u;
    sdf_eval_values[o] = oracle;
    sdf_eval_values[o+1u] = hit;
    sdf_eval_values[o+2u] = iterations;
    sdf_eval_values[o+3u] = crossing;
    sdf_eval_values[o+4u] = hit > 0.5 ? t : -1.0;
    sdf_eval_values[o+5u] = float(invalid);
    sdf_eval_values[o+6u] = limit_exhausted;
    sdf_eval_values[o+7u] = 0.0;
}
"#;

fn dispatch(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    args: &Args,
    mode: u32,
    epsilon: f32,
) -> Result<Vec<[f32; 8]>> {
    let side = if mode == 0 { args.probes } else { args.rays };
    let total = if mode == 0 {
        side.pow(3)
    } else {
        6 * side * side
    };
    let limits = gpu.device.limits();
    let batch = total
        .min(16_384)
        .min(limits.max_compute_workgroups_per_dimension * 256)
        .min((limits.max_storage_buffer_binding_size / 32) as u32)
        .min((limits.max_buffer_size / 32) as u32);
    ensure!(batch > 0, "device buffer limit too small");
    let uniform = gpu.buffer(
        "self check parameters",
        48,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let values = gpu.buffer(
        "self check values",
        u64::from(batch) * 32,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let unused_a = gpu.buffer("unused partials", 32, wgpu::BufferUsages::STORAGE);
    let unused_b = gpu.buffer("unused totals", 32, wgpu::BufferUsages::STORAGE);
    let readback = gpu.buffer(
        "self check readback",
        u64::from(batch) * 32,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let buffers = [&uniform, &values, &unused_a, &unused_b];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("self check resources"),
        layout: &gpu.layout,
        entries: &entries,
    });
    let mut output = Vec::new();
    for offset in (0..total).step_by(batch as usize) {
        let count = batch.min(total - offset);
        let words = [
            args.bounds[0].to_bits(),
            args.bounds[1].to_bits(),
            args.bounds[2].to_bits(),
            epsilon.to_bits(),
            args.bounds[3].to_bits(),
            args.bounds[4].to_bits(),
            args.bounds[5].to_bits(),
            (args.max_steps as f32).to_bits(),
            side,
            offset,
            count,
            args.oracle_steps,
        ];
        let bytes: [u8; 48] = array::from_fn(|i| words[i / 4].to_le_bytes()[i % 4]);
        gpu.queue.write_buffer(&uniform, 0, &bytes);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("self check batch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("self check"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values, 0, &readback, 0, u64::from(count) * 32);
        gpu.queue.submit([encoder.finish()]);
        let floats = gpu.read_floats(&readback, count * 8)?;
        output.extend(floats.as_chunks::<8>().0.iter());
    }
    Ok(output)
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = fraction * (sorted.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    Some(sorted[low] + (index - low as f64) * (sorted[high] - sorted[low]))
}

fn fraction(part: usize, total: usize) -> Option<f64> {
    (total > 0).then(|| part as f64 / total as f64)
}

fn summarize(probes: &[[f32; 8]], rays: &[[f32; 8]], epsilon: f32) -> Value {
    let valid: Vec<_> = probes
        .iter()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    let bracketed: Vec<_> = valid.iter().filter(|p| p[3] > 1.5).collect();
    let outside: Vec<_> = bracketed.iter().filter(|p| p[0] > epsilon).collect();
    // A bracketed zero supplies an upper bound on nearest-surface distance.
    let overshoots: Vec<f64> = outside
        .iter()
        .filter(|p| p[0] - p[2] > 2.0 * epsilon)
        .map(|p| f64::from((p[0] - p[2]) / p[2].max(epsilon)))
        .collect();
    let discrepancies: Vec<f64> = bracketed
        .iter()
        .map(|p| f64::from((p[0].abs() - p[2]).abs() / p[2].max(epsilon)))
        .collect();
    let gradients: Vec<f64> = valid
        .iter()
        .map(|p| f64::from((p[1] - 1.0).abs()))
        .collect();
    let verified_rays: Vec<_> = rays.iter().filter(|r| r[5] == 0.0 && r[0] >= 0.0).collect();
    let correct: Vec<_> = verified_rays
        .iter()
        .filter(|r| r[1] > 0.5 && (r[4] - r[0]).abs() <= 4.0 * epsilon)
        .collect();
    let iterations: Vec<f64> = correct.iter().map(|r| f64::from(r[2])).collect();
    json!({
        "field": {
            "finite_fraction": fraction(valid.len(), probes.len()),
            "outside_bracketed_roots": outside.len(),
            "overshoot_fraction": fraction(overshoots.len(), outside.len()),
            "overshoot_relative_max": percentile(&overshoots, 1.0),
            "candidate_relative_discrepancy_p95": percentile(&discrepancies, 0.95),
            "gradient_magnitude_error_p95": percentile(&gradients, 0.95),
            "gradient_above_one_fraction": fraction(valid.iter().filter(|p| p[1] > 1.02).count(), valid.len()),
        },
        "rays": {
            "verified_first_crossings": verified_rays.len(),
            "correct_hit_fraction": fraction(correct.len(), verified_rays.len()),
            "missed_fraction": fraction(verified_rays.iter().filter(|r| r[1] < 0.5).count(), verified_rays.len()),
            "early_hit_fraction": fraction(verified_rays.iter().filter(|r| r[1] > 0.5 && r[4] < r[0] - 4.0 * epsilon).count(), verified_rays.len()),
            "late_hit_fraction": fraction(verified_rays.iter().filter(|r| r[1] > 0.5 && r[4] > r[0] + 4.0 * epsilon).count(), verified_rays.len()),
            "pre_crossing_overestimate_fraction": fraction(verified_rays.iter().filter(|r| r[3] > 0.5).count(), verified_rays.len()),
            "invalid_fraction": fraction(rays.iter().filter(|r| r[5] > 0.5).count(), rays.len()),
            "unverified_hits": rays.iter().filter(|r| r[5] == 0.0 && r[0] < 0.0 && r[1] > 0.5).count(),
            "step_limit_fraction": fraction(rays.iter().filter(|r| r[6] > 0.5).count(), rays.len()),
            "iterations_median_correct_hits": percentile(&iterations, 0.5),
            "iterations_p95_correct_hits": percentile(&iterations, 0.95),
        }
    })
}

pub(crate) fn validate(args: &Args) -> Result<f32> {
    let settings = Settings {
        bounds: args.bounds,
        resolution: args.probes,
        batch_size: 1_048_576,
    };
    settings.validate().context("invalid_settings")?;
    ensure!(
        args.rays > 0
            && args.rays <= 64
            && args.probes <= 32
            && args.oracle_steps > 0
            && args.oracle_steps <= 4096
            && args.max_steps > 0
            && args.max_steps <= 1024,
        "invalid_settings: ray/step limits exceeded"
    );
    let scale = (0..3)
        .map(|a| args.bounds[a + 3] - args.bounds[a])
        .fold(0.0_f32, f32::max);
    let epsilon = args.epsilon.unwrap_or(scale / 10_000.0);
    ensure!(
        epsilon.is_finite() && epsilon > 0.0 && epsilon < scale / 16.0,
        "invalid_settings: epsilon must be positive and small relative to bounds"
    );
    Ok(epsilon)
}

pub(crate) fn input(args: &Args) -> Value {
    json!({
        "bounds": args.bounds,
        "epsilon": args.epsilon,
        "field": {"probes_per_axis": args.probes},
        "rays": {
            "rays_per_face_side": args.rays,
            "oracle_steps": args.oracle_steps,
            "max_steps": args.max_steps,
        },
    })
}

pub(crate) fn evaluate_prepared(
    gpu: &Gpu,
    source: &str,
    args: &Args,
    report: &mut Value,
) -> Result<Value> {
    let epsilon = validate(args)?;
    report["input"]["epsilon"] = json!(epsilon);
    report["adapter"] = json!(gpu.adapter);
    report["stage"] = json!("probe_compile");
    let probes = gpu
        .pipeline(
            "self check probes",
            format!("{HEADER}\n#line 1 1\n{source}\n#line 1 0\n{PROBES}"),
        )
        .context("shader_compile_failed")?;
    report["stage"] = json!("ray_compile");
    let rays = gpu
        .pipeline(
            "self check rays",
            format!("{HEADER}\n#line 1 1\n{source}\n#line 1 0\n{RAYS}"),
        )
        .context("shader_compile_failed")?;
    report["stage"] = json!("field_probes");
    let probe_results = dispatch(gpu, &probes, args, 0, epsilon).context("gpu_execution_failed")?;
    report["stage"] = json!("ray_tracing");
    let ray_results = dispatch(gpu, &rays, args, 1, epsilon).context("gpu_execution_failed")?;
    report["stage"] = json!("complete");
    let status = if probe_results
        .iter()
        .any(|p| !p[0].is_finite() || !p[1].is_finite())
        || ray_results.iter().any(|r| r[5] > 0.5)
    {
        "invalid_samples"
    } else if !ray_results.iter().any(|r| r[0] >= 0.0) {
        "no_first_crossings_observed"
    } else {
        "ok"
    };
    Ok(json!({"status": status, "metrics": summarize(&probe_results, &ray_results, epsilon)}))
}

fn evaluate(args: &Args, report: &mut Value) -> Result<Value> {
    validate(args)?;
    report["stage"] = json!("shader_read");
    let source = fs::read_to_string(&args.shader).context("shader_read_failed")?;
    report["stage"] = json!("adapter_initialization");
    let gpu = Gpu::new().context("adapter_initialization_failed")?;
    evaluate_prepared(&gpu, &source, args, report)
}

pub fn run(args: Args) -> Result<bool> {
    if let Some(output) = &args.output {
        ensure!(
            !crate::batch::same_file(&args.shader, output),
            "output must not overwrite shader"
        );
    }
    let started = Instant::now();
    let mut report = json!({
        "schema_version": 1, "tool_version": env!("CARGO_PKG_VERSION"), "metric": "self_check",
        "shader": {"path": args.shader}, "input": input(&args),
        "metrics": null,
    });
    let result = panic::catch_unwind(AssertUnwindSafe(|| evaluate(&args, &mut report)));
    let valid = match result {
        Ok(Ok(value)) => {
            report
                .as_object_mut()
                .unwrap()
                .extend(value.as_object().unwrap().clone());
            value["status"] == "ok"
        }
        Ok(Err(error)) => {
            report["status"] = json!(error.to_string());
            report["error"] = json!(format!("{error:#}"));
            false
        }
        Err(_) => {
            report["status"] = json!("evaluation_panicked");
            false
        }
    };
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    let text = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = &args.output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
        eprintln!(
            "{}: {}",
            report["status"].as_str().unwrap_or("unknown"),
            path.display()
        );
    } else {
        io::stdout().lock().write_all(text.as_bytes())?;
    }
    Ok(valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summaries_do_not_treat_unknown_rays_as_safe() {
        let probes = [[2.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0]];
        let rays = [
            [1.0, 0.0, 3.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        ];
        let m = summarize(&probes, &rays, 0.001);
        assert!(m["field"].get("probes").is_none());
        assert!(m["rays"].get("total").is_none());
        assert_eq!(m["field"]["finite_fraction"], 1.0);
        assert_eq!(m["field"]["overshoot_fraction"], 1.0);
        assert_eq!(m["field"]["overshoot_relative_max"], 1.0);
        assert_eq!(m["rays"]["verified_first_crossings"], 1);
        assert_eq!(m["rays"]["correct_hit_fraction"], 0.0);
        assert_eq!(m["rays"]["missed_fraction"], 1.0);
        assert_eq!(m["rays"]["pre_crossing_overestimate_fraction"], 1.0);
        assert_eq!(m["rays"]["unverified_hits"], 0);
        assert!(m["rays"]["iterations_median_correct_hits"].is_null());
    }

    #[test]
    fn empty_denominators_are_null_not_perfect() {
        let m = summarize(&[], &[], 0.001);
        assert!(m["field"]["finite_fraction"].is_null());
        assert!(m["field"]["overshoot_fraction"].is_null());
        assert!(m["rays"]["correct_hit_fraction"].is_null());
        assert!(m["rays"]["step_limit_fraction"].is_null());
        let m = summarize(
            &[[0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            &[[-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]],
            0.001,
        );
        assert_eq!(m["field"]["finite_fraction"], 1.0);
        assert!(m["field"]["overshoot_fraction"].is_null());
        assert!(m["rays"]["correct_hit_fraction"].is_null());
    }
}
