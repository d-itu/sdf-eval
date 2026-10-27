//! GPU lattice sampling followed by CPU meshing, sampling, and nearest neighbors.
use crate::{gpu::Gpu, iou::Settings, mesh::Mesh, metrics};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::array;

pub(crate) const HEADER: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std140) uniform sdf_eval_Parameters {
    vec4 sdf_eval_lower;
    vec4 sdf_eval_upper;
    uvec4 sdf_eval_grid;
};
layout(set = 0, binding = 1, std430) buffer sdf_eval_Values { float sdf_eval_values[]; };
"#;
pub(crate) const MAIN: &str = r#"
void main() {
    uint local = gl_GlobalInvocationID.x;
    if (local >= sdf_eval_grid.z) return;
    uint i = sdf_eval_grid.y + local;
    uint side = sdf_eval_grid.x;
    vec3 position = vec3(i % side, (i / side) % side, i / (side * side));
    vec3 fraction;
    if (sdf_eval_grid.w == 1u)
        fraction = (position + 0.5) / float(side);
    else
        fraction = position / float(side - 1u);
    vec3 p = sdf_eval_lower.xyz + fraction * (sdf_eval_upper.xyz - sdf_eval_lower.xyz);
    sdf_eval_values[local] = sdf(p);
}
"#;

pub(crate) fn field(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    settings: Settings,
    centers: bool,
) -> Result<(Vec<f32>, u32)> {
    let side = settings.resolution + u32::from(!centers);
    let total = side * side * side;
    let limits = gpu.device.limits();
    let batch = u64::from(settings.batch_size)
        .min(u64::from(total))
        .min(u64::from(limits.max_compute_workgroups_per_dimension) * 256)
        .min(limits.max_storage_buffer_binding_size / 4)
        .min(limits.max_buffer_size / 4) as u32;
    let uniform = gpu.buffer(
        "surface parameters",
        48,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let values = gpu.buffer(
        "surface values",
        u64::from(batch) * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let unused_a = gpu.buffer("unused partials", 32, wgpu::BufferUsages::STORAGE);
    let unused_b = gpu.buffer("unused totals", 32, wgpu::BufferUsages::STORAGE);
    let readback = gpu.buffer(
        "surface readback",
        u64::from(batch) * 4,
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
        label: Some("surface resources"),
        layout: &gpu.layout,
        entries: &entries,
    });
    let mut field = Vec::new();
    field.try_reserve_exact(total as usize)?;
    for offset in (0..total).step_by(batch as usize) {
        let count = batch.min(total - offset);
        let words = [
            settings.bounds[0].to_bits(),
            settings.bounds[1].to_bits(),
            settings.bounds[2].to_bits(),
            0,
            settings.bounds[3].to_bits(),
            settings.bounds[4].to_bits(),
            settings.bounds[5].to_bits(),
            0,
            side,
            offset,
            count,
            u32::from(centers),
        ];
        let bytes: [u8; 48] = array::from_fn(|i| words[i / 4].to_le_bytes()[i % 4]);
        gpu.queue.write_buffer(&uniform, 0, &bytes);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("surface batch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface sampling"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values, 0, &readback, 0, u64::from(count) * 4);
        gpu.queue.submit([encoder.finish()]);
        let values = gpu.read_floats(&readback, count)?;
        ensure!(values.iter().all(|v| v.is_finite()), "non_finite_values");
        field.extend(values);
    }
    Ok((field, batch))
}

pub(crate) fn sample_field(
    gpu: &Gpu,
    source: &str,
    settings: Settings,
    centers: bool,
    name: &str,
    report: &mut Value,
) -> Result<Vec<f32>> {
    report["stage"] = json!(format!("{name}_compile"));
    let pipeline = gpu
        .pipeline(
            name,
            format!("{HEADER}\n#line 1 1\n{source}\n#line 1 0\n{MAIN}"),
        )
        .with_context(|| format!("{name}_compile_failed"))?;
    report["stage"] = json!(format!("{name}_field"));
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let memory = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = gpu.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let result = field(gpu, &pipeline, settings, centers);
    let internal_error = futures::executor::block_on(internal.pop());
    let memory_error = futures::executor::block_on(memory.pop());
    let validation_error = futures::executor::block_on(scope.pop());
    if let Some(error) = internal_error.or(memory_error).or(validation_error) {
        return Err(anyhow::Error::from(error).context("gpu_execution_failed"));
    }
    let (values, batch) = result?;
    report["effective_batch_size"] = json!(batch);
    Ok(values)
}

pub(crate) fn cloud(
    gpu: &Gpu,
    source: &str,
    settings: Settings,
    samples: usize,
    seed: u64,
    name: &str,
    report: &mut Value,
) -> Result<Vec<crate::mesh::Point>> {
    let values = sample_field(gpu, source, settings, false, name, report)?;
    report["stage"] = json!(format!("{name}_mesh"));
    let mesh = Mesh::extract(&values, settings.bounds, settings.resolution)?;
    drop(values);
    report[format!("{name}_mesh")] = json!(mesh.info);
    Ok(mesh.sample(samples, seed))
}

pub fn validate(samples: usize, thresholds: &[f64]) -> Result<()> {
    ensure!(samples > 0, "samples must be positive");
    ensure!(
        !thresholds.is_empty() && thresholds.iter().all(|t| t.is_finite() && *t > 0.0),
        "thresholds must be finite and positive"
    );
    Ok(())
}

pub fn evaluate(
    reference: &str,
    prediction: &str,
    settings: Settings,
    samples: usize,
    seed: u64,
    thresholds: &[f64],
    report: &mut Value,
) -> Result<Value> {
    report["lattice_nodes"] = json!((u64::from(settings.resolution) + 1).pow(3));
    report["stage"] = json!("adapter_initialization");
    let gpu = Gpu::new().context("adapter_initialization_failed")?;
    report["adapter"] = json!(gpu.adapter);
    let reference = cloud(
        &gpu,
        reference,
        settings,
        samples,
        seed,
        "reference",
        report,
    )?;
    let prediction = cloud(
        &gpu,
        prediction,
        settings,
        samples,
        seed ^ 0x9e3779b97f4a7c15,
        "prediction",
        report,
    )?;
    report["stage"] = json!("nearest_neighbors");
    let metrics = metrics::compare(&reference, &prediction, thresholds)?;
    report["stage"] = json!("complete");
    Ok(json!({
        "status": "ok",
        "metrics": metrics,
        "boundary_occupied": report["reference_mesh"]["boundary_occupied"] == true || report["prediction_mesh"]["boundary_occupied"] == true,
    }))
}
