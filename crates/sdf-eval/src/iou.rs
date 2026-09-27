//! GPU volume IoU over a shared sampling box; no alignment is performed.

use std::array;

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{
    gpu::{Adapter, Gpu},
    shader,
};

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub bounds: [f32; 6],
    pub resolution: u32,
    pub batch_size: u32,
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.resolution > 0, "resolution must be positive");
        ensure!(self.batch_size > 0, "batch size must be positive");
        ensure!(
            self.bounds.iter().all(|v| v.is_finite()),
            "bounds must be finite"
        );
        for axis in 0..3 {
            let extent = self.bounds[axis + 3] - self.bounds[axis];
            ensure!(
                extent.is_finite() && extent > 0.0,
                "each upper bound must exceed its lower bound with a finite extent"
            );
        }
        let n = u64::from(self.resolution);
        let total = n
            .checked_mul(n)
            .and_then(|square| square.checked_mul(n + 6));
        ensure!(
            total.is_some_and(|t| t <= u64::from(u32::MAX)),
            "sample count exceeds 32-bit GPU indexing"
        );
        Ok(())
    }

    pub fn voxel_size(&self) -> [f32; 3] {
        array::from_fn(|axis| (self.bounds[axis + 3] - self.bounds[axis]) / self.resolution as f32)
    }
}

#[derive(Debug, Serialize, Default, PartialEq, Eq)]
pub struct Counts {
    pub intersection: u64,
    pub union: u64,
    pub reference_inside: u64,
    pub prediction_inside: u64,
    pub reference_non_finite: u64,
    pub prediction_non_finite: u64,
    pub reference_boundary_inside: u64,
    pub prediction_boundary_inside: u64,
}

impl From<[u64; 8]> for Counts {
    fn from(c: [u64; 8]) -> Self {
        Self {
            intersection: c[0],
            union: c[1],
            reference_inside: c[2],
            prediction_inside: c[3],
            reference_non_finite: c[4],
            prediction_non_finite: c[5],
            reference_boundary_inside: c[6],
            prediction_boundary_inside: c[7],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Evaluation {
    pub status: &'static str,
    pub iou: Option<f64>,
    pub counts: Counts,
    pub boundary_occupied: bool,
    pub volume_samples: u64,
    pub boundary_samples: u64,
    pub effective_batch_size: u32,
    pub adapter: Adapter,
}

pub fn evaluate(reference: &str, prediction: &str, settings: Settings) -> Result<Evaluation> {
    settings.validate()?;
    let gpu = Gpu::new().context("adapter_initialization_failed")?;
    let reference = gpu
        .pipeline("reference", shader::reference(reference))
        .context("reference_compile_failed")?;
    let prediction = gpu
        .pipeline("prediction", shader::prediction(prediction))
        .context("prediction_compile_failed")?;
    let reduce = gpu
        .pipeline("reduction", shader::reduce())
        .context("reduction_compile_failed")?;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let memory = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = gpu.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let result = sample(&gpu, &reference, &prediction, &reduce, settings);
    let internal_error = futures::executor::block_on(internal.pop());
    let memory_error = futures::executor::block_on(memory.pop());
    let validation_error = futures::executor::block_on(scope.pop());
    if let Some(error) = internal_error.or(memory_error).or(validation_error) {
        return Err(anyhow::Error::from(error).context("gpu_execution_failed"));
    }
    let (counts, batch) = result.context("gpu_execution_failed")?;
    let status = if counts.reference_non_finite + counts.prediction_non_finite > 0 {
        "non_finite_values"
    } else if counts.union == 0 {
        "empty_union"
    } else {
        "ok"
    };
    let iou = (status == "ok").then(|| counts.intersection as f64 / counts.union as f64);
    let n = u64::from(settings.resolution);
    Ok(Evaluation {
        status,
        iou,
        boundary_occupied: counts.reference_boundary_inside + counts.prediction_boundary_inside > 0,
        counts,
        volume_samples: n * n * n,
        boundary_samples: 6 * n * n,
        effective_batch_size: batch,
        adapter: gpu.adapter,
    })
}

fn sample(
    gpu: &Gpu,
    reference: &wgpu::ComputePipeline,
    prediction: &wgpu::ComputePipeline,
    reduce: &wgpu::ComputePipeline,
    settings: Settings,
) -> Result<(Counts, u32)> {
    let n = settings.resolution;
    let volume = n * n * n;
    let total = volume + 6 * n * n;
    let limits = gpu.device.limits();
    let batch = u64::from(settings.batch_size)
        .min(u64::from(total))
        .min(u64::from(limits.max_compute_workgroups_per_dimension) * 256)
        .min(limits.max_storage_buffer_binding_size / 4)
        .min(limits.max_buffer_size / 4) as u32;
    let uniform = gpu.buffer(
        "parameters",
        48,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let occupancy = gpu.buffer(
        "reference occupancy",
        u64::from(batch) * 4,
        wgpu::BufferUsages::STORAGE,
    );
    let partials = gpu.buffer(
        "partial counts",
        u64::from(batch.div_ceil(256)) * 32,
        wgpu::BufferUsages::STORAGE,
    );
    let totals = gpu.buffer(
        "totals",
        32,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = gpu.buffer(
        "readback",
        32,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let buffers = [&uniform, &occupancy, &partials, &totals];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("IoU resources"),
        layout: &gpu.layout,
        entries: &entries,
    });
    let mut accumulated = [0u64; 8];
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
            n,
            offset,
            count,
            volume,
        ];
        let bytes: [u8; 48] = array::from_fn(|i| words[i / 4].to_le_bytes()[i % 4]);
        gpu.queue.write_buffer(&uniform, 0, &bytes);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("IoU batch"),
            });
        for (pipeline, groups) in [
            (reference, count.div_ceil(256)),
            (prediction, count.div_ceil(256)),
            (reduce, 1),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IoU pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&totals, 0, &readback, 0, 32);
        gpu.queue.submit([encoder.finish()]);
        let counts = gpu.read_counts(&readback)?;
        for (sum, count) in accumulated.iter_mut().zip(counts) {
            *sum += u64::from(count);
        }
    }
    Ok((accumulated.into(), batch))
}
