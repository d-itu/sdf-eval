use std::{array, borrow::Cow, env};

use anyhow::{Context, Result, anyhow};
use futures::{channel::oneshot, executor::block_on};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Adapter {
    pub name: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: Adapter,
    pub layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        block_on(async {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
            // Prefer Vulkan on Linux; WGPU_BACKEND can override this selection.
            if cfg!(target_os = "linux") && env::var_os("WGPU_BACKEND").is_none() {
                descriptor.backends = wgpu::Backends::VULKAN;
            }
            let instance = wgpu::Instance::new(descriptor);
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await?;
            let info = adapter.get_info();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("sdf-eval"),
                    ..Default::default()
                })
                .await?;
            let entries: Vec<_> = (0..4)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("IoU resources"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IoU layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            Ok(Self {
                device,
                queue,
                layout,
                pipeline_layout,
                adapter: Adapter {
                    name: info.name,
                    backend: format!("{:?}", info.backend),
                    driver: info.driver,
                    driver_info: info.driver_info,
                },
            })
        })
    }

    pub fn pipeline(&self, label: &str, source: String) -> Result<wgpu::ComputePipeline> {
        let mut frontend = wgpu::naga::front::glsl::Frontend::default();
        let mut ir = frontend
            .parse(
                &wgpu::naga::front::glsl::Options::from(wgpu::naga::ShaderStage::Compute),
                &source,
            )
            .map_err(|error| anyhow!(error.emit_to_string(&source)))?;
        for (_, function) in ir.functions.iter_mut() {
            workgroup_barriers(&mut function.body);
        }
        for entry in &mut ir.entry_points {
            workgroup_barriers(&mut entry.function.body);
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Naga(Cow::Owned(ir)),
            });
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.into());
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&self.pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.into());
        }
        Ok(pipeline)
    }

    pub fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn read_floats(&self, buffer: &wgpu::Buffer, count: u32) -> Result<Vec<f32>> {
        let slice = buffer.slice(..u64::from(count) * 4);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        block_on(receiver).map_err(|_| anyhow!("readback callback dropped"))??;
        let mapped = slice.get_mapped_range()?;
        let values = mapped
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        drop(mapped);
        buffer.unmap();
        Ok(values)
    }

    pub fn read_counts(&self, buffer: &wgpu::Buffer) -> Result<[u32; 8]> {
        let slice = buffer.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        block_on(receiver).map_err(|_| anyhow!("readback callback dropped"))??;
        let mapped = slice.get_mapped_range().context("map counts")?;
        let counts =
            array::from_fn(|i| u32::from_le_bytes(mapped[i * 4..i * 4 + 4].try_into().unwrap()));
        drop(mapped);
        buffer.unmap();
        Ok(counts)
    }
}

// Naga 30 lowers GLSL barrier() to Barrier::all(), incorrectly requiring
// subgroup features. GLSL compute barrier() synchronizes workgroup shared data.
fn workgroup_barriers(block: &mut wgpu::naga::Block) {
    use wgpu::naga::{Barrier, Statement};
    for statement in block.iter_mut() {
        match statement {
            Statement::ControlBarrier(flags) => *flags = Barrier::WORK_GROUP,
            Statement::Block(body) => workgroup_barriers(body),
            Statement::If { accept, reject, .. } => {
                workgroup_barriers(accept);
                workgroup_barriers(reject);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    workgroup_barriers(&mut case.body);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                workgroup_barriers(body);
                workgroup_barriers(continuing);
            }
            _ => {}
        }
    }
}
