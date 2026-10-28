//! Serial manifest evaluation with bounded, per-reference-group caches.
use std::{
    collections::HashSet,
    fs,
    io::{self, BufWriter, Read, Write},
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    slice,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    gpu::Gpu,
    iou::{self, Settings},
    mesh::Point,
    metrics, self_check, shader, surface, topology,
};

#[derive(clap::Args)]
pub struct Args {
    /// JSON manifest; defaults to stdin. Relative shader paths use the manifest directory or cwd.
    #[arg(long)]
    input: Option<PathBuf>,
    /// JSONL results; defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
enum Metric {
    Iou,
    Surface,
    Topology,
    SelfCheck,
}
impl Metric {
    fn name(self) -> &'static str {
        match self {
            Self::Iou => "iou",
            Self::Surface => "surface",
            Self::Topology => "topology",
            Self::SelfCheck => "self_check",
        }
    }
}

#[derive(Default, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelfCheckOptions {
    probes: Option<u32>,
    rays: Option<u32>,
    oracle_steps: Option<u32>,
    max_steps: Option<u32>,
    epsilon: Option<f32>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceOptions {
    samples: Option<usize>,
    seed: Option<u64>,
    thresholds: Option<Vec<f64>>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    metrics: Option<Vec<Metric>>,
    resolution: Option<u32>,
    batch_size: Option<u32>,
    surface: Option<SurfaceOptions>,
    self_check: Option<SelfCheckOptions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobSpec {
    id: String,
    reference: Option<PathBuf>,
    prediction: PathBuf,
    bounds: [f32; 6],
    metrics: Option<Vec<Metric>>,
    resolution: Option<u32>,
    batch_size: Option<u32>,
    surface: Option<SurfaceOptions>,
    self_check: Option<SelfCheckOptions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    #[serde(default)]
    defaults: Options,
    jobs: Vec<JobSpec>,
}

struct Job {
    id: String,
    reference: Option<PathBuf>,
    prediction: PathBuf,
    settings: Settings,
    metrics: Vec<Metric>,
    samples: usize,
    seed: u64,
    thresholds: Vec<f64>,
    self_check: self_check::Args,
}

fn load(input: Option<&Path>) -> Result<Vec<Job>> {
    let (text, base) = if let Some(input) = input {
        (
            fs::read_to_string(input)?,
            input.parent().unwrap_or_else(|| Path::new(".")),
        )
    } else {
        let mut text = String::new();
        io::stdin().lock().read_to_string(&mut text)?;
        (text, Path::new("."))
    };
    let manifest: Manifest = serde_json::from_str(&text).context("invalid_manifest")?;
    ensure!(manifest.schema_version == 1, "unsupported schema_version");
    ensure!(!manifest.jobs.is_empty(), "jobs must not be empty");
    let defaults = manifest.defaults;
    let empty = SurfaceOptions::default();
    let default_surface = defaults.surface.as_ref().unwrap_or(&empty);
    let mut ids = HashSet::new();
    manifest
        .jobs
        .into_iter()
        .map(|job| {
            ensure!(
                !job.id.trim().is_empty() && ids.insert(job.id.clone()),
                "job id must be nonempty and unique: {}",
                job.id
            );
            ensure!(
                !job.prediction.as_os_str().is_empty(),
                "prediction path must not be empty: {}",
                job.id
            );
            let settings = Settings {
                bounds: job.bounds,
                resolution: job.resolution.or(defaults.resolution).unwrap_or(256),
                batch_size: job.batch_size.or(defaults.batch_size).unwrap_or(1_048_576),
            };
            settings
                .validate()
                .with_context(|| format!("invalid settings for {}", job.id))?;
            let metrics = job
                .metrics
                .or_else(|| defaults.metrics.clone())
                .unwrap_or_else(|| vec![Metric::Iou]);
            let unique: HashSet<_> = metrics.iter().collect();
            ensure!(
                !metrics.is_empty() && unique.len() == metrics.len(),
                "metrics must be nonempty and unique: {}",
                job.id
            );
            ensure!(
                metrics.iter().all(|m| *m == Metric::SelfCheck)
                    || job
                        .reference
                        .as_ref()
                        .is_some_and(|p| !p.as_os_str().is_empty()),
                "reference required for reference-based metrics: {}",
                job.id
            );
            let check = job.self_check.as_ref().or(defaults.self_check.as_ref());
            let check = check.cloned().unwrap_or_default();
            let self_check = self_check::Args {
                shader: base.join(&job.prediction),
                bounds: job.bounds,
                probes: check.probes.unwrap_or(16),
                rays: check.rays.unwrap_or(32),
                oracle_steps: check.oracle_steps.unwrap_or(512),
                max_steps: check.max_steps.unwrap_or(256),
                epsilon: check.epsilon,
                output: None,
            };
            if metrics.contains(&Metric::SelfCheck) || job.self_check.is_some() {
                self_check::validate(&self_check)
                    .with_context(|| format!("invalid self_check settings for {}", job.id))?;
            }
            let options = job.surface.as_ref().unwrap_or(&empty);
            let samples = options
                .samples
                .or(default_surface.samples)
                .unwrap_or(100_000);
            let seed = options.seed.or(default_surface.seed).unwrap_or(42);
            let thresholds = options
                .thresholds
                .as_ref()
                .or(default_surface.thresholds.as_ref())
                .cloned()
                .unwrap_or_default();
            if metrics.contains(&Metric::Surface) || job.surface.is_some() {
                surface::validate(samples, &thresholds)
                    .with_context(|| format!("invalid surface settings for {}", job.id))?;
            }
            Ok(Job {
                id: job.id,
                reference: job.reference.map(|p| base.join(p)),
                prediction: base.join(job.prediction),
                settings,
                metrics,
                samples,
                seed,
                thresholds,
                self_check,
            })
        })
        .collect()
}

#[derive(PartialEq, Eq)]
struct GroupKey {
    source: String,
    bounds: [u32; 6],
    resolution: u32,
    batch_size: u32,
    samples: usize,
    seed: u64,
}

#[derive(Default)]
struct Cache {
    iou_pipeline: Option<wgpu::ComputePipeline>,
    occupancy: Vec<wgpu::Buffer>,
    cloud: Option<(Vec<Point>, metrics::Index, Value, Value)>,
    topology: Option<(topology::Topology, Value)>,
}

impl Cache {
    fn evaluate(
        &mut self,
        gpu: &Gpu,
        reduce: &mut Option<wgpu::ComputePipeline>,
        job: &Job,
        sources: (&str, &str),
        metric: Metric,
        report: &mut Value,
    ) -> Result<Value> {
        let settings = job.settings;
        let (reference, prediction) = sources;
        report["adapter"] = json!(gpu.adapter);
        match metric {
            Metric::SelfCheck => {
                self_check::evaluate_prepared(gpu, prediction, &job.self_check, report)
            }
            Metric::Iou => {
                if self.iou_pipeline.is_none() {
                    self.iou_pipeline = Some(
                        gpu.pipeline("reference", shader::reference(reference))
                            .context("reference_compile_failed")?,
                    );
                }
                let prediction = gpu
                    .pipeline("prediction", shader::prediction(prediction))
                    .context("prediction_compile_failed")?;
                if reduce.is_none() {
                    *reduce = Some(
                        gpu.pipeline("reduction", shader::reduce())
                            .context("reduction_compile_failed")?,
                    );
                }
                let reused = !self.occupancy.is_empty();
                let value = iou::evaluate_prepared(
                    gpu,
                    self.iou_pipeline.as_ref().unwrap(),
                    prediction,
                    reduce.as_ref().unwrap(),
                    settings,
                    Some(&mut self.occupancy),
                );
                // Failed GPU execution must not leave a partially valid cache.
                if value.is_err() {
                    self.occupancy.clear();
                }
                report["reference_cache_hit"] = json!(reused);
                Ok(serde_json::to_value(value?)?)
            }
            Metric::Surface => {
                report["reference_cache_hit"] = json!(self.cloud.is_some());
                if self.cloud.is_none() {
                    let cloud = surface::cloud(
                        gpu,
                        reference,
                        settings,
                        job.samples,
                        job.seed,
                        "reference",
                        report,
                    )?;
                    let index = metrics::index(&cloud)?;
                    self.cloud = Some((
                        cloud,
                        index,
                        report["reference_mesh"].clone(),
                        report["effective_batch_size"].clone(),
                    ));
                }
                let (cloud, index, info, batch) = self.cloud.as_ref().unwrap();
                report["reference_mesh"] = info.clone();
                report["effective_batch_size"] = batch.clone();
                let prediction = surface::cloud(
                    gpu,
                    prediction,
                    settings,
                    job.samples,
                    job.seed ^ 0x9e3779b97f4a7c15,
                    "prediction",
                    report,
                )?;
                report["stage"] = json!("nearest_neighbors");
                let metrics = metrics::compare_indexed(cloud, index, &prediction, &job.thresholds)?;
                report["stage"] = json!("complete");
                Ok(
                    json!({"status": "ok", "metrics": metrics, "boundary_occupied": info["boundary_occupied"] == true || report["prediction_mesh"]["boundary_occupied"] == true}),
                )
            }
            Metric::Topology => {
                report["reference_cache_hit"] = json!(self.topology.is_some());
                if self.topology.is_none() {
                    let values =
                        surface::sample_field(gpu, reference, settings, true, "reference", report)?;
                    let occupied: Vec<_> = values.iter().map(|&v| v < 0.0).collect();
                    drop(values);
                    report["stage"] = json!("reference_topology");
                    self.topology = Some((
                        topology::calculate(&occupied, settings.resolution as usize),
                        report["effective_batch_size"].clone(),
                    ));
                }
                let (reference, batch) = self.topology.as_ref().unwrap();
                report["reference_topology"] = json!(reference);
                report["effective_batch_size"] = batch.clone();
                let values =
                    surface::sample_field(gpu, prediction, settings, true, "prediction", report)?;
                let occupied: Vec<_> = values.iter().map(|&v| v < 0.0).collect();
                drop(values);
                report["stage"] = json!("prediction_topology");
                let prediction = topology::calculate(&occupied, settings.resolution as usize);
                report["prediction_topology"] = json!(prediction);
                let boundary = reference.boundary_occupied || prediction.boundary_occupied;
                report["boundary_occupied"] = json!(boundary);
                ensure!(
                    reference.occupied_voxels > 0 && prediction.occupied_voxels > 0,
                    "empty_occupancy"
                );
                ensure!(!boundary, "boundary_occupied");
                let r = reference.betti;
                let p = prediction.betti;
                report["stage"] = json!("complete");
                Ok(json!({"status": "ok", "metrics": {
                    "betti_absolute_difference": [r[0].abs_diff(p[0]), r[1].abs_diff(p[1]), r[2].abs_diff(p[2])],
                    "betti_exact_match": r == p,
                }}))
            }
        }
    }
}

fn report(job: &Job, metric: Metric) -> Value {
    if metric == Metric::SelfCheck {
        return json!({
            "schema_version": 1, "tool_version": env!("CARGO_PKG_VERSION"),
            "id": job.id, "metric": metric.name(),
            "shader": {"path": job.prediction},
            "input": self_check::input(&job.self_check), "metrics": null,
        });
    }
    let mut report = json!({
        "schema_version": 1, "tool_version": env!("CARGO_PKG_VERSION"),
        "id": job.id, "metric": metric.name(),
        "reference": {"path": job.reference}, "prediction": {"path": job.prediction},
        "bounds": job.settings.bounds, "resolution": ([job.settings.resolution; 3]),
        "voxel_size": job.settings.voxel_size(), "requested_batch_size": job.settings.batch_size,
        "reference_cache_hit": false,
    });
    match metric {
        Metric::SelfCheck => unreachable!(),
        Metric::Iou => report["iou"] = Value::Null,
        Metric::Surface => {
            report["metrics"] = Value::Null;
            report["samples_per_surface"] = json!(job.samples);
            report["seed"] = json!(job.seed);
            report["thresholds"] = json!(job.thresholds);
            report["lattice_nodes"] = json!((u64::from(job.settings.resolution) + 1).pow(3));
            report["meshing"] = json!("marching_tetrahedra");
            report["sampling"] = json!("area_weighted_chacha8_independent");
            report["projection"] = json!(false);
        }
        Metric::Topology => {
            report["metrics"] = Value::Null;
            report["complex"] = json!("closed_voxel_cubes");
        }
    }
    if let Some(reference) = &job.reference {
        report["reference"] = json!({"path": reference});
    } else {
        report.as_object_mut().unwrap().remove("reference");
    }
    report
}

pub(crate) fn same_file(left: &Path, right: &Path) -> bool {
    if let (Ok(left), Ok(right)) = (left.canonicalize(), right.canonicalize())
        && left == right
    {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(left), Ok(right)) = (fs::metadata(left), fs::metadata(right)) {
            return left.dev() == right.dev() && left.ino() == right.ino();
        }
    }
    false
}

pub fn run(args: Args) -> Result<bool> {
    let jobs = load(args.input.as_deref())?;
    // Snapshot reference contents once. A path alone is not a safe cache key.
    let references: Vec<_> = jobs
        .iter()
        .map(|job| {
            job.reference.as_ref().map_or_else(
                || Ok(String::new()),
                |path| fs::read_to_string(path).map_err(|e| format!("reference_read_failed: {e}")),
            )
        })
        .collect();
    let mut groups: Vec<(GroupKey, Vec<usize>)> = Vec::new();
    let mut unreadable = Vec::new();
    for (i, job) in jobs.iter().enumerate() {
        let Ok(source) = &references[i] else {
            unreadable.push(i);
            continue;
        };
        let key = GroupKey {
            source: source.clone(),
            bounds: job.settings.bounds.map(f32::to_bits),
            resolution: job.settings.resolution,
            batch_size: job.settings.batch_size,
            samples: job.samples,
            seed: job.seed,
        };
        if let Some((_, indices)) = groups.iter_mut().find(|(k, _)| *k == key) {
            indices.push(i);
        } else {
            groups.push((key, vec![i]));
        }
    }
    // Do not truncate the manifest or shader inputs through an output alias.
    if let Some(output) = &args.output
        && output.exists()
    {
        for path in args
            .input
            .as_deref()
            .into_iter()
            .chain(jobs.iter().flat_map(|j| {
                j.reference
                    .as_deref()
                    .into_iter()
                    .chain([j.prediction.as_path()])
            }))
        {
            ensure!(
                !same_file(path, output),
                "output must not overwrite an input"
            );
        }
    }
    let mut output: Box<dyn Write> = if let Some(path) = &args.output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        Box::new(BufWriter::new(fs::File::create(path)?))
    } else {
        Box::new(io::stdout().lock())
    };
    let mut gpu = None;
    let mut reduce = None;
    let mut valid = true;
    // Unreadable references still produce a failure for every requested metric.
    let work = groups
        .iter()
        .map(|(_, indices)| indices.as_slice())
        .chain(unreadable.iter().map(slice::from_ref));
    for indices in work {
        let mut cache = Cache::default();
        for &i in indices {
            let job = &jobs[i];
            let prediction = fs::read_to_string(&job.prediction)
                .map_err(|e| format!("prediction_read_failed: {e}"));
            for &metric in &job.metrics {
                let started = Instant::now();
                let mut report = report(job, metric);
                let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<Value> {
                    let reference = if metric == Metric::SelfCheck {
                        ""
                    } else {
                        references[i]
                            .as_ref()
                            .map_err(|e| anyhow::anyhow!("{e}"))
                            .context("reference_read_failed")?
                    };
                    let prediction = prediction
                        .as_ref()
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .context("prediction_read_failed")?;
                    if gpu.is_none() {
                        gpu = Some(Gpu::new().context("adapter_initialization_failed")?);
                    }
                    cache.evaluate(
                        gpu.as_ref().unwrap(),
                        &mut reduce,
                        job,
                        (reference, prediction),
                        metric,
                        &mut report,
                    )
                }));
                match result {
                    Ok(Ok(value)) => report
                        .as_object_mut()
                        .unwrap()
                        .extend(value.as_object().unwrap().clone()),
                    Ok(Err(error)) => {
                        report["status"] = json!(error.to_string());
                        report["error"] = json!(format!("{error:#}"));
                    }
                    Err(payload) => {
                        report["status"] = json!("evaluation_panicked");
                        report["error"] = json!(
                            payload
                                .downcast_ref::<String>()
                                .map(String::as_str)
                                .or_else(|| payload.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown panic")
                        );
                        // Do not reuse potentially poisoned GPU state after a panic.
                        cache = Cache::default();
                        reduce = None;
                        gpu = None;
                    }
                }
                valid &= report["status"] == "ok";
                report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
                serde_json::to_writer(&mut output, &report)?;
                output.write_all(b"\n")?;
                output.flush()?;
                eprintln!("{} {}: {}", job.id, metric.name(), report["status"]);
            }
        }
    }
    Ok(valid)
}
