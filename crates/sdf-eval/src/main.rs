mod batch;
mod gpu;
mod iou;
mod mesh;
mod metrics;
mod self_check;
mod shader;
mod surface;
mod topology;

use std::{
    fs,
    io::{self, Write},
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use serde_json::json;

#[derive(Parser)]
#[command(
    version,
    about = "Evaluate aligned GLSL SDFs with volume and surface metrics"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure volume overlap within an explicit shared bounding box.
    Iou(IouArgs),
    /// Sample zero-isosurfaces and compute Chamfer distance and F-scores.
    Surface(SurfaceArgs),
    /// Compare Betti numbers of occupied closed voxel cubes.
    Topology(IouArgs),
    /// Evaluate a JSON manifest and stream one JSONL record per job and metric.
    Batch(batch::Args),
    /// Assess a shader against its own zero set and test ray-march safety.
    SelfCheck(self_check::Args),
}

#[derive(Args)]
struct SurfaceArgs {
    #[command(flatten)]
    common: IouArgs,
    /// Number of independent area-weighted samples per surface.
    #[arg(long, default_value_t = 100_000)]
    samples: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Positive distance thresholds in input coordinate units.
    #[arg(long, value_delimiter = ',', required = true)]
    thresholds: Vec<f64>,
}

#[derive(Args)]
struct IouArgs {
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    prediction: PathBuf,
    /// xmin,ymin,zmin,xmax,ymax,zmax; neither input is transformed.
    #[arg(long, value_parser = parse_bounds, allow_hyphen_values = true)]
    bounds: [f32; 6],
    /// Cells per axis: centers for IoU/topology, corners for surface extraction.
    #[arg(long, default_value_t = 256)]
    resolution: u32,
    /// Maximum sample positions per batch; clamped to device limits.
    #[arg(long, default_value_t = 1_048_576)]
    batch_size: u32,
    /// Write JSON here; otherwise write to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn parse_bounds(text: &str) -> Result<[f32; 6], String> {
    let values: Vec<f32> = text
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())?;
    values
        .try_into()
        .map_err(|_| "expected six comma-separated coordinates".to_owned())
}

fn run(command: Command) -> Result<bool> {
    let command = match command {
        Command::Batch(args) => return batch::run(args),
        Command::SelfCheck(args) => return self_check::run(args),
        command => command,
    };
    let topology = matches!(&command, Command::Topology(_));
    let (args, surface_options) = match command {
        Command::Iou(args) | Command::Topology(args) => (args, None),
        Command::Surface(args) => (
            args.common,
            Some((args.samples, args.seed, args.thresholds)),
        ),
        Command::Batch(_) | Command::SelfCheck(_) => unreachable!(),
    };
    if let Some(output) = &args.output {
        for input in [&args.reference, &args.prediction] {
            ensure!(
                !batch::same_file(input, output),
                "output must not overwrite an input"
            );
        }
    }
    let started = Instant::now();
    let settings = iou::Settings {
        bounds: args.bounds,
        resolution: args.resolution,
        batch_size: args.batch_size,
    };
    let mut report = json!({
        "schema_version": 1,
        "tool_version": env!("CARGO_PKG_VERSION"),
        "reference": {"path": args.reference},
        "prediction": {"path": args.prediction},
        "bounds": settings.bounds,
        "resolution": [settings.resolution, settings.resolution, settings.resolution],
        "requested_batch_size": settings.batch_size,
        "iou": null,
    });
    if topology {
        report.as_object_mut().unwrap().remove("iou");
        report["metric"] = json!("topology");
        report["metrics"] = serde_json::Value::Null;
        report["complex"] = json!("closed_voxel_cubes");
    }
    if let Some((samples, seed, thresholds)) = &surface_options {
        report.as_object_mut().unwrap().remove("iou");
        report["metric"] = json!("surface");
        report["metrics"] = serde_json::Value::Null;
        report["samples_per_surface"] = json!(samples);
        report["seed"] = json!(seed);
        report["thresholds"] = json!(thresholds);
        report["meshing"] = json!("marching_tetrahedra");
        report["sampling"] = json!("area_weighted_chacha8_independent");
        report["projection"] = json!(false);
    }
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<serde_json::Value> {
        settings.validate().context("invalid_settings")?;
        if let Some((samples, _, thresholds)) = &surface_options {
            surface::validate(*samples, thresholds).context("invalid_settings")?;
        }
        report["voxel_size"] = json!(settings.voxel_size());
        let reference = fs::read_to_string(&args.reference).context("reference_read_failed")?;
        let prediction = fs::read_to_string(&args.prediction).context("prediction_read_failed")?;
        if let Some((samples, seed, thresholds)) = &surface_options {
            surface::evaluate(
                &reference,
                &prediction,
                settings,
                *samples,
                *seed,
                thresholds,
                &mut report,
            )
        } else if topology {
            topology::evaluate(&reference, &prediction, settings, &mut report)
        } else {
            Ok(serde_json::to_value(iou::evaluate(
                &reference,
                &prediction,
                settings,
            )?)?)
        }
    }));
    let valid = match result {
        Ok(Ok(value)) => {
            let valid = value["status"] == "ok";
            report
                .as_object_mut()
                .unwrap()
                .extend(value.as_object().unwrap().clone());
            valid
        }
        Ok(Err(error)) => {
            report["status"] = json!(error.to_string());
            report["error"] = json!(format!("{error:#}"));
            false
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            report["status"] = json!("evaluation_panicked");
            report["error"] = json!(message);
            false
        }
    };
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    let text = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = args.output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
        eprintln!("{}: {}", report["status"].as_str().unwrap(), path.display());
    } else {
        io::stdout().lock().write_all(text.as_bytes())?;
    }
    Ok(valid)
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}
