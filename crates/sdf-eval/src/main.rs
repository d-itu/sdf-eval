mod gpu;
mod iou;
mod shader;

use std::{
    fs,
    io::{self, Write},
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::json;

#[derive(Parser)]
#[command(version, about = "Evaluate aligned GLSL SDFs using GPU volumetric IoU")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure volume overlap within an explicit shared bounding box.
    Iou(IouArgs),
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

fn run(args: IouArgs) -> Result<bool> {
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
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<iou::Evaluation> {
        settings.validate().context("invalid_settings")?;
        report["voxel_size"] = json!(settings.voxel_size());
        let reference = fs::read_to_string(&args.reference).context("reference_read_failed")?;
        let prediction = fs::read_to_string(&args.prediction).context("prediction_read_failed")?;
        iou::evaluate(&reference, &prediction, settings)
    }));
    let valid = match result {
        Ok(Ok(evaluation)) => {
            let valid = evaluation.status == "ok";
            let value = serde_json::to_value(evaluation)?;
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
    let Command::Iou(args) = Cli::parse().command;
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}
