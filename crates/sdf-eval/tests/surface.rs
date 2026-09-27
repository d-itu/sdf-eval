//! Surface protocol integration tests. GPU cases are opt-in like the IoU suite.
use serde_json::Value;
use std::{path::Path, process::Command};

fn evaluate(reference: &str, prediction: &str, extra: &[&str]) -> Value {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
        .arg("surface")
        .arg("--reference")
        .arg(fixtures.join(reference))
        .arg("--prediction")
        .arg(fixtures.join(prediction))
        .args([
            "--bounds=-2,-2,-2,2,2,2",
            "--resolution",
            "24",
            "--samples",
            "2000",
            "--thresholds",
            "0.4,0.6",
        ])
        .args(extra)
        .output()
        .unwrap();
    let value: Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&output.stderr)));
    assert_eq!(output.status.success(), value["status"] == "ok", "{value}");
    value
}

#[test]
#[ignore = "requires a compute adapter"]
fn surface_metrics_and_reproducibility() {
    let nested = evaluate("sphere.glsl", "small_sphere.glsl", &[]);
    assert_eq!(nested["status"], "ok", "{nested}");
    let metrics = &nested["metrics"];
    assert!((metrics["chamfer_l2_mean"].as_f64().unwrap() - 0.5).abs() < 0.012);
    assert!((metrics["chamfer_l2_squared_mean"].as_f64().unwrap() - 0.25).abs() < 0.015);
    assert_eq!(metrics["f_scores"][0]["f_score"], 0.0);
    assert_eq!(metrics["f_scores"][1]["f_score"], 1.0);
    assert_eq!(nested["boundary_occupied"], false);
    let split = evaluate("sphere.glsl", "small_sphere.glsl", &["--batch-size", "257"]);
    assert_eq!(metrics, &split["metrics"]);
    assert_eq!(nested["reference_mesh"], split["reference_mesh"]);
    let same = evaluate("sphere.glsl", "sphere.glsl", &[]);
    let cd = same["metrics"]["chamfer_l2_mean"].as_f64().unwrap();
    assert!(cd > 0.0 && cd < 0.06, "{same}");
    assert_eq!(same["metrics"]["f_scores"][0]["f_score"], 1.0);
}

#[test]
#[ignore = "requires a compute adapter"]
fn surface_failures_and_clipped_boundary() {
    for (reference, prediction, status, stage) in [
        (
            "empty.glsl",
            "sphere.glsl",
            "no_surface_extracted",
            "reference_mesh",
        ),
        (
            "sphere.glsl",
            "full.glsl",
            "no_surface_extracted",
            "prediction_mesh",
        ),
        (
            "infinity.glsl",
            "sphere.glsl",
            "non_finite_values",
            "reference_field",
        ),
        (
            "sphere.glsl",
            "nan.glsl",
            "non_finite_values",
            "prediction_field",
        ),
        (
            "invalid.glsl",
            "sphere.glsl",
            "reference_compile_failed",
            "reference_compile",
        ),
        (
            "sphere.glsl",
            "invalid.glsl",
            "prediction_compile_failed",
            "prediction_compile",
        ),
    ] {
        let value = evaluate(reference, prediction, &[]);
        assert_eq!(value["status"], status, "{value}");
        assert_eq!(value["stage"], stage);
        assert!(value["metrics"].is_null());
    }
    let value = evaluate("disjoint_sphere.glsl", "sphere.glsl", &[]);
    assert_eq!(value["status"], "ok", "{value}");
    assert_eq!(value["boundary_occupied"], true);
}

#[test]
fn invalid_surface_settings_fail_before_reading_inputs() {
    for (samples, thresholds) in [("0", "0.1"), ("10", "0"), ("10", "NaN")] {
        let output = Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
            .args([
                "surface",
                "--reference",
                "unused",
                "--prediction",
                "unused",
                "--bounds=-1,-1,-1,1,1,1",
                "--samples",
                samples,
                "--thresholds",
                thresholds,
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["status"], "invalid_settings");
        assert!(value["metrics"].is_null());
    }
}

#[test]
fn thresholds_are_required() {
    let output = Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
        .args([
            "surface",
            "--reference",
            "unused",
            "--prediction",
            "unused",
            "--bounds=-1,-1,-1,1,1,1",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--thresholds"));
}
