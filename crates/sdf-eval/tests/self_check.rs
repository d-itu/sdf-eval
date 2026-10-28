//! Reference-free GPU diagnostics on analytic fields.
use serde_json::Value;
use std::{path::Path, process::Command};

fn check(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let output = Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
        .arg("self-check")
        .arg("--shader")
        .arg(path)
        .args([
            "--bounds=-2,-2,-2,2,2,2",
            "--probes",
            "8",
            "--rays",
            "12",
            "--oracle-steps",
            "512",
        ])
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&output.stderr)));
    assert_eq!(
        output.status.success(),
        report["status"] == "ok",
        "{report}"
    );
    report
}

#[test]
#[ignore = "requires a compute adapter"]
fn scale_changes_detect_overshoot_and_iteration_cost() {
    let exact = check("sphere.glsl");
    let high = check("overscaled_sphere.glsl");
    let low = check("underscaled_sphere.glsl");
    let field = |v: &Value, key: &str| v["metrics"]["field"][key].as_f64().unwrap();
    let rays = |v: &Value, key: &str| v["metrics"]["rays"][key].as_f64().unwrap();
    assert_eq!(field(&exact, "overshoot_fraction"), 0.0);
    assert!(field(&high, "overshoot_fraction") > 0.0, "{high}");
    assert_eq!(field(&low, "overshoot_fraction"), 0.0);
    assert!(
        rays(&high, "pre_crossing_overestimate_fraction") > 0.0,
        "{high}"
    );
    assert_eq!(rays(&low, "missed_fraction"), 0.0, "{low}");
    assert!(
        low["metrics"]["rays"]["iterations_median_correct_hits"]
            .as_f64()
            .unwrap()
            > exact["metrics"]["rays"]["iterations_median_correct_hits"]
                .as_f64()
                .unwrap()
    );
}

#[test]
#[ignore = "requires a compute adapter"]
fn invalid_fields_do_not_report_safe() {
    let nan = check("nan.glsl");
    assert_eq!(nan["status"], "invalid_samples");
    assert!(nan["metrics"]["field"]["finite_fraction"].as_f64().unwrap() < 1.0);
    let empty = check("empty.glsl");
    assert_eq!(empty["status"], "no_first_crossings_observed");
    assert!(empty["metrics"]["rays"]["correct_hit_fraction"].is_null());
}
