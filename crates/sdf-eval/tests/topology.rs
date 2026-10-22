use std::{path::Path, process::Command};

use serde_json::{Value, json};

fn evaluate(reference: &str, prediction: &str, extra: &[&str]) -> Value {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
        .arg("topology")
        .arg("--reference")
        .arg(fixtures.join(reference))
        .arg("--prediction")
        .arg(fixtures.join(prediction))
        .arg("--bounds=-2,-2,-2,2,2,2")
        .args(if extra.contains(&"--resolution") {
            Vec::new()
        } else {
            vec!["--resolution", "24"]
        })
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
fn known_solids_and_batch_invariance() {
    for (fixture, betti) in [
        ("sphere.glsl", [1, 0, 0]),
        ("torus.glsl", [1, 1, 0]),
        ("shell.glsl", [1, 0, 1]),
    ] {
        let value = evaluate("sphere.glsl", fixture, &[]);
        assert_eq!(value["status"], "ok", "{value}");
        assert_eq!(value["reference_topology"]["betti"], json!([1, 0, 0]));
        assert_eq!(value["prediction_topology"]["betti"], json!(betti));
        assert_eq!(value["metrics"]["betti_exact_match"], betti == [1, 0, 0]);
        assert_eq!(
            value["metrics"]["betti_absolute_difference"],
            json!([0, betti[1], betti[2]])
        );
        let split = evaluate("sphere.glsl", fixture, &["--batch-size", "257"]);
        assert_eq!(value["metrics"], split["metrics"]);
        assert_eq!(value["prediction_topology"], split["prediction_topology"]);
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn invalid_topology_is_not_a_match() {
    for (fixture, status) in [
        ("empty.glsl", "empty_occupancy"),
        ("full.glsl", "boundary_occupied"),
        ("nan.glsl", "non_finite_values"),
        ("invalid.glsl", "prediction_compile_failed"),
    ] {
        let value = evaluate("sphere.glsl", fixture, &[]);
        assert_eq!(value["status"], status, "{value}");
        assert!(value["metrics"].is_null());
    }
    let single = evaluate("sphere.glsl", "sphere.glsl", &["--resolution", "1"]);
    assert_eq!(single["status"], "boundary_occupied");
    assert_eq!(single["reference_topology"]["betti"], json!([1, 0, 0]));
}

#[test]
fn invalid_resolution_fails_before_input_read() {
    let value = evaluate("missing", "missing", &["--resolution", "0"]);
    assert_eq!(value["status"], "invalid_settings");
    assert!(value["metrics"].is_null());
}
