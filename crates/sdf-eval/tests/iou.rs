//! End-to-end tests requiring a working compute adapter (hardware or software).

use serde_json::Value;
use std::{path::Path, process::Command};

fn evaluate(reference: &str, prediction: &str, bounds: &str, n: u32, batch: u32) -> Value {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
        .arg("iou")
        .arg("--reference")
        .arg(fixtures.join(reference))
        .arg("--prediction")
        .arg(fixtures.join(prediction))
        .arg(format!("--bounds={bounds}"))
        .arg("--resolution")
        .arg(n.to_string())
        .arg("--batch-size")
        .arg(batch.to_string())
        .output()
        .unwrap();
    let result: Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&output.stderr)));
    assert_eq!(
        output.status.success(),
        result["status"] == "ok",
        "{result}"
    );
    result
}

#[test]
#[ignore = "requires a compute adapter; run cargo test -- --include-ignored"]
fn gpu_iou_contract() {
    let bounds = "-2,-2,-2,2,2,2";
    let same = evaluate("sphere.glsl", "sphere.glsl", bounds, 32, 1_048_576);
    assert_eq!(same["iou"], 1.0, "{same}");
    assert_eq!(same["boundary_occupied"], false);
    let split = evaluate("sphere.glsl", "sphere.glsl", bounds, 32, 257);
    assert_eq!(same["counts"], split["counts"]);
    let disjoint = evaluate("sphere.glsl", "disjoint_sphere.glsl", bounds, 64, 65536);
    assert_eq!(disjoint["iou"], 0.0, "{disjoint}");
    let nested = evaluate("sphere.glsl", "small_sphere.glsl", bounds, 64, 65536);
    assert!(
        (nested["iou"].as_f64().unwrap() - 0.125).abs() < 0.005,
        "{nested}"
    );
    let cubes = evaluate("cube.glsl", "shifted_cube.glsl", bounds, 64, 65536);
    assert_eq!(cubes["iou"], 0.5, "{cubes}");
    let swapped = evaluate("shifted_cube.glsl", "cube.glsl", bounds, 64, 65536);
    assert_eq!(cubes["iou"], swapped["iou"]);

    let empty = evaluate("empty.glsl", "empty.glsl", bounds, 8, 300);
    assert_eq!(empty["status"], "empty_union");
    assert!(empty["iou"].is_null());
    let one_empty = evaluate("sphere.glsl", "empty.glsl", bounds, 16, 257);
    assert_eq!(one_empty["iou"], 0.0);
    let full = evaluate("full.glsl", "full.glsl", bounds, 8, 257);
    assert_eq!(full["iou"], 1.0);
    assert_eq!(full["counts"]["intersection"], 512);
    assert_eq!(full["counts"]["reference_boundary_inside"], 384);
    assert_eq!(full["counts"]["prediction_boundary_inside"], 384);
    assert_eq!(full["boundary_occupied"], true);
    let invalid = evaluate("infinity.glsl", "nan.glsl", bounds, 8, 257);
    assert_eq!(invalid["status"], "non_finite_values", "{invalid}");
    assert_eq!(invalid["counts"]["reference_non_finite"], 896);
    assert_eq!(invalid["counts"]["prediction_non_finite"], 896);
    assert!(invalid["iou"].is_null());
    let bad = evaluate("sphere.glsl", "invalid.glsl", bounds, 8, 257);
    assert_eq!(bad["status"], "prediction_compile_failed", "{bad}");
    let bad = evaluate("invalid.glsl", "sphere.glsl", bounds, 8, 257);
    assert_eq!(bad["status"], "reference_compile_failed", "{bad}");
}

#[test]
#[ignore = "requires a compute adapter; run cargo test -- --include-ignored"]
fn gpu_counts_match_cpu_in_asymmetric_box() {
    // Independently enumerate volume centers and face centers on the CPU.
    let lower = [-2.0f32, -1.25, -1.75];
    let upper = [2.5f32, 1.75, 2.25];
    let n = 17u32;
    let center = |axis: usize, i: u32| {
        lower[axis] + (i as f32 + 0.5) / n as f32 * (upper[axis] - lower[axis])
    };
    let inside = |p: [f32; 3], shift: f32| {
        (p[0] - shift).abs() < 0.75 && p[1].abs() < 0.75 && p[2].abs() < 0.75
    };
    let mut intersection = 0;
    let mut union = 0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = [center(0, x), center(1, y), center(2, z)];
                let a = inside(p, 0.0);
                let b = inside(p, 0.5);
                intersection += u64::from(a && b);
                union += u64::from(a || b);
            }
        }
    }
    let gpu = evaluate(
        "cube.glsl",
        "shifted_cube.glsl",
        "-2,-1.25,-1.75,2.5,1.75,2.25",
        n,
        509,
    );
    assert_eq!(gpu["counts"]["intersection"], intersection);
    assert_eq!(gpu["counts"]["union"], union);
    assert_eq!(gpu["volume_samples"], n.pow(3));
    assert_eq!(gpu["boundary_samples"], 6 * n * n);
    assert_eq!(gpu["boundary_occupied"], false);
    // All six faces cut through the cube; count each face independently.
    let clipped = evaluate(
        "cube.glsl",
        "shifted_cube.glsl",
        "-0.5,-0.5,-0.5,0.5,0.5,0.5",
        n,
        509,
    );
    let mut a_boundary = 0u64;
    let mut b_boundary = 0u64;
    for axis in 0..3 {
        for side in [-0.5, 0.5] {
            for i in 0..n {
                for j in 0..n {
                    let mut p = [0.0; 3];
                    p[axis] = side;
                    p[(axis + 1) % 3] = -0.5 + (i as f32 + 0.5) / n as f32;
                    p[(axis + 2) % 3] = -0.5 + (j as f32 + 0.5) / n as f32;
                    a_boundary += u64::from(inside(p, 0.0));
                    b_boundary += u64::from(inside(p, 0.5));
                }
            }
        }
    }
    assert_eq!(clipped["counts"]["reference_boundary_inside"], a_boundary);
    assert_eq!(clipped["counts"]["prediction_boundary_inside"], b_boundary);
}

#[test]
fn invalid_settings_report_without_gpu() {
    for (bounds, n, batch) in [
        ("1,1,1,-1,-1,-1", 16, 256),
        ("-1,-1,-1,1,1,1", 0, 256),
        ("-1,-1,-1,1,1,1", 16, 0),
        ("-1,-1,-1,1,1,1", u32::MAX, 256),
    ] {
        let result = evaluate("sphere.glsl", "sphere.glsl", bounds, n, batch);
        assert_eq!(result["status"], "invalid_settings");
        assert!(result["iou"].is_null());
    }
}
