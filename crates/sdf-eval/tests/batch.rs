//! Manifest validation and opt-in GPU parity tests.
use std::{
    fs,
    path::PathBuf,
    process::{self, Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/batch-tests")
            .join(format!(
                "{}-{}",
                process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn run(&self, manifest: &Value, extra: &[&str]) -> Output {
        let path = self.0.join("manifest.json");
        fs::write(&path, serde_json::to_vec(manifest).unwrap()).unwrap();
        Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
            .arg("batch")
            .arg("--input")
            .arg(path)
            .args(extra)
            .output()
            .unwrap()
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn job(id: &str, reference: &str, prediction: &str) -> Value {
    json!({"id": id, "reference": reference, "prediction": prediction, "bounds": [-2,-2,-2,2,2,2]})
}
fn manifest() -> Value {
    json!({"schema_version": 1, "defaults": {"resolution": 12, "batch_size": 257,
        "metrics": ["iou", "surface", "topology"],
        "surface": {"samples": 300, "seed": 42, "thresholds": [0.4,0.6]}},
        "jobs": [job("first", "reference.glsl", "prediction.glsl")]})
}
fn records(output: &Output) -> Vec<Value> {
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn invalid_manifests_do_not_truncate_output() {
    let work = Workspace::new();
    let output = work.0.join("results.jsonl");
    fs::write(&output, "keep").unwrap();
    let mut cases = Vec::new();
    let original = manifest();
    let mut bad = original.clone();
    bad["schema_version"] = json!(2);
    cases.push(bad);
    let mut bad = original.clone();
    bad["jobs"] = json!([]);
    cases.push(bad);
    let mut bad = original.clone();
    bad["jobs"][0]["resolution"] = json!(0);
    cases.push(bad);
    let mut bad = original.clone();
    bad["jobs"][0]["metrics"] = json!(["iou", "iou"]);
    cases.push(bad);
    let mut bad = original.clone();
    bad["jobs"][0]["surface"] = json!({"thresholds": []});
    cases.push(bad);
    let mut bad = original.clone();
    bad["jobs"] = json!([original["jobs"][0], original["jobs"][0]]);
    cases.push(bad);
    let mut bad = original.clone();
    bad["jobs"][0]["self_check"] = json!({"probes": 33});
    bad["jobs"][0]["metrics"] = json!(["self_check"]);
    cases.push(bad);
    let mut bad = original.clone();
    bad["jobs"][0].as_object_mut().unwrap().remove("reference");
    cases.push(bad);
    let mut bad = original;
    bad["jobs"][0]["unknown"] = json!(true);
    cases.push(bad);
    for bad in cases {
        let result = work.run(&bad, &["--output", output.to_str().unwrap()]);
        assert!(!result.status.success());
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep");
    }
}

#[test]
fn missing_inputs_produce_each_metric_record_and_continue() {
    let work = Workspace::new();
    fs::write(
        work.0.join("reference.glsl"),
        "float sdf(vec3 p) { return length(p)-1.0; }",
    )
    .unwrap();
    let mut input = manifest();
    input["jobs"] = json!([
        job("missing-reference", "absent.glsl", "absent.glsl"),
        job("missing-prediction", "reference.glsl", "absent.glsl")
    ]);
    let result = work.run(&input, &[]);
    assert!(!result.status.success());
    let rows = records(&result);
    assert_eq!(rows.len(), 6);
    for row in rows {
        let expected = if row["id"] == "missing-reference" {
            "reference_read_failed"
        } else {
            "prediction_read_failed"
        };
        assert_eq!(row["status"], expected, "{row}");
        assert_eq!(row["reference_cache_hit"], false);
        assert!(row["elapsed_seconds"].is_number());
    }
}

#[test]
fn output_must_not_overwrite_inputs() {
    let work = Workspace::new();
    let shader = work.0.join("reference.glsl");
    fs::write(&shader, "keep shader").unwrap();
    let result = work.run(&manifest(), &["--output", shader.to_str().unwrap()]);
    assert!(!result.status.success());
    assert_eq!(fs::read_to_string(shader).unwrap(), "keep shader");
    #[cfg(unix)]
    {
        let alias = work.0.join("alias.jsonl");
        fs::hard_link(work.0.join("reference.glsl"), &alias).unwrap();
        let result = work.run(&manifest(), &["--output", alias.to_str().unwrap()]);
        assert!(!result.status.success());
        assert_eq!(fs::read_to_string(alias).unwrap(), "keep shader");
    }
    let input = work.0.join("manifest.json");
    let result = work.run(&manifest(), &["--output", input.to_str().unwrap()]);
    assert!(!result.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(input).unwrap()).unwrap(),
        manifest()
    );
}

#[test]
#[ignore = "requires a compute adapter"]
fn self_check_batch_supports_single_input_and_mixed_metrics() {
    let work = Workspace::new();
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    fs::copy(fixtures.join("sphere.glsl"), work.0.join("sphere.glsl")).unwrap();
    fs::copy(
        fixtures.join("overscaled_sphere.glsl"),
        work.0.join("high.glsl"),
    )
    .unwrap();
    fs::write(work.0.join("bad.glsl"), "invalid GLSL").unwrap();
    let input = json!({"schema_version": 1,
    "defaults": {"metrics": ["self_check"],
        "self_check": {"probes": 8, "rays": 12, "oracle_steps": 512}},
    "jobs": [
        {"id": "single", "prediction": "sphere.glsl", "bounds": [-2,-2,-2,2,2,2]},
        {"id": "high", "prediction": "high.glsl", "bounds": [-2,-2,-2,2,2,2],
            "self_check": {"probes": 8, "rays": 12, "oracle_steps": 512}},
        {"id": "mixed", "reference": "sphere.glsl", "prediction": "sphere.glsl",
            "bounds": [-2,-2,-2,2,2,2], "metrics": ["iou", "self_check"]},
        {"id": "bad", "prediction": "bad.glsl", "bounds": [-2,-2,-2,2,2,2]},
        {"id": "missing", "prediction": "absent.glsl", "bounds": [-2,-2,-2,2,2,2]}
    ]});
    let output = work.run(&input, &[]);
    assert!(!output.status.success());
    let rows = records(&output);
    assert_eq!(rows.len(), 6, "{}", String::from_utf8_lossy(&output.stderr));
    let get = |id: &str, metric: &str| {
        rows.iter()
            .find(|r| r["id"] == id && r["metric"] == metric)
            .unwrap()
    };
    let single = get("single", "self_check");
    assert_eq!(single["status"], "ok", "{single}");
    assert!(single["metrics"]["field"].get("probes").is_none());
    assert!(single["metrics"]["rays"].get("total").is_none());
    assert_eq!(single["input"]["field"]["probes_per_axis"], 8);
    assert_eq!(single["input"]["rays"]["rays_per_face_side"], 12);
    assert_eq!(single["input"]["rays"]["oracle_steps"], 512);
    assert_eq!(
        single["input"]["bounds"],
        json!([-2.0, -2.0, -2.0, 2.0, 2.0, 2.0])
    );
    assert!(single["input"]["epsilon"].as_f64().unwrap() > 0.0);
    assert!(single.get("resolution").is_none());
    assert!(single.get("requested_batch_size").is_none());
    assert!(single.get("probes_per_axis").is_none());
    assert!(single.get("reference").is_none());
    assert!(
        get("high", "self_check")["metrics"]["field"]["overshoot_fraction"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert_eq!(get("mixed", "iou")["status"], "ok");
    assert_eq!(get("mixed", "self_check")["metrics"], single["metrics"]);
    assert_eq!(get("bad", "self_check")["status"], "shader_compile_failed");
    assert_eq!(
        get("missing", "self_check")["status"],
        "prediction_read_failed"
    );
    let direct = Command::new(env!("CARGO_BIN_EXE_sdf-eval"))
        .arg("self-check")
        .arg("--shader")
        .arg(work.0.join("sphere.glsl"))
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
    assert!(direct.status.success());
    let direct: Value = serde_json::from_slice(&direct.stdout).unwrap();
    assert_eq!(single["metrics"], direct["metrics"]);
    assert_eq!(single["input"], direct["input"]);
}

#[test]
#[ignore = "requires a compute adapter"]
fn gpu_batch_matches_single_commands_and_recovers_from_bad_prediction() {
    let work = Workspace::new();
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    fs::copy(fixtures.join("sphere.glsl"), work.0.join("reference.glsl")).unwrap();
    fs::copy(
        fixtures.join("sphere.glsl"),
        work.0.join("reference-copy.glsl"),
    )
    .unwrap();
    fs::copy(
        fixtures.join("small_sphere.glsl"),
        work.0.join("prediction.glsl"),
    )
    .unwrap();
    fs::write(work.0.join("bad.glsl"), "invalid GLSL").unwrap();
    let mut input = manifest();
    input["jobs"].as_array_mut().unwrap().extend([
        job("bad", "reference.glsl", "bad.glsl"),
        job("cached", "reference-copy.glsl", "prediction.glsl"),
    ]);
    let mut thresholds = job("thresholds", "reference.glsl", "prediction.glsl");
    thresholds["surface"] = json!({"thresholds": [0.7]});
    input["jobs"].as_array_mut().unwrap().push(thresholds);
    let mut different = job("different-grid", "reference.glsl", "prediction.glsl");
    different["resolution"] = json!(14);
    input["jobs"].as_array_mut().unwrap().push(different);
    let result = work.run(&input, &[]);
    assert!(!result.status.success());
    let rows = records(&result);
    assert_eq!(
        rows.len(),
        15,
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for metric in ["iou", "surface", "topology"] {
        let get = |id| {
            rows.iter()
                .find(|r| r["id"] == id && r["metric"] == metric)
                .unwrap()
        };
        let first = get("first");
        let cached = get("cached");
        assert_eq!(first["status"], "ok", "{first}");
        assert_eq!(cached["status"], "ok", "{cached}");
        assert_eq!(first["reference_cache_hit"], false);
        assert_eq!(cached["reference_cache_hit"], true);
        assert_eq!(get("different-grid")["reference_cache_hit"], false);
        assert_eq!(get("bad")["status"], "prediction_compile_failed");
        let mut command = Command::new(env!("CARGO_BIN_EXE_sdf-eval"));
        command
            .arg(metric)
            .arg("--reference")
            .arg(work.0.join("reference.glsl"))
            .arg("--prediction")
            .arg(work.0.join("prediction.glsl"))
            .args([
                "--bounds=-2,-2,-2,2,2,2",
                "--resolution",
                "12",
                "--batch-size",
                "257",
            ]);
        if metric == "surface" {
            command.args([
                "--samples",
                "300",
                "--seed",
                "42",
                "--thresholds",
                "0.4,0.6",
            ]);
        }
        let single = command.output().unwrap();
        assert!(
            single.status.success(),
            "{}",
            String::from_utf8_lossy(&single.stderr)
        );
        let single: Value = serde_json::from_slice(&single.stdout).unwrap();
        let fields: &[&str] = match metric {
            "iou" => &[
                "iou",
                "counts",
                "boundary_occupied",
                "volume_samples",
                "boundary_samples",
            ],
            "surface" => &[
                "metrics",
                "reference_mesh",
                "prediction_mesh",
                "boundary_occupied",
            ],
            _ => &[
                "metrics",
                "reference_topology",
                "prediction_topology",
                "boundary_occupied",
            ],
        };
        for &field in fields {
            assert_eq!(first[field], single[field], "{metric}: {field}");
            assert_eq!(cached[field], single[field], "cached {metric}: {field}");
        }
    }
    let changed = rows
        .iter()
        .find(|r| r["id"] == "thresholds" && r["metric"] == "surface")
        .unwrap();
    assert_eq!(changed["reference_cache_hit"], true);
    assert_eq!(changed["thresholds"], json!([0.7]));
    assert_eq!(changed["metrics"]["f_scores"].as_array().unwrap().len(), 1);
}
