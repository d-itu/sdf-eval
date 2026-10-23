#!/usr/bin/env python3
"""Score an aligned GLSL SDF against a task-specific reference shader."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

IOU_RESOLUTION = 256
SURFACE_RESOLUTION = 128
TOPOLOGY_RESOLUTION = 128
SURFACE_SAMPLES = 100_000
SEED = 42
TIMEOUT = 300


class EvaluationError(Exception):
    """The reference, configuration, or evaluation environment is unusable."""


def bounds_arg(text):
    try:
        values = tuple(float(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected six numeric bounds") from error
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError("expected six finite bounds")
    if any(values[i] >= values[i + 3] for i in range(3)):
        raise argparse.ArgumentTypeError("upper bounds must exceed lower bounds")
    return values


def run_metric(binary, metric, gt, gen, bounds, threshold=None):
    command = [
        str(binary.resolve()), metric, "--reference", str(gt), "--prediction", str(gen),
        f"--bounds={','.join(map(str, bounds))}",
        "--resolution", str({"iou": IOU_RESOLUTION, "surface": SURFACE_RESOLUTION,
                             "topology": TOPOLOGY_RESOLUTION}[metric]),
    ]
    if metric == "surface":
        command += ["--samples", str(SURFACE_SAMPLES), "--seed", str(SEED),
                    "--thresholds", str(threshold)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvaluationError(f"{metric} evaluator could not run: {error}") from error
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationError(f"{metric} evaluator returned invalid JSON: {result.stderr.strip()}") from error
    if not isinstance(report, dict) or not isinstance(report.get("status"), str):
        raise EvaluationError(f"{metric} evaluator returned an invalid report")
    if (result.returncode == 0) != (report["status"] == "ok"):
        raise EvaluationError(f"{metric} evaluator exit code contradicts its status")
    return report


def reference_failure(report):
    status = report["status"]
    stage = report.get("stage", "")
    return (status.startswith("reference_") or
            (status == "no_surface_extracted" and stage == "reference_mesh") or
            (status == "non_finite_values" and stage.startswith("reference_")))


def score(gt, gen, binary, bounds):
    if not gt.is_file():
        raise EvaluationError(f"reference shader is missing: {gt}")
    if not binary.is_file():
        raise EvaluationError(f"sdf-eval executable is missing: {binary}")
    threshold = 0.02 * max(bounds[i + 3] - bounds[i] for i in range(3))
    settings = {
        "bounds": list(bounds), "iou_resolution": IOU_RESOLUTION,
        "surface_resolution": SURFACE_RESOLUTION, "topology_resolution": TOPOLOGY_RESOLUTION,
        "surface_samples": SURFACE_SAMPLES,
        "surface_seed": SEED, "f_score_threshold": threshold,
    }
    items = [
        {"id": "iou", "title": "Volumetric IoU", "max_points": 50, "value": None, "earned_points": 0.0},
        {"id": "surface_f_score", "title": "Surface F-score", "max_points": 30,
         "value": None, "earned_points": 0.0},
        {"id": "topology", "title": "Betti number similarity", "max_points": 20,
         "value": None, "earned_points": 0.0},
    ]
    report = {"status": "scored", "settings": settings, "items": items, "total_score": 0.0}
    if not gen.is_file():
        report["status"] = "invalid_submission"
        report["reason"] = "generated shader is missing"
        return report

    iou = run_metric(binary, "iou", gt, gen, bounds)
    if reference_failure(iou):
        raise EvaluationError(f"reference IoU failed: {iou['status']}")
    if iou["status"] == "prediction_compile_failed":
        report["status"] = "invalid_submission"
        report["reason"] = iou["status"]
        return report
    counts = iou.get("counts")
    if not isinstance(counts, dict):
        raise EvaluationError("IoU counts are missing")
    if counts.get("reference_non_finite", 0) != 0:
        raise EvaluationError("reference IoU contains non-finite values")
    if iou["status"] in ("ok", "empty_union", "non_finite_values"):
        if not isinstance(counts.get("reference_inside"), int) or counts["reference_inside"] <= 0:
            raise EvaluationError("reference has no occupied volume within bounds")
        if counts.get("reference_boundary_inside", 0) != 0:
            raise EvaluationError("reference occupies the sampling boundary")
    if iou["status"] == "non_finite_values" and counts.get("prediction_non_finite", 0) > 0:
        report["status"] = "invalid_submission"
        report["reason"] = "prediction contains non-finite values"
        return report
    if iou["status"] != "ok":
        raise EvaluationError(f"IoU evaluation failed: {iou['status']}")
    if counts.get("prediction_boundary_inside", 0) != 0:
        report["status"] = "invalid_submission"
        report["reason"] = "prediction occupies the sampling boundary"
        return report
    value = iou.get("iou")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise EvaluationError("IoU evaluator returned an invalid metric")
    items[0]["value"] = value
    items[0]["earned_points"] = 50 * value

    surface = run_metric(binary, "surface", gt, gen, bounds, threshold)
    if reference_failure(surface):
        raise EvaluationError(f"reference surface failed: {surface['status']}")
    if surface["status"] == "no_surface_extracted" and surface.get("stage") == "prediction_mesh":
        report["status"] = "partial"
        report["reason"] = "prediction has no extractable surface"
    elif surface["status"] in ("prediction_compile_failed", "non_finite_values") and surface.get("stage", "").startswith("prediction_"):
        report["status"] = "invalid_submission"
        report["reason"] = f"prediction surface evaluation failed: {surface['status']}"
        items[0]["earned_points"] = 0.0
        return report
    elif surface["status"] != "ok":
        raise EvaluationError(f"surface evaluation failed: {surface['status']}")
    else:
        for role in ("reference", "prediction"):
            mesh = surface.get(f"{role}_mesh")
            if not isinstance(mesh, dict) or not isinstance(mesh.get("boundary_occupied"), bool):
                raise EvaluationError(f"{role} boundary diagnostic is missing")
            if mesh["boundary_occupied"]:
                if role == "reference":
                    raise EvaluationError("reference surface reaches the sampling boundary")
                report["status"] = "invalid_submission"
                report["reason"] = "prediction surface reaches the sampling boundary"
                items[0]["earned_points"] = 0.0
                return report
        try:
            f_score = surface["metrics"]["f_scores"][0]["f_score"]
        except (KeyError, IndexError, TypeError) as error:
            raise EvaluationError("surface F-score is missing") from error
        if not isinstance(f_score, (int, float)) or not math.isfinite(f_score) or not 0 <= f_score <= 1:
            raise EvaluationError("surface evaluator returned an invalid F-score")
        items[1]["value"] = f_score
        items[1]["earned_points"] = 30 * f_score

    topology = run_metric(binary, "topology", gt, gen, bounds)
    if reference_failure(topology):
        raise EvaluationError(f"reference topology failed: {topology['status']}")
    reference = topology.get("reference_topology")
    if not isinstance(reference, dict) or not isinstance(reference.get("occupied_voxels"), int):
        raise EvaluationError("reference topology is missing")
    if reference["occupied_voxels"] == 0 or reference.get("boundary_occupied") is not False:
        raise EvaluationError("reference topology is empty or reaches the sampling boundary")
    if topology["status"] in ("prediction_compile_failed", "non_finite_values") and topology.get("stage", "").startswith("prediction_"):
        report["status"] = "invalid_submission"
        report["reason"] = f"prediction topology evaluation failed: {topology['status']}"
        for item in items:
            item["earned_points"] = 0.0
        return report
    prediction = topology.get("prediction_topology")
    if topology["status"] == "empty_occupancy" and isinstance(prediction, dict) and prediction.get("occupied_voxels") == 0:
        report["status"] = "partial"
        reason = "prediction has no occupied topology voxels"
        report["reason"] = f"{report['reason']}; {reason}" if "reason" in report else reason
    elif topology["status"] == "boundary_occupied" and isinstance(prediction, dict) and prediction.get("boundary_occupied") is True:
        report["status"] = "invalid_submission"
        report["reason"] = "prediction topology reaches the sampling boundary"
        for item in items:
            item["earned_points"] = 0.0
        return report
    elif topology["status"] != "ok":
        raise EvaluationError(f"topology evaluation failed: {topology['status']}")
    else:
        reference_betti = reference.get("betti")
        prediction_betti = prediction.get("betti") if isinstance(prediction, dict) else None
        if any(not isinstance(betti, list) or len(betti) != 3 or
               any(type(value) is not int or value < 0 for value in betti)
               for betti in (reference_betti, prediction_betti)):
            raise EvaluationError("topology Betti numbers are missing or invalid")
        differences = [abs(a - b) for a, b in zip(reference_betti, prediction_betti)]
        total_betti = sum(reference_betti) + sum(prediction_betti)
        items[2]["value"] = differences
        items[2]["reference_betti"] = reference_betti
        items[2]["prediction_betti"] = prediction_betti
        items[2]["earned_points"] = 20.0 * (1 - sum(differences) / total_betti)
    report["total_score"] = sum(item["earned_points"] for item in items)
    return report


def markdown(report):
    lines = ["# SDF Evaluation", "", f"Status: {report['status']}"]
    if "reason" in report:
        lines.append(f"Reason: {report['reason']}")
    lines += ["", "| Metric | Value | Points | Maximum |", "| --- | ---: | ---: | ---: |"]
    for item in report["items"]:
        value = "N/A" if item["value"] is None else str(item["value"])
        lines.append(f"| {item['title']} | {value} | {item['earned_points']} | {item['max_points']} |")
    lines += ["", f"Total score: {report['total_score']} / 100", "",
              "IoU and topology use occupied voxel centers; surface F-score uses independently sampled zero-isosurfaces.",
              "Neither shader is aligned or rescaled. See settings in score.json for sampling parameters.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Score an aligned generated GLSL SDF against a reference.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth GLSL SDF file from verification references.")
    parser.add_argument("--gen", type=Path, required=True, help="Generated GLSL SDF file from task outputs.")
    parser.add_argument("--json", type=Path, default=Path("score.json"), help="JSON score output path (default: score.json).")
    parser.add_argument("--md", type=Path, default=Path("report.md"), help="Markdown report output path (default: report.md).")
    parser.add_argument("--bounds", type=bounds_arg, required=True,
                        help="Task-specific xmin,ymin,zmin,xmax,ymax,zmax in shader coordinates.")
    parser.add_argument("--evaluator", type=Path, default=Path("./sdf-eval"),
                        help="Path to the sdf-eval binary (default: ./sdf-eval in the current directory).")
    args = parser.parse_args()
    if args.json.resolve() == args.md.resolve():
        parser.error("--json and --md must be different paths")
    try:
        result = score(args.gt, args.gen, args.evaluator, args.bounds)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        for path in (args.json, args.md):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
        args.md.write_text(markdown(result), encoding="utf-8")
    except (EvaluationError, OSError) as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
