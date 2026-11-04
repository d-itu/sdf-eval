#!/usr/bin/env python3
"""Score an aligned GLSL SDF against a task-specific reference shader."""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

from runtime import bootstrap

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
    if iou["status"] not in ("ok", "empty_union", "non_finite_values"):
        raise EvaluationError(f"IoU evaluation failed: {iou['status']}: {iou.get('error', '')}")
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


def helper(name):
    root = Path(__file__).resolve().parent
    path = root / f"{name}.py"
    if not path.exists():
        path = root.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_images(args, report):
    config = json.loads(args.image_config.read_text())
    allowed = {"distance", "target", "azimuths", "elevations", "top", "bottom",
               "fov", "width", "height", "antialiasing"}
    if not isinstance(config, dict) or not config.keys() <= allowed or "distance" not in config:
        raise EvaluationError("image config must contain distance and only documented camera/render options")
    # Reuse the renderer's CLI validation before starting any render.
    render_script = Path(helper("render_views").__file__)
    root = args.artifacts.resolve()
    if root.exists():
        raise EvaluationError(f"artifact directory already exists: {root}")
    for role, shader in (("reference", args.gt), ("prediction", args.gen)):
        if role == "prediction" and report["status"] == "invalid_submission":
            report["image_metrics"] = {"status": "invalid_submission", "reason": report["reason"]}
            return
        command = [sys.executable, str(render_script), str(shader.resolve()),
                   "--output", str(root / role), "--renderer", str(args.renderer.resolve())]
        options = {"azimuths": 8, "elevations": [-30, 0, 30], "top": True, "bottom": True,
                   "antialiasing": 4} | config
        for key, value in options.items():
            if key in ("top", "bottom"):
                if type(value) is not bool:
                    raise EvaluationError(f"{key} must be boolean")
                if value:
                    command.append(f"--{key}")
            else:
                encoded = ",".join(map(str, value)) if isinstance(value, list) else str(value)
                command.append(f"--{key.replace('_', '-')}={encoded}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        if result.returncode:
            # Renderer errors cannot reliably distinguish a driver fault from a shader fault.
            raise EvaluationError(f"{role} rendering failed: {result.stderr.strip()}")
    images = helper("evaluate_images")
    report["image_metrics"] = images.evaluate(argparse.Namespace(
        reference=root / "reference/views.json", prediction=root / "prediction/views.json",
        device="cpu", cache_dir=args.weights,
    ))


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Score an aligned generated GLSL SDF against a reference.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference directory containing sdf.glsl.")
    parser.add_argument("--output", type=Path, required=True, help="Prediction directory containing sdf.glsl.")
    parser.add_argument("--score", type=Path, required=True, help="JSON score output file.")
    parser.add_argument("--bounds", type=bounds_arg, default=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
                        help="Optional xmin,ymin,zmin,xmax,ymax,zmax (default: -2,-2,-2,2,2,2).")
    parser.add_argument("--evaluator", type=Path, default=package / "bin/sdf-eval",
                        help="Path to sdf-eval (default: bin/sdf-eval beside this entry point).")
    parser.add_argument("--image-config", type=Path, help="Fixed reference-derived image camera configuration JSON; enables SSIM/LPIPS.")
    parser.add_argument("--renderer", type=Path, default=package / "bin/sdf-view", help="Path to sdf-view.")
    parser.add_argument("--artifacts", type=Path, default=Path("image-artifacts"), help="New directory for rendered evaluation images.")
    parser.add_argument("--weights", type=Path, default=package / "weights", help="Torch weight cache containing checkpoints/.")
    args = parser.parse_args()
    if not args.reference.is_dir():
        parser.error("--reference must be a directory")
    if not args.output.is_dir():
        parser.error("--output must be a directory")
    gt = args.reference / "sdf.glsl"
    gen = args.output / "sdf.glsl"
    if args.score.resolve() in (gt.resolve(), gen.resolve()):
        parser.error("--score must not overwrite a shader")
    if args.image_config:
        bootstrap()
    try:
        result = score(gt, gen, args.evaluator, args.bounds)
        if args.image_config:
            args.gt = gt
            args.gen = gen
            score_images(args, result)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.score.parent.mkdir(parents=True, exist_ok=True)
        args.score.write_text(text, encoding="utf-8")
    except (EvaluationError, OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
