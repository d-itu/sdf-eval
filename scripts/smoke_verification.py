#!/usr/bin/env python3
"""Exercise a standalone bundle or development entry point with real shaders."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "crates/sdf-eval/tests/fixtures"
    (root / ".work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / ".work", prefix="smoke-") as directory:
        work = Path(directory)
        config = work / "images.json"
        config.write_text(json.dumps({"distance": 4, "azimuths": 1, "elevations": [0],
                                     "top": False, "bottom": False, "width": 64,
                                     "height": 64, "antialiasing": 1}))
        reference = work / "reference"
        reference.mkdir()
        (reference / "sdf.glsl").write_bytes((fixtures / "sphere.glsl").read_bytes())
        reports = {}
        for case, shader in (("identical", "sphere.glsl"), ("changed", "small_sphere.glsl"),
                             ("invalid", "invalid.glsl")):
            output = work / case / "output"
            output.mkdir(parents=True)
            (output / "sdf.glsl").write_bytes((fixtures / shader).read_bytes())
            command = [sys.executable, str(args.entry.resolve()), "--reference", str(reference),
                       "--output", str(output),
                       "--image-config", str(config), "--artifacts", str(work / case / "images"),
                       "--report", str(work / f"{case}.details.json"),
                       "--score", str(work / f"{case}.json")]
            for key in ("evaluator", "renderer", "weights"):
                value = getattr(args, key)
                if value:
                    command += [f"--{key}", str(value.resolve())]
            subprocess.run(command, cwd=work, check=True, timeout=1800)
            result = json.loads((work / f"{case}.json").read_text())
            reports[case] = json.loads((work / f"{case}.details.json").read_text())
            items = result["Physical, Geometric, and Structural Capability"]["Geometric Form and Topological Structure"]
            total = sum(item["score"] * item["weight"] for item in items)
            if abs(total - result["total_score"]) > 1e-8 or result["total_score"] != reports[case]["total_score"]:
                raise RuntimeError("production scores disagree with the diagnostic report")
        same = reports["identical"]["image_metrics"]["metrics"]
        changed = reports["changed"]["image_metrics"]["metrics"]
        if not (same["ssim_mean"] > 0.999 and abs(same["lpips_mean"]) < 1e-5):
            raise RuntimeError("identical shader image metrics are not near perfect")
        if not (changed["ssim_mean"] < same["ssim_mean"] and changed["lpips_mean"] > same["lpips_mean"]):
            raise RuntimeError("changed shader did not reduce image similarity")
        invalid = reports["invalid"]
        if invalid["status"] != "invalid_submission" or invalid["total_score"] != 0:
            raise RuntimeError("invalid shader did not receive invalid-submission status")
        if invalid["image_metrics"]["status"] != "invalid_submission":
            raise RuntimeError("invalid shader did not propagate to image status")
        print(json.dumps({case: {"score": report["total_score"],
                                "image_status": report["image_metrics"]["status"],
                                "metrics": report["image_metrics"].get("metrics")}
                          for case, report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
