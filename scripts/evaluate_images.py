#!/usr/bin/env python3
"""Evaluate matched multi-view renders using RGB SSIM and AlexNet LPIPS."""

import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile


RENDER_FIELDS = {
    "width", "height", "antialiasing", "background", "light_direction",
    "light_color", "object_color", "light_intensity", "ambient",
}
CAMERA_FIELDS = ("camera_position", "camera_target", "camera_up", "fov")


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def load_manifest(path):
    path = path.resolve()
    data = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(data, dict) or type(data.get("schema_version", 1)) is not int or data.get("schema_version", 1) != 1:
        raise ValueError(f"{path}: expected manifest schema version 1")
    settings = data.get("render")
    if not isinstance(settings, dict) or not RENDER_FIELDS <= settings.keys():
        raise ValueError(f"{path}: missing required render settings")
    for key in ("width", "height"):
        if type(settings[key]) is not int or settings[key] < 64:
            raise ValueError(f"{path}: {key} must be an integer >= 64 for AlexNet LPIPS")
    if settings["background"] != "transparent":
        raise ValueError(f"{path}: expected transparent render background")
    if type(settings["antialiasing"]) is not int or settings["antialiasing"] not in (1, 4):
        raise ValueError(f"{path}: antialiasing must be 1 or 4")
    for key in ("light_intensity", "ambient"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{path}: invalid {key}")
    light = settings["light_direction"]
    if not isinstance(light, list) or len(light) != 3 or not all(
        type(x) in (int, float) and math.isfinite(x) for x in light
    ) or not any(light):
        raise ValueError(f"{path}: invalid light_direction")
    for key in ("light_color", "object_color"):
        if not isinstance(settings[key], str) or not settings[key]:
            raise ValueError(f"{path}: invalid {key}")
    views = data.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"{path}: views must be a nonempty list")
    indexed = {}
    for view in views:
        if not isinstance(view, dict):
            raise ValueError(f"{path}: each view must be an object")
        name = view.get("image")
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"{path}: image must be a relative path without '..'")
        image = (path.parent / name).resolve()
        if not image.is_relative_to(path.parent) or name in indexed:
            raise ValueError(f"{path}: duplicate or escaping image path: {name}")
        for key in CAMERA_FIELDS:
            value = view.get(key)
            values = [value] if key == "fov" else value
            if not isinstance(values, list) or len(values) != (1 if key == "fov" else 3) or not all(
                type(x) in (int, float) and math.isfinite(x) for x in values
            ):
                raise ValueError(f"{path}: invalid {key} in {name}")
        if not 0 < view["fov"] < 180 or view["camera_position"] == view["camera_target"] or not any(view["camera_up"]):
            raise ValueError(f"{path}: degenerate camera in {name}")
        direction = [target - position for target, position in zip(view["camera_target"], view["camera_position"])]
        up = view["camera_up"]
        cross = [direction[1] * up[2] - direction[2] * up[1],
                 direction[2] * up[0] - direction[0] * up[2],
                 direction[0] * up[1] - direction[1] * up[0]]
        if not all(math.isfinite(x) for x in cross) or not any(cross):
            raise ValueError(f"{path}: camera_up is parallel to viewing direction in {name}")
        indexed[name] = view
    return data, indexed


def matched_views(reference, prediction):
    ref, ref_views = load_manifest(reference)
    pred, pred_views = load_manifest(prediction)
    if ref["render"] != pred["render"]:
        raise ValueError("reference and prediction render settings differ")
    if ref_views.keys() != pred_views.keys():
        raise ValueError("reference and prediction image sets differ")
    for name in ref_views:
        if any(ref_views[name][key] != pred_views[name][key] for key in CAMERA_FIELDS):
            raise ValueError(f"camera settings differ: {name}")
    return ref, list(ref_views)


def load_rgb(path, settings):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGBA":
            raise ValueError(f"{path}: expected an RGBA PNG")
        if image.size != (settings["width"], settings["height"]):
            raise ValueError(f"{path}: dimensions differ from manifest")
        rgba = np.asarray(image, dtype=np.float32) / 255.0
    return rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4])


def evaluate(args):
    manifest, names = matched_views(args.reference, args.prediction)
    # Validate every input before loading weights or computing partial results.
    for name in names:
        for path in (args.reference, args.prediction):
            load_rgb(path.resolve().parent / name, manifest["render"])

    import numpy as np
    import torch
    import lpips
    from skimage.metrics import structural_similarity

    torch.hub.set_dir(str(args.cache_dir.resolve()))
    device = torch.device(args.device)
    model = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()
    results = []
    for name in names:
        reference = load_rgb(args.reference.resolve().parent / name, manifest["render"])
        prediction = load_rgb(args.prediction.resolve().parent / name, manifest["render"])
        ssim = float(structural_similarity(
            reference, prediction, channel_axis=-1, data_range=1.0,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        ))
        tensors = [torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
                   for image in (reference, prediction)]
        with torch.inference_mode():
            distance = float(model(*tensors).item())
        if not math.isfinite(ssim) or not math.isfinite(distance):
            raise ValueError(f"non-finite metric: {name}")
        results.append({"image": name, "ssim": ssim, "lpips": distance})
    return {
        "schema_version": 1,
        "status": "ok",
        "reference": str(args.reference.resolve()),
        "prediction": str(args.prediction.resolve()),
        "metrics": {f"{key}_mean": float(np.mean([view[key] for view in results]))
                    for key in ("ssim", "lpips")},
        "settings": {
            "view_count": len(names), "render": manifest["render"],
            "views": manifest["views"], "background_rgb": [1, 1, 1],
            "color_space": "sRGB values; no linearization",
            "ssim": {"data_range": 1.0, "gaussian_weights": True, "sigma": 1.5,
                     "use_sample_covariance": False, "channel_axis": -1},
            "lpips": {"net": "alex", "version": "0.1", "input_range": [-1, 1]},
            "device": str(device),
            "packages": {name: version(name) for name in ("numpy", "Pillow", "torch", "torchvision", "scikit-image", "lpips")},
        },
        "views": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="reference views.json")
    parser.add_argument("--prediction", type=Path, required=True, help="prediction views.json")
    parser.add_argument("--output", type=Path, required=True, help="new output JSON file")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu; e.g. cuda)")
    parser.add_argument("--cache-dir", type=Path, default=Path("results/.cache/torch"), help="project-local pretrained weight cache")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        result = evaluate(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump(result, file, indent=2, allow_nan=False)
                file.write("\n")
            # Exclusive publication prevents overwriting an existing report.
            os.link(temporary, args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result["metrics"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
