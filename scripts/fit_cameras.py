#!/usr/bin/env python3
"""Fit camera framing to the alpha bounds without changing shader geometry."""

import argparse
import json
import math
from pathlib import Path

from render_dataset import ROOT, render


def normalized(vector):
    length = math.sqrt(sum(v * v for v in vector))
    return [v / length for v in vector]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def fit(folder):
    path = folder / "metadata.json"
    for attempt in range(5):
        metadata = json.loads(path.read_text())
        if metadata["status"] != "rendered":
            return
        check = metadata["image_check"]
        bounds = check["foreground_bounds"]
        settings = metadata["render"]
        width, height = settings["width"], settings["height"]
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
        centered = abs(center[0] - width / 2) < 12 and abs(center[1] - height / 2) < 12
        if not check["touches_border"] and 330 <= span <= 395 and centered:
            return
        eye, target = settings["camera_position"], settings["camera_target"]
        offset = [a - b for a, b in zip(eye, target)]
        distance = math.sqrt(sum(v * v for v in offset))
        factor = 1.6 if check["touches_border"] else max(0.65, min(1.4, span / 365))
        shift = [0, 0, 0]
        if not check["touches_border"]:
            forward = normalized([-v for v in offset])
            right = normalized(cross(forward, settings["camera_up"]))
            up = cross(right, forward)
            pixel_scale = 2 * distance * math.tan(math.radians(settings["fov"] / 2)) / height
            shift = [pixel_scale * ((center[0] - width / 2) * r - (center[1] - height / 2) * u)
                     for r, u in zip(right, up)]
        settings["camera_target"] = [round(t + s, 6) for t, s in zip(target, shift)]
        settings["camera_position"] = [round(t + s + o * factor, 6) for t, s, o in zip(target, shift, offset)]
        metadata["framing"] = "Camera fitted to foreground bounds; shader coordinates unchanged."
        path.write_text(json.dumps(metadata, indent=2) + "\n")
        render(folder, 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*")
    args = parser.parse_args()
    folders = [ROOT / "dataset" / name for name in args.models] if args.models else sorted((ROOT / "dataset").iterdir())
    for folder in folders:
        if (folder / "metadata.json").exists():
            fit(folder)
