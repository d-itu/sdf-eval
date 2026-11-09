#!/usr/bin/env python3
"""Render dataset shaders through the pinned Rust sdf-view executable."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def render(folder, timeout):
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    shader = folder / "sdf.glsl"
    source = shader.read_text()
    notices = re.findall(r"/\*.*?\*/", source, re.S)
    metadata["source"]["notices"] = [
        notice for notice in notices if re.search(r"copyright|license", notice, re.I)
    ]
    metadata["shader_sha256"] = hashlib.sha256(shader.read_bytes()).hexdigest()
    settings = metadata.setdefault("render", {
        "width": 512,
        "height": 512,
        "camera_position": [2.4, 1.6, 3.2],
        "camera_target": [0, 0, 0],
        "camera_up": [0, 1, 0],
        "fov": 45,
        "light_direction": [-0.5, 0.8, 1],
        "light_color": [1, 1, 1],
        "light_intensity": 0.85,
        "ambient": 0.15,
    })
    metadata["renderer"] = {
        "repository": "https://github.com/d-itu/sdf-view",
        "commit": "51ff7f4f9e026d852520263374056a6af0740bbc",
        "profile": "dev",
    }
    command = [str(ROOT / "target/debug/sdf-view"), str(shader), "-o", str(folder / "render.png")]
    for key, value in settings.items():
        command.extend(["--" + key.replace("_", "-"),
                        ",".join(map(str, value)) if isinstance(value, list) else str(value)])
    started = time.monotonic()
    env = dict(os.environ)
    env["MESA_SHADER_CACHE_DIR"] = str(ROOT / ".cache/mesa")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode:
            metadata["status"] = "render_failed"
            metadata["error"] = result.stderr.strip()
        else:
            image = Image.open(folder / "render.png").convert("RGBA")
            alpha = image.getchannel("A")
            bounds = alpha.getbbox()
            coverage = sum(value > 0 for value in alpha.get_flattened_data()) / (image.width * image.height)
            metadata["image_check"] = {
                "foreground_bounds": bounds,
                "foreground_fraction": coverage,
                "touches_border": bool(bounds and (bounds[0] == 0 or bounds[1] == 0
                                        or bounds[2] == image.width or bounds[3] == image.height)),
            }
            metadata["status"] = "rendered" if bounds else "empty_image"
            metadata.pop("error", None)
    except subprocess.TimeoutExpired:
        metadata["status"] = "render_timeout"
        metadata["error"] = f"Rendering exceeded {timeout} seconds."
    metadata["visual_review"] = "pending"
    metadata["render_seconds"] = round(time.monotonic() - started, 3)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{folder.name}: {metadata['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    folders = [ROOT / "dataset" / name for name in args.models] if args.models else sorted((ROOT / "dataset").iterdir())
    for folder in folders:
        if (folder / "sdf.glsl").is_file():
            render(folder, args.timeout)


if __name__ == "__main__":
    main()
