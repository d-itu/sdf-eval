#!/usr/bin/env python3
"""Render an SDF from orbit cameras and optional top and bottom cameras."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def target_vector(value):
    try:
        vector = [float(component) for component in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected X,Y,Z") from error
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise argparse.ArgumentTypeError("expected three finite coordinates: X,Y,Z")
    return vector


def elevation_list(value):
    try:
        elevations = [int(angle) for angle in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integer degrees") from error
    if len(set(elevations)) != len(elevations) or not all(-90 < angle < 90 for angle in elevations):
        raise argparse.ArgumentTypeError("use distinct angles between -90 and 90; use --top/--bottom for poles")
    return elevations


def azimuth_label(index, count):
    # Integer division avoids rounding adjacent camera names to the same angle.
    whole, remainder = divmod(index * 360, count)
    if remainder:
        return f"{whole:03d}p{remainder}of{count}"
    return f"{whole:03d}"


def cameras(args):
    for index in range(args.azimuths):
        azimuth = index * 360 / args.azimuths
        az = math.radians(azimuth)
        for elevation in args.elevations:
            el = math.radians(elevation)
            offset = [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)]
            yield {
                "image": f"az{azimuth_label(index, args.azimuths)}_el{elevation:+03d}.png",
                "azimuth": azimuth,
                "elevation": elevation,
                "camera_position": [target + args.distance * component for target, component in zip(args.target, offset)],
                "camera_target": args.target,
                "camera_up": [0, 1, 0],
                "fov": args.fov,
            }
    for name, enabled, sign in [("top", args.top, 1), ("bottom", args.bottom, -1)]:
        if enabled:
            yield {
                "image": f"{name}.png",
                "azimuth": None,
                "elevation": sign * 90,
                "camera_position": [args.target[0], args.target[1] + sign * args.distance, args.target[2]],
                "camera_target": args.target,
                "camera_up": [0, 0, -sign],
                "fov": args.fov,
            }


def render(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    views = list(cameras(args))
    settings = {
        "width": args.width,
        "height": args.height,
        "antialiasing": args.antialiasing,
        "background": "transparent",
        "light_direction": [-0.5, 0.8, 1],
        "light_color": "white",
        "object_color": "rgb(137,196,237)",
        "light_intensity": 0.85,
        "ambient": 0.15,
    }
    # Publish the output only after every view has rendered successfully.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
        staging = Path(staging)
        for view in views:
            command = [args.renderer, str(args.input.resolve()), "-o", str(staging / view["image"])]
            options = settings | {key: view[key] for key in ("camera_position", "camera_target", "camera_up", "fov")}
            for key, value in options.items():
                encoded = ",".join(map(str, value)) if isinstance(value, list) else str(value)
                command.append(f"--{key.replace('_', '-')}={encoded}")
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"{view['image']}: renderer timed out after {args.timeout:g} seconds") from error
            if result.returncode:
                raise RuntimeError(f"{view['image']}: {result.stderr.strip() or result.stdout.strip() or f'renderer exited with code {result.returncode}'}")
            print(f"Rendered {view['image']}", flush=True)
        manifest = {
            "schema_version": 1,
            "shader": str(args.input.resolve()),
            "renderer": args.renderer,
            "distance": args.distance,
            "render": settings,
            "views": views,
        }
        (staging / "views.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    print(f"Saved {len(views)} views to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="GLSL file defining float sdf(vec3 p)")
    parser.add_argument("--output", type=Path, required=True, help="new output directory (must not exist)")
    parser.add_argument("--distance", type=positive_float, required=True, help="camera distance from target in shader units")
    parser.add_argument("--target", type=target_vector, default=[0, 0, 0], help="look-at point X,Y,Z (default: 0,0,0)")
    parser.add_argument("--azimuths", type=positive_int, default=8, help="number of equally spaced orbit directions (default: 8)")
    parser.add_argument("--elevations", type=elevation_list, default=[-30, 0, 30], help="integer degrees, e.g. --elevations=-30,0,30")
    parser.add_argument("--top", action="store_true", help="include a camera at +Y with up=-Z")
    parser.add_argument("--bottom", action="store_true", help="include a camera at -Y with up=+Z")
    parser.add_argument("--fov", type=positive_float, default=45, help="vertical field of view in degrees (default: 45)")
    parser.add_argument("--width", type=positive_int, default=512)
    parser.add_argument("--height", type=positive_int, default=512)
    parser.add_argument("--antialiasing", type=int, choices=[1, 4], default=1)
    parser.add_argument("--renderer", default="sdf-view", help="renderer executable (default: sdf-view; CPU: sdf-view-software)")
    parser.add_argument("--timeout", type=positive_float, default=90, help="timeout per view in seconds (default: 90)")
    args = parser.parse_args()
    if args.fov >= 180:
        parser.error("--fov must be less than 180 degrees")
    try:
        render(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
