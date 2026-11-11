#!/usr/bin/env python3
"""Prepare one local reconstruction task with multi-view inputs."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from build_verification import prepare_materials
from render_views import positive_float

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    if not args.reference.is_file():
        raise ValueError(f"reference shader is missing: {args.reference}")
    distance = positive_float(str(args.distance))

    output = args.output.resolve()
    updating = output.exists() and args.update_verification
    if output.exists() and not args.update_verification:
        raise ValueError(f"output directory already exists: {output}")
    if updating and not (output / "verification").is_dir():
        raise ValueError(
            f"verification directory is missing: {output / 'verification'}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        task = Path(temporary) / "task"
        task.mkdir()
        prepare_materials(task / "verification", args.reference)
        (task / "verification/reference/image-config.json").write_text(
            json.dumps({"distance": distance}, indent=2) + "\n", encoding="utf-8"
        )
        binary_dir = task / "verification/scripts/bin"
        binary_dir.mkdir()
        (binary_dir / "sdf-eval").symlink_to(
            os.path.relpath(
                args.sdf_eval.resolve(), output / "verification/scripts/bin"
            )
        )
        if updating:
            current = output / "verification"
            backup = Path(temporary) / "previous-verification"
            current.rename(backup)
            try:
                (task / "verification").rename(current)
            except OSError:
                backup.rename(current)
                raise
            return
        (task / "run/output").mkdir(parents=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("render_views.py")),
            str(task / "verification/reference/sdf.glsl"),
            "--output",
            str(task / "run/input"),
            "--renderer",
            args.renderer,
            f"--distance={distance}",
            "--top",
            "--bottom",
            "--antialiasing=4",
        ]
        subprocess.run(command, check=True)
        manifest_path = task / "run/input/views.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("shader", None)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        prompt = (
            (ROOT / "benchmark_prompt.md")
            .read_text(encoding="utf-8")
            .format(
                title=args.title,
                input="Reference views: `input/*.png`. Camera settings: `input/views.json`.",
                tools=f"Renderer: `{args.renderer}`. Use the camera settings in `input/views.json`.",
                output="Write the completed shader to `output/sdf.glsl`.",
            )
        )
        (task / "prompt.md").write_text(prompt, encoding="utf-8")
        task.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="Reference GLSL shader file"
    )
    parser.add_argument("--output", type=Path, required=True, help="Task directory")
    parser.add_argument(
        "--update-verification",
        action="store_true",
        help="Replace verification/ in an existing task without changing run/ or prompt.md",
    )
    parser.add_argument(
        "--distance",
        type=positive_float,
        required=True,
        help="Reference-derived orbit camera distance in shader units",
    )
    parser.add_argument(
        "--renderer",
        default="sdf-view",
        help="Renderer executable (CPU: sdf-view-software)",
    )
    parser.add_argument(
        "--sdf-eval",
        type=Path,
        default=ROOT / "target/debug/sdf-eval",
        help="Evaluator executable to link (default: target/debug/sdf-eval)",
    )
    parser.add_argument(
        "--title",
        default="SDF Reconstruction",
        help="Task title (default: SDF Reconstruction)",
    )
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
