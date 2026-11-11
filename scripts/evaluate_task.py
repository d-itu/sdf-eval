#!/usr/bin/env python3
"""Evaluate one development benchmark task with the shared verifier."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(directories):
    digest = hashlib.sha256()
    for directory in directories:
        digest.update(str(directory.name).encode())
        digest.update(b"\0")
        if not directory.is_dir():
            digest.update(b"missing\0")
            continue
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            digest.update(str(relative).encode())
            digest.update(b"\0")
            if path.is_symlink():
                digest.update(b"link\0")
                digest.update(str(path.readlink()).encode())
            elif path.is_file():
                digest.update(b"file\0")
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                digest.update(b"directory\0")
            digest.update(b"\0")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", type=Path, help="Task directory containing verification/ and run/"
    )
    parser.add_argument(
        "--python", type=Path, help="Python with verification dependencies installed"
    )
    parser.add_argument(
        "--evaluator", type=Path, default=ROOT / "target/debug/sdf-eval"
    )
    parser.add_argument("--renderer", type=Path, help="sdf-view executable")
    parser.add_argument("--weights", type=Path, default=ROOT / "results/.cache/torch")
    args = parser.parse_args()

    task = args.task.resolve()
    reference = task / "verification/reference"
    output = task / "run/output"
    if not (reference / "sdf.glsl").is_file():
        parser.error(f"reference shader is missing: {reference / 'sdf.glsl'}")
    if not output.is_dir():
        parser.error(f"submission directory is missing: {output}")

    python = args.python or ROOT / ".venv/bin/python"
    if not python.is_file():
        if args.python:
            parser.error(f"Python executable is missing: {python}")
        python = Path(sys.executable)
    verification = task / "verification"
    score = verification / "score.json"
    report = verification / "report.json"
    cache = verification / "evaluation-cache.json"
    try:
        key = fingerprint((task / "run/input", output, reference))
        previous = (
            json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
        )
    except OSError as error:
        parser.exit(1, f"error: {error}\n")
    except (ValueError, TypeError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    artifact_name = previous.get("artifacts")
    if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
        previous = {}
        artifact_name = "image-artifacts"
    cached_artifacts = verification / artifact_name
    if (
        previous.get("fingerprint") == key
        and score.is_file()
        and report.is_file()
        and (
            not previous.get("artifacts_present")
            or all(
                (cached_artifacts / role / "views.json").is_file()
                for role in ("reference", "prediction")
            )
        )
    ):
        return 0

    artifacts = verification / "image-artifacts"
    suffix = 1
    while artifacts.exists():
        artifacts = verification / f"image-artifacts-{suffix}"
        suffix += 1
    command = [
        str(python),
        str(ROOT / "verification/scripts/evaluate.py"),
        "--reference",
        str(reference),
        "--output",
        str(output),
        "--score",
        str(score),
        "--report",
        str(report),
        "--evaluator",
        str(args.evaluator.resolve()),
        "--artifacts",
        str(artifacts),
    ]
    if (reference / "image-config.json").is_file():
        renderer = args.renderer or shutil.which("sdf-view")
        if not renderer:
            parser.error("image evaluation requires sdf-view; pass --renderer")
        command.extend(("--renderer", str(Path(renderer).resolve())))
        command.extend(("--weights", str(args.weights.resolve())))
    try:
        cache.unlink(missing_ok=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0 and score.is_file() and report.is_file():
            cache.write_text(
                json.dumps(
                    {
                        "fingerprint": key,
                        "artifacts": artifacts.name,
                        "artifacts_present": artifacts.exists(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return result.returncode
    except OSError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
