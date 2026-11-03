#!/usr/bin/env python3
"""Build an Ubuntu 24.04 x86_64 / CPython 3.11 verification upload directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
RENDERER_REVISION = "cfb01c7535db99cec2351408672d069101789596"
WEIGHT_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
WEIGHT_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def run(*command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/verification")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("build on Ubuntu 24.04 x86_64 (locally or in CI)")
    release = platform.freedesktop_os_release()
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        parser.error("production binaries must be built on Ubuntu 24.04; use the CI workflow")
    output = args.output.resolve()
    if output.exists():
        parser.error("output must not exist")
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside the repository")
    work = ROOT / ".work/verification"
    work.mkdir(parents=True, exist_ok=True)
    for key, path in {"UV_PYTHON_INSTALL_DIR": ROOT / ".work/python",
                      "PIP_CACHE_DIR": ROOT / ".cache/pip",
                      "CARGO_HOME": ROOT / ".work/cargo"}.items():
        os.environ.setdefault(key, str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".verification-") as temporary:
        staging = Path(temporary) / "verification"
        staging.mkdir()
        for name in ("evaluate.py", "runtime.py", "scoring.md", "pyproject.toml", "uv.lock", "requirements.lock"):
            shutil.copy2(ROOT / "verification" / name, staging / name)
        for name in ("evaluate_images.py", "render_views.py"):
            shutil.copy2(ROOT / "scripts" / name, staging / name)
        run("uv", "venv", "--python", "3.11", "--seed", "--clear", str(work / "download-env"))
        run(str(work / "download-env/bin/python"), "-m", "pip", "download", "--only-binary=:all:",
            "--require-hashes", "-r", str(staging / "requirements.lock"), "-d", str(staging / "wheels"))
        run("cargo", "build", "--locked", "--release", "--target-dir", str(work / "eval-target"), "-p", "sdf-eval", cwd=ROOT)
        renderer = work / "sdf-view"
        if not renderer.exists():
            run("git", "clone", "https://github.com/d-itu/sdf-view", str(renderer))
        run("git", "checkout", "--detach", RENDERER_REVISION, cwd=renderer)
        run("cargo", "build", "--locked", "--release", "--target-dir", str(work / "renderer-target"), cwd=renderer)
        (staging / "bin").mkdir()
        shutil.copy2(work / "eval-target/release/sdf-eval", staging / "bin/sdf-eval")
        shutil.copy2(work / "renderer-target/release/sdf-view", staging / "bin/sdf-view")
        weights = staging / "weights/checkpoints"
        weights.mkdir(parents=True)
        weight = weights / "alexnet-owt-7be5be79.pth"
        urllib.request.urlretrieve(WEIGHT_URL, weight)
        with weight.open("rb") as stream:
            weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if weight_hash != WEIGHT_SHA256:
            raise RuntimeError("AlexNet weight checksum does not match the upstream identifier")
        (staging / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "platform": "ubuntu-24.04-x86_64", "python": "3.11",
            "renderer_revision": RENDERER_REVISION, "alexnet_sha256": WEIGHT_SHA256,
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }, indent=2) + "\n")
        run(str(work / "download-env/bin/python"), str(staging / "evaluate.py"), "--help")
        run(str(work / "download-env/bin/python"), str(ROOT / "scripts/smoke_verification.py"),
            "--entry", str(staging / "evaluate.py"))
        shutil.rmtree(staging / ".runtime")
        shutil.rmtree(staging / "__pycache__", ignore_errors=True)
        staging.rename(output)
    print(f"Upload the contents of {output}")


if __name__ == "__main__":
    main()
