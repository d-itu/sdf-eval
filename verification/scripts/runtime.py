"""Bootstrap the uploaded wheelhouse using the host's Python 3.11."""

import hashlib
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import venv


def bootstrap():
    root = Path(__file__).resolve().parent
    if not (root / "wheels").is_dir():
        return
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The production bundle requires CPython 3.11")
    fingerprint = hashlib.sha256((root / "requirements.lock").read_bytes()).hexdigest()
    environment = root / ".runtime" / fingerprint
    python = environment / "bin/python"
    marker = environment / ".ready"
    if os.environ.get("SDF_VERIFICATION_RUNTIME") == str(environment):
        return
    environment.parent.mkdir(parents=True, exist_ok=True)
    with (environment.parent / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not marker.exists():
            venv.EnvBuilder(with_pip=True).create(environment)
            subprocess.run([
                str(python), "-m", "pip", "install", "--no-index", "--no-cache-dir", "--require-hashes",
                "--find-links", str(root / "wheels"), "-r", str(root / "requirements.lock"),
            ], check=True)
            marker.touch()
    os.execve(str(python), [str(python), str(root / "evaluate.py"), *sys.argv[1:]],
              os.environ | {"SDF_VERIFICATION_RUNTIME": str(environment)})
