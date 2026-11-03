"""Verify bundled dependency bootstrap and cache reuse."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runtime

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


class RuntimeTests(unittest.TestCase):
    def test_install_is_offline_and_reused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "results") as directory:
            root = Path(directory)
            (root / "wheels").mkdir()
            (root / "requirements.lock").write_text("# Empty test environment\n")

            def create(path):
                path.mkdir(parents=True)

            with patch.object(runtime, "__file__", str(root / "runtime.py")), patch.object(
                    runtime.sys, "version_info", (3, 11, 16)), patch.object(
                    runtime.os, "environ", {}), patch.object(runtime.venv, "EnvBuilder") as builder, patch.object(
                    runtime.subprocess, "run") as install, patch.object(runtime.os, "execve") as execute:
                builder.return_value.create.side_effect = create
                runtime.bootstrap()
                command = install.call_args.args[0]
                self.assertIn("--no-index", command)
                self.assertIn("--require-hashes", command)
                self.assertIn("--no-cache-dir", command)
                runtime.bootstrap()
                self.assertEqual(install.call_count, 1)
                self.assertEqual(execute.call_count, 2)

    def test_development_does_not_bootstrap(self):
        with patch.object(runtime.os, "execve") as execute:
            runtime.bootstrap()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
