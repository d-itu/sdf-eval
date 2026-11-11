"""Tests for the development task evaluation wrapper."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_task import main


class EvaluateTaskTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "results").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.temporary.cleanup)
        self.task = Path(self.temporary.name) / "task"
        self.reference = self.task / "verification/reference"
        self.output = self.task / "run/output"
        self.reference.mkdir(parents=True)
        self.output.mkdir(parents=True)
        (self.task / "run/input").mkdir()
        (self.reference / "sdf.glsl").write_text("reference", encoding="utf-8")

    def invoke(self, *options):
        with patch("sys.argv", ["evaluate_task.py", str(self.task), *options]):
            return main()

    def test_passes_task_paths_and_local_image_resources(self):
        (self.reference / "image-config.json").write_text(
            json.dumps({"distance": 3}), encoding="utf-8"
        )
        with (
            patch(
                "evaluate_task.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
            patch("evaluate_task.shutil.which", return_value="/usr/bin/sdf-view"),
        ):
            self.assertEqual(self.invoke("--python", sys.executable), 0)
        command = run.call_args.args[0]
        self.assertEqual(
            command[:2],
            [sys.executable, str(ROOT / "verification/scripts/evaluate.py")],
        )
        for key, value in (
            ("--reference", self.reference),
            ("--output", self.output),
            ("--score", self.task / "verification/score.json"),
            ("--report", self.task / "verification/report.json"),
            ("--evaluator", ROOT / "target/debug/sdf-eval"),
            ("--renderer", Path("/usr/bin/sdf-view")),
            ("--weights", ROOT / "results/.cache/torch"),
            ("--artifacts", self.task / "verification/image-artifacts"),
        ):
            self.assertEqual(command[command.index(key) + 1], str(value))

    def test_geometry_only_propagates_exit_status(self):
        with patch("evaluate_task.subprocess.run") as run:
            run.return_value.returncode = 7
            self.assertEqual(self.invoke("--python", sys.executable), 7)
        command = run.call_args.args[0]
        self.assertNotIn("--renderer", command)
        self.assertNotIn("--weights", command)

    def test_retry_uses_a_new_artifact_directory(self):
        (self.task / "verification/image-artifacts").mkdir()
        (self.task / "verification/image-artifacts-1").mkdir()
        with patch("evaluate_task.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(self.invoke("--python", sys.executable), 0)
        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--artifacts") + 1],
            str(self.task / "verification/image-artifacts-2"),
        )

    def test_reuses_results_until_input_or_output_changes(self):
        def evaluate(*_args, **_kwargs):
            (self.task / "verification/score.json").write_text("{}")
            (self.task / "verification/report.json").write_text("{}")
            return subprocess.CompletedProcess([], 0)

        with patch("evaluate_task.subprocess.run", side_effect=evaluate) as run:
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            self.assertEqual(run.call_count, 1)
            (self.task / "run/input/views.json").write_text("input")
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            (self.output / "sdf.glsl").write_text("prediction")
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            (self.reference / "sdf.glsl").write_text("updated reference")
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            self.assertEqual(run.call_count, 4)

    def test_missing_artifacts_or_results_invalidates_cache(self):
        (self.reference / "image-config.json").write_text('{"distance": 3}')

        def evaluate(command, **_kwargs):
            artifacts = Path(command[command.index("--artifacts") + 1])
            for role in ("reference", "prediction"):
                (artifacts / role).mkdir(parents=True)
                (artifacts / role / "views.json").write_text("{}")
            (self.task / "verification/score.json").write_text("{}")
            (self.task / "verification/report.json").write_text("{}")
            return subprocess.CompletedProcess([], 0)

        with (
            patch("evaluate_task.subprocess.run", side_effect=evaluate) as run,
            patch("evaluate_task.shutil.which", return_value="/usr/bin/sdf-view"),
        ):
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            self.assertEqual(run.call_count, 1)
            (self.task / "verification/image-artifacts/prediction/views.json").unlink()
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                run.call_args.args[0][-2:],
                ["--weights", str(ROOT / "results/.cache/torch")],
            )
            self.assertEqual(
                run.call_args.args[0][run.call_args.args[0].index("--artifacts") + 1],
                str(self.task / "verification/image-artifacts-1"),
            )
            (self.task / "verification/score.json").unlink()
            self.assertEqual(self.invoke("--python", sys.executable), 0)
            self.assertEqual(run.call_count, 3)

    def test_invalid_task_does_not_run_evaluator(self):
        (self.reference / "sdf.glsl").unlink()
        with (
            patch("evaluate_task.subprocess.run") as run,
            self.assertRaises(SystemExit) as error,
        ):
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_missing_renderer_does_not_run_evaluator(self):
        (self.reference / "image-config.json").write_text('{"distance": 3}')
        with (
            patch("evaluate_task.shutil.which", return_value=None),
            patch("evaluate_task.subprocess.run") as run,
            self.assertRaises(SystemExit) as error,
        ):
            self.invoke("--python", sys.executable)
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
