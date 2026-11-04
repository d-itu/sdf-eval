"""Test orchestration and report behavior without GPU or scientific packages."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import evaluate

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


class PipelineTests(unittest.TestCase):
    def test_directory_cli_writes_only_json_with_default_bounds(self):
        with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
            root = Path(directory)
            reference = root / "reference"
            output = root / "output"
            reference.mkdir()
            output.mkdir()
            (reference / "sdf.glsl").write_text("reference shader")
            (output / "sdf.glsl").write_text("prediction shader")
            score_path = root / "reports/score.json"
            report = {"status": "scored", "total_score": 100}
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--score", str(score_path)]), patch.object(
                    evaluate, "score", return_value=report) as scorer:
                self.assertEqual(evaluate.main(), 0)
            scorer.assert_called_once_with(reference / "sdf.glsl", output / "sdf.glsl",
                                           Path(evaluate.__file__).parent / "bin/sdf-eval",
                                           (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0))
            self.assertEqual(json.loads(score_path.read_text()), report)
            self.assertEqual(list(score_path.parent.iterdir()), [score_path])

    def test_missing_prediction_shader_is_scored_as_invalid(self):
        with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
            root = Path(directory)
            reference = root / "reference"
            output = root / "output"
            reference.mkdir()
            output.mkdir()
            (reference / "sdf.glsl").write_text("reference shader")
            score_path = root / "score.json"
            evaluator = root / "sdf-eval"
            evaluator.touch()
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--score", str(score_path),
                                            "--evaluator", str(evaluator)]):
                # A missing prediction should still produce an invalid-submission report.
                self.assertEqual(evaluate.main(), 0)
            self.assertEqual(json.loads(score_path.read_text())["status"], "invalid_submission")

    def test_score_cannot_overwrite_shader(self):
        with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
            root = Path(directory)
            reference = root / "reference"
            output = root / "output"
            reference.mkdir()
            output.mkdir()
            shader = reference / "sdf.glsl"
            shader.write_text("reference shader")
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--score", str(shader)]):
                with self.assertRaises(SystemExit):
                    evaluate.main()
            self.assertEqual(shader.read_text(), "reference shader")

    def test_image_scoring_receives_shader_files_from_directories(self):
        with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
            root = Path(directory)
            reference = root / "reference"
            output = root / "output"
            reference.mkdir()
            output.mkdir()
            config = root / "camera.json"
            config.write_text('{"distance": 4}')
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--score", str(root / "score.json"),
                                            "--image-config", str(config)]), patch.object(
                    evaluate, "bootstrap"), patch.object(evaluate, "score", return_value={
                        "status": "invalid_submission"}), patch.object(evaluate, "score_images") as images:
                self.assertEqual(evaluate.main(), 0)
            args, report = images.call_args.args
            self.assertEqual(args.gt, reference / "sdf.glsl")
            self.assertEqual(args.gen, output / "sdf.glsl")
            self.assertEqual(report["status"], "invalid_submission")

    def test_render_configuration_is_shared_and_metrics_are_reported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "results") as directory:
            root = Path(directory)
            config = root / "camera.json"
            config.write_text(json.dumps({"distance": 4, "target": [1, 2, 3]}))
            args = argparse.Namespace(image_config=config, artifacts=root / "images",
                                      gt=root / "gt.glsl", gen=root / "gen.glsl",
                                      renderer=root / "renderer", weights=root / "weights")
            report = {"status": "scored"}
            images = argparse.Namespace(evaluate=lambda args: {"status": "ok", "metrics": {
                "ssim_mean": 1.0, "lpips_mean": 0.0}})
            render = argparse.Namespace(__file__=str(root / "render_views.py"))
            with patch.object(evaluate, "helper", side_effect=[render, images]), patch.object(
                    evaluate.subprocess, "run", return_value=argparse.Namespace(returncode=0)) as run:
                evaluate.score_images(args, report)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][8:], commands[1][8:])
            self.assertIn("--target=1,2,3", commands[0])
            self.assertEqual(report["image_metrics"]["metrics"]["lpips_mean"], 0)

    def test_invalid_submission_does_not_render_prediction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "results") as directory:
            root = Path(directory)
            config = root / "camera.json"
            config.write_text('{"distance": 4}')
            args = argparse.Namespace(image_config=config, artifacts=root / "images",
                                      gt=root / "gt", gen=root / "gen", renderer=root / "renderer")
            report = {"status": "invalid_submission", "reason": "prediction_compile_failed"}
            render = argparse.Namespace(__file__=str(root / "render_views.py"))
            with patch.object(evaluate, "helper", return_value=render), patch.object(
                    evaluate.subprocess, "run", return_value=argparse.Namespace(returncode=0)) as run:
                evaluate.score_images(args, report)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(report["image_metrics"]["status"], "invalid_submission")

    def test_render_error_is_not_a_numeric_score(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "results") as directory:
            root = Path(directory)
            config = root / "camera.json"
            config.write_text('{"distance": 4}')
            args = argparse.Namespace(image_config=config, artifacts=root / "images",
                                      gt=root / "gt", renderer=root / "renderer", gen=root / "gen")
            with patch.object(evaluate, "helper", return_value=argparse.Namespace(__file__="render.py")), patch.object(
                    evaluate.subprocess, "run", return_value=argparse.Namespace(returncode=1, stderr="driver failed")):
                with self.assertRaisesRegex(evaluate.EvaluationError, "reference rendering failed"):
                    evaluate.score_images(args, {"status": "scored"})


if __name__ == "__main__":
    unittest.main()
