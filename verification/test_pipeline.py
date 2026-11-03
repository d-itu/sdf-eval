"""Test orchestration and report behavior without GPU or scientific packages."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


class PipelineTests(unittest.TestCase):
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
