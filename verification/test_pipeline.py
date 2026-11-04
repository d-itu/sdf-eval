"""Test orchestration and report behavior without GPU or scientific packages."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import evaluate

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def diagnostic_report(status="scored"):
    return {"status": status, "total_score": 0.0 if status == "invalid_submission" else 100.0,
            "items": [{"title": name, "max_points": points,
                       "earned_points": 0.0 if status == "invalid_submission" else float(points)}
                      for name, points in (("Volumetric IoU", 50), ("Surface F-score", 30),
                                           ("Betti number similarity", 20))]}


class PipelineTests(unittest.TestCase):
    def test_production_schema_and_weighted_partial_score(self):
        report = diagnostic_report()
        for item, points in zip(report["items"], (25.0, 0.0, 10.0)):
            item["earned_points"] = points
        report["total_score"] = 35.0
        result = evaluate.scoring_result(report)
        self.assertEqual(set(result), {"Physical, Geometric, and Structural Capability", "total_score"})
        items = result["Physical, Geometric, and Structural Capability"]["Geometric Form and Topological Structure"]
        self.assertEqual([item["score"] for item in items], [50.0, 0.0, 50.0])
        self.assertEqual([item["weight"] for item in items], [0.5, 0.3, 0.2])
        self.assertTrue(all(set(item) == {"name", "score", "weight"} for item in items))
        self.assertEqual(sum(item["score"] * item["weight"] for item in items), result["total_score"])

    def test_default_score_and_optional_diagnostic_report(self):
        with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
            root = Path(directory)
            reference, output = root / "reference", root / "output"
            reference.mkdir()
            output.mkdir()
            report = diagnostic_report("invalid_submission")
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--report", str(root / "details.json")]), patch.object(
                    evaluate, "score", return_value=report):
                # The default is relative to the process working directory.
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    self.assertEqual(evaluate.main(), 0)
                finally:
                    os.chdir(previous)
            self.assertEqual(json.loads((root / "score.json").read_text()), evaluate.scoring_result(report))
            self.assertEqual(json.loads((root / "details.json").read_text()), report)

    def test_cli_rejects_files_as_inputs_and_unsafe_destinations(self):
        with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
            root = Path(directory)
            reference, output = root / "reference", root / "output"
            reference.mkdir()
            output.mkdir()
            shader = output / "sdf.glsl"
            shader.write_text("submission")
            for ref, out, destination in ((shader, output, root / "score.json"),
                                           (reference, shader, root / "score.json"),
                                           (reference, output, root),
                                           (reference, output, output / "score.json")):
                with self.subTest(reference=ref, output=out, score=destination), patch.object(
                        sys, "argv", ["evaluate.py", "--reference", str(ref), "--output", str(out),
                                      "--score", str(destination)]):
                    with self.assertRaises(SystemExit):
                        evaluate.main()
            self.assertEqual(shader.read_text(), "submission")

    def test_evaluation_failure_does_not_publish_result_files(self):
        with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
            root = Path(directory)
            reference, output = root / "reference", root / "output"
            reference.mkdir()
            output.mkdir()
            score_path, report_path = root / "score.json", root / "details.json"
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--score", str(score_path),
                                            "--report", str(report_path)]), patch.object(
                    evaluate, "score", side_effect=evaluate.EvaluationError("reference is unusable")):
                self.assertEqual(evaluate.main(), 1)
            self.assertFalse(score_path.exists())
            self.assertFalse(report_path.exists())

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
            report = diagnostic_report()
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--score", str(score_path)]), patch.object(
                    evaluate, "score", return_value=report) as scorer:
                self.assertEqual(evaluate.main(), 0)
            scorer.assert_called_once_with(reference / "sdf.glsl", output / "sdf.glsl",
                                           Path(evaluate.__file__).parent / "bin/sdf-eval",
                                           (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0))
            self.assertEqual(json.loads(score_path.read_text()), evaluate.scoring_result(report))
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
            self.assertEqual(json.loads(score_path.read_text())["total_score"], 0)

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
            config.rename(reference / "image-config.json")
            with patch.object(sys, "argv", ["evaluate.py", "--reference", str(reference),
                                            "--output", str(output), "--score", str(root / "score.json")]), patch.object(
                    evaluate, "bootstrap"), patch.object(evaluate, "score", return_value=
                        diagnostic_report("invalid_submission")), patch.object(evaluate, "score_images") as images:
                self.assertEqual(evaluate.main(), 0)
            args, report = images.call_args.args
            self.assertEqual(args.gt, reference / "sdf.glsl")
            self.assertEqual(args.gen, output / "sdf.glsl")
            self.assertEqual(args.image_config, reference / "image-config.json")
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
