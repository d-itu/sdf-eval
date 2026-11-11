"""Tests for single-task benchmark packaging."""

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_benchmark import prepare


class PrepareBenchmarkTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "results").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.reference = root / "dataset/Sphere/source.glsl"
        self.reference.parent.mkdir(parents=True)
        self.reference.write_text("reference shader")
        (self.reference.parent / "metadata.json").write_text('{"unused": true}')
        self.output = root / "task"

    def prepare(self, **overrides):
        args = Namespace(
            reference=self.reference,
            output=self.output,
            distance=4,
            renderer="sdf-view",
            title="SDF Reconstruction",
            update_verification=False,
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        prepare(args)

    def fake_render(self, command, check):
        self.assertTrue(check)
        self.assertIn("--distance=4.0", command)
        self.assertIn("--antialiasing=4", command)
        self.assertIn("--bottom", command)
        self.assertIn("--top", command)
        input_dir = Path(command[command.index("--output") + 1])
        input_dir.mkdir()
        (input_dir / "az000_el+00.png").touch()
        (input_dir / "views.json").write_text(
            json.dumps({"shader": command[2], "views": []})
        )

    def test_creates_isolated_task_with_matching_cameras(self):
        with patch("prepare_benchmark.subprocess.run", side_effect=self.fake_render):
            self.prepare()
        self.assertEqual(
            {p.name for p in self.output.iterdir()},
            {"run", "verification", "prompt.md"},
        )
        self.assertEqual(
            {p.name for p in (self.output / "run").iterdir()}, {"input", "output"}
        )
        self.assertEqual(list((self.output / "run/output").iterdir()), [])
        self.assertEqual(
            {p.name for p in (self.output / "run/input").iterdir()},
            {"views.json", "az000_el+00.png"},
        )
        self.assertNotIn(
            "shader", json.loads((self.output / "run/input/views.json").read_text())
        )
        prompt = (self.output / "prompt.md").read_text()
        self.assertIn("# Task Instruction: SDF Reconstruction", prompt)
        self.assertNotIn("Sphere", prompt)
        self.assertIn("Renderer: `sdf-view`", prompt)
        self.assertNotIn("{title}", prompt)
        self.assertNotIn("verification/", prompt)
        self.assertEqual(
            (self.output / "verification/reference/sdf.glsl").read_text(),
            "reference shader",
        )
        self.assertEqual(
            {p.name for p in (self.output / "verification").iterdir()},
            {"reference"},
        )
        self.assertFalse(list(self.output.rglob("metadata.json")))
        self.assertEqual(
            json.loads(
                (self.output / "verification/reference/image-config.json").read_text()
            ),
            {"distance": 4.0},
        )
        self.assertTrue((ROOT / "verification/scripts/evaluate.py").is_file())

    def test_explicit_title(self):
        with patch("prepare_benchmark.subprocess.run", side_effect=self.fake_render):
            self.prepare(title="Custom Object")
        self.assertIn(
            "# Task Instruction: Custom Object", (self.output / "prompt.md").read_text()
        )

    def test_updates_only_verification_in_existing_task(self):
        with patch("prepare_benchmark.subprocess.run", side_effect=self.fake_render):
            self.prepare()
        (self.output / "run/output/sdf.glsl").write_text("prediction")
        (self.output / "verification/scripts").mkdir()
        (self.output / "verification/scripts/evaluate.py").write_text("old entry")
        prompt = (self.output / "prompt.md").read_bytes()
        views = (self.output / "run/input/views.json").read_bytes()
        self.reference.write_text("updated reference")
        with patch("prepare_benchmark.subprocess.run") as render:
            self.prepare(distance=5, update_verification=True)
        render.assert_not_called()
        self.assertEqual((self.output / "prompt.md").read_bytes(), prompt)
        self.assertEqual((self.output / "run/input/views.json").read_bytes(), views)
        self.assertEqual(
            (self.output / "run/output/sdf.glsl").read_text(), "prediction"
        )
        self.assertEqual(
            {p.name for p in (self.output / "verification").iterdir()},
            {"reference"},
        )
        self.assertFalse((self.output / "verification/scripts").exists())
        self.assertEqual(
            (self.output / "verification/reference/sdf.glsl").read_text(),
            "updated reference",
        )
        self.assertEqual(
            json.loads(
                (self.output / "verification/reference/image-config.json").read_text()
            ),
            {"distance": 5.0},
        )

    def test_update_requires_existing_verification(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "verification directory is missing"):
            self.prepare(update_verification=True)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_update_flag_creates_new_task_when_output_is_missing(self):
        with patch("prepare_benchmark.subprocess.run", side_effect=self.fake_render):
            self.prepare(update_verification=True)
        self.assertTrue((self.output / "run/input/views.json").is_file())
        self.assertTrue((self.output / "verification/reference/sdf.glsl").is_file())

    def test_failed_update_preserves_existing_task(self):
        with patch("prepare_benchmark.subprocess.run", side_effect=self.fake_render):
            self.prepare()
        original = (self.output / "verification/reference/sdf.glsl").read_text()
        with (
            patch(
                "prepare_benchmark.shutil.copy2",
                side_effect=OSError("copy failed"),
            ),
            self.assertRaisesRegex(OSError, "copy failed"),
        ):
            self.prepare(distance=5, update_verification=True)
        self.assertEqual(
            (self.output / "verification/reference/sdf.glsl").read_text(), original
        )
        self.assertTrue((self.output / "run/input/views.json").is_file())

    def test_rejects_missing_reference_and_existing_output(self):
        self.reference.unlink()
        with self.assertRaisesRegex(ValueError, "reference shader is missing"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.reference.write_text("reference shader")
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()

    def test_failure_does_not_publish_partial_task(self):
        with (
            patch(
                "prepare_benchmark.subprocess.run",
                side_effect=OSError("renderer failed"),
            ),
            self.assertRaisesRegex(OSError, "renderer failed"),
        ):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
