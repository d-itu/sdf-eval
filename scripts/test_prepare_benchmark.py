"""Tests for single-task benchmark packaging."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_benchmark import prepare

ROOT = Path(__file__).resolve().parents[1]


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

    def fake_render(self, command, check):
        self.assertTrue(check)
        self.assertIn("--distance=4.0", command)
        self.assertIn("--antialiasing=4", command)
        self.assertIn("--bottom", command)
        self.assertIn("--top", command)
        input_dir = Path(command[command.index("--output") + 1])
        input_dir.mkdir()
        (input_dir / "az000_el+00.png").touch()
        (input_dir / "views.json").write_text(json.dumps({"shader": command[2], "views": []}))

    def test_creates_isolated_task_with_matching_cameras(self):
        with patch("prepare_benchmark.subprocess.run", side_effect=self.fake_render):
            prepare(self.reference, self.output, 4, "sdf-view")
        self.assertEqual({p.name for p in self.output.iterdir()}, {"run", "verification"})
        self.assertEqual({p.name for p in (self.output / "run").iterdir()}, {"input", "output"})
        self.assertEqual(list((self.output / "run/output").iterdir()), [])
        self.assertEqual({p.name for p in (self.output / "run/input").iterdir()},
                         {"views.json", "az000_el+00.png"})
        shader = json.loads((self.output / "run/input/views.json").read_text())["shader"]
        self.assertEqual(shader, "../../verification/reference/sdf.glsl")
        self.assertEqual((self.output / "run/input" / shader).resolve(),
                         self.output / "verification/reference/sdf.glsl")
        self.assertEqual((self.output / "verification/reference/sdf.glsl").read_text(), "reference shader")
        self.assertFalse(list(self.output.rglob("metadata.json")))
        self.assertEqual(json.loads((self.output / "verification/reference/image-config.json").read_text()),
                         {"distance": 4.0})
        self.assertTrue((self.output / "verification/scripts/evaluate.py").is_file())
        self.assertTrue((self.output / "verification/scripts/render_views.py").is_file())

    def test_rejects_missing_reference_and_existing_output(self):
        self.reference.unlink()
        with self.assertRaisesRegex(ValueError, "reference shader is missing"):
            prepare(self.reference, self.output, 4, "sdf-view")
        self.assertFalse(self.output.exists())
        self.reference.write_text("reference shader")
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.reference, self.output, 4, "sdf-view")

    def test_failure_does_not_publish_partial_task(self):
        with (patch("prepare_benchmark.subprocess.run", side_effect=OSError("renderer failed")),
              self.assertRaisesRegex(OSError, "renderer failed")):
            prepare(self.reference, self.output, 4, "sdf-view")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
