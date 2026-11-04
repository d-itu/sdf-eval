"""Verify the production directory contract without downloading dependencies."""

import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_verification", ROOT / "scripts/build_verification.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BundleTests(unittest.TestCase):
    def test_materials_include_one_entry_and_only_required_reference_files(self):
        (ROOT / "results").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            reference = root / "source"
            reference.mkdir()
            (reference / "sdf.glsl").write_text("reference shader")
            (reference / "unused.png").touch()
            config = root / "camera.json"
            config.write_text('{"distance": 4}')
            staging = root / "verification"
            scripts = builder.prepare_materials(staging, reference, config)
            self.assertEqual(list(staging.rglob("evaluate.py")), [scripts / "evaluate.py"])
            self.assertEqual({path.name for path in staging.iterdir()}, {"reference", "rubric", "scripts"})
            self.assertEqual({path.name for path in (staging / "reference").iterdir()},
                             {"sdf.glsl", "image-config.json"})
            self.assertEqual((staging / "reference/sdf.glsl").read_text(), "reference shader")
            self.assertEqual((staging / "rubric/scoring.md").read_bytes(),
                             (ROOT / "verification/scoring.md").read_bytes())
            self.assertFalse((scripts / "pyproject.toml").exists())
            result = subprocess.run([sys.executable, str(scripts / "evaluate.py"), "--help"],
                                    capture_output=True, text=True, check=True, cwd=root)
            self.assertIn("--reference", result.stdout)


if __name__ == "__main__":
    unittest.main()
