"""Tests for manifest matching and image metric behavior."""

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_images import evaluate, load_manifest, main, matched_views


def temporary_directory():
    root = Path(__file__).resolve().parents[1] / "results/.cache"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def manifest():
    return {
        "schema_version": 1,
        "render": {
            "width": 64, "height": 64, "antialiasing": 4,
            "background": "transparent", "light_direction": [-0.5, 0.8, 1],
            "light_color": "white", "object_color": "rgb(137,196,237)",
            "light_intensity": 0.85, "ambient": 0.15,
        },
        "views": [{"image": name, "camera_position": [0, 0, 3],
                   "camera_target": [0, 0, 0], "camera_up": [0, 1, 0], "fov": 45}
                  for name in ("front.png", "back.png")],
    }


class ManifestTests(unittest.TestCase):
    def test_report_publication_and_failure(self):
        with temporary_directory() as temp:
            output = Path(temp) / "scores.json"
            command = ["evaluate_images.py", "--reference", "ref.json", "--prediction", "pred.json",
                       "--output", str(output)]
            with patch("sys.argv", command), patch("evaluate_images.evaluate", side_effect=ValueError("missing view")):
                self.assertEqual(main(), 1)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])
            report = {"metrics": {"ssim_mean": 1.0, "lpips_mean": 0.0}}
            with patch("sys.argv", command), patch("evaluate_images.evaluate", return_value=report):
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads(output.read_text()), report)
            with patch("sys.argv", command), patch("evaluate_images.evaluate") as compute:
                self.assertEqual(main(), 1)
                compute.assert_not_called()
            self.assertEqual(json.loads(output.read_text()), report)

    def test_matching_ignores_order_and_provenance(self):
        with temporary_directory() as temp:
            ref, pred = Path(temp) / "ref.json", Path(temp) / "pred.json"
            data = manifest()
            ref.write_text(json.dumps(data))
            del data["schema_version"]
            data["shader"] = "another.glsl"
            data["views"].reverse()
            pred.write_text(json.dumps(data))
            self.assertEqual(matched_views(ref, pred)[1], ["front.png", "back.png"])

    def test_rejects_mismatched_protocol(self):
        with temporary_directory() as temp:
            ref, pred = Path(temp) / "ref.json", Path(temp) / "pred.json"
            ref.write_text(json.dumps(manifest()))
            for change in ("camera", "render", "missing"):
                data = manifest()
                if change == "camera":
                    data["views"][0]["fov"] = 60
                elif change == "render":
                    data["render"]["ambient"] = 0.2
                else:
                    data["views"].pop()
                pred.write_text(json.dumps(data))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    matched_views(ref, pred)

    def test_rejects_duplicate_paths_traversal_and_invalid_camera(self):
        with temporary_directory() as temp:
            path = Path(temp) / "views.json"
            for change in ("duplicate", "traversal", "nan", "version"):
                data = manifest()
                if change == "duplicate":
                    data["views"].append(copy.deepcopy(data["views"][0]))
                elif change == "traversal":
                    data["views"][0]["image"] = "../outside.png"
                elif change == "nan":
                    data["views"][0]["fov"] = float("nan")
                else:
                    data["schema_version"] = 2
                path.write_text(json.dumps(data))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_manifest(path)


class MetricTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RUN_LPIPS_TESTS") == "1",
                         "requires metric dependencies and pretrained weights")
    def test_identity_transparent_rgb_and_changed_geometry(self):
        from PIL import Image, ImageDraw

        with temporary_directory() as temp:
            root = Path(temp)
            for name in ("ref", "pred"):
                directory = root / name
                directory.mkdir()
                (directory / "views.json").write_text(json.dumps(manifest()))
                for view in ("front.png", "back.png"):
                    # Different hidden RGB must not affect the score.
                    image = Image.new("RGBA", (64, 64), (255 if name == "pred" else 0, 0, 0, 0))
                    ImageDraw.Draw(image).ellipse((12, 12, 52, 52), fill=(137, 196, 237, 255))
                    image.save(directory / view)
            args = argparse.Namespace(reference=root / "ref/views.json", prediction=root / "pred/views.json",
                                      cache_dir=Path("results/.cache/torch"), device="cpu")
            identity = evaluate(args)
            self.assertAlmostEqual(identity["metrics"]["ssim_mean"], 1.0, places=6)
            self.assertAlmostEqual(identity["metrics"]["lpips_mean"], 0.0, places=6)
            Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(root / "pred/front.png")
            changed = evaluate(args)
            self.assertLess(changed["metrics"]["ssim_mean"], 1.0)
            self.assertGreater(changed["metrics"]["lpips_mean"], 0.0)
            (root / "pred/back.png").unlink()
            with self.assertRaises(OSError):
                evaluate(args)


if __name__ == "__main__":
    unittest.main()
