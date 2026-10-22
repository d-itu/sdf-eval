"""Tests for orbit cameras and atomic multi-view rendering."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from render_views import cameras, main


class RenderViewsTests(unittest.TestCase):
    def test_orbit_and_poles(self):
        with patch("sys.argv", ["render_views.py", "sphere.glsl", "--output", "unused", "--distance", "2", "--azimuths", "4", "--elevations=0", "--top", "--bottom"]):
            # Exercise the argument parser without launching the renderer.
            with patch("render_views.render") as render:
                self.assertEqual(main(), 0)
                args = render.call_args.args[0]
        views = list(cameras(args))
        self.assertEqual([view["image"] for view in views], [
            "az000_el+00.png", "az090_el+00.png", "az180_el+00.png", "az270_el+00.png", "top.png", "bottom.png",
        ])
        self.assertEqual(views[0]["camera_position"], [0, 0, 2])
        self.assertAlmostEqual(views[1]["camera_position"][0], 2)
        self.assertEqual(views[-2]["camera_position"], [0, 2, 0])
        self.assertEqual(views[-2]["camera_up"], [0, 0, -1])
        self.assertEqual(views[-1]["camera_up"], [0, 0, 1])

    def test_failure_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "views"
            with patch("sys.argv", ["render_views.py", "sphere.glsl", "--output", str(output), "--distance", "2", "--azimuths", "2", "--elevations=0"]):
                with patch("render_views.subprocess.run", side_effect=[
                    subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 1, stderr="shader error"),
                ]):
                    self.assertEqual(main(), 1)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_manifest_camera_matches_render_command(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "views"

            def fake_run(command, **kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(b"PNG")
                return subprocess.CompletedProcess(command, 0)

            with patch("sys.argv", ["render_views.py", "sphere.glsl", "--output", str(output), "--distance", "3", "--azimuths", "1", "--elevations=0", "--top"]):
                with patch("render_views.subprocess.run", side_effect=fake_run) as run:
                    self.assertEqual(main(), 0)
            manifest = json.loads((output / "views.json").read_text())
            self.assertEqual(len(manifest["views"]), 2)
            for call, view in zip(run.call_args_list, manifest["views"]):
                command = call.args[0]
                self.assertIn("--camera-position=" + ",".join(map(str, view["camera_position"])), command)
                self.assertTrue((output / view["image"]).exists())
            self.assertNotIn("shader_sha256", manifest)
            self.assertNotIn("status", manifest["views"][0])


if __name__ == "__main__":
    unittest.main()
