#!/usr/bin/env python3
"""Validate the prepared dataset and generate an offline image gallery."""

import hashlib
import html
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
folders = sorted(p for p in (ROOT / "dataset").iterdir() if p.is_dir())
assert len(folders) == 63, f"Expected 63 models, found {len(folders)}"
cards = []
for folder in folders:
    assert {p.name for p in folder.iterdir()} == {"sdf.glsl", "render.png", "metadata.json"}, folder
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["name"] == folder.name
    assert metadata["status"] == "rendered", folder
    assert hashlib.sha256((folder / "sdf.glsl").read_bytes()).hexdigest() == metadata["shader_sha256"], folder
    with Image.open(folder / "render.png") as image:
        image.load()
        assert image.size == (metadata["render"]["width"], metadata["render"]["height"]), folder
        assert image.mode == "RGBA", folder
        bounds = image.getchannel("A").getbbox()
        assert bounds and list(bounds) == metadata["image_check"]["foreground_bounds"], folder
        assert 0 < bounds[0] < bounds[2] < image.width, folder
        assert 0 < bounds[1] < bounds[3] < image.height, folder
    name = html.escape(folder.name)
    cards.append(f'<article><a href="dataset/{name}/render.png"><img src="dataset/{name}/render.png" alt="{name}"></a>'
                 f'<h2>{name}</h2><a href="dataset/{name}/sdf.glsl">GLSL</a> · '
                 f'<a href="dataset/{name}/metadata.json">Metadata</a></article>')
page = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SDF dataset preview</title>
<style>
body { font-family: sans-serif; margin: 2rem; background: #fafafa; color: #222; }
main { display: grid; grid-template-columns: repeat(auto-fill,minmax(240px,1fr)); gap: 1rem; }
article { background: #eee; padding: 1rem; border-radius: 8px; }
img { width: 100%; aspect-ratio: 1; object-fit: contain; }
h2 { font-size: 1rem; } a { color: #164f8c; }
</style>
<h1>SDF dataset: 63 models</h1>
<p>All renders passed automated file, hash, nonempty foreground, and framing checks.
Visual review remains pending. Images use transparent backgrounds.</p>
<main>''' + "\n".join(cards) + "</main></html>\n"
(ROOT / "preview.html").write_text(page)
print(f"Validated {len(folders)} models: shader hashes, PNG decoding, dimensions, foreground, and framing.")
print("Generated preview.html; visual review remains pending.")
