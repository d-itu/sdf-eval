Reconstruct the pictured object as a static 3D SDF in GLSL. Match its silhouette, proportions, and details; infer hidden surfaces. Ignore color, lighting, and background.

- Define `float sdf(vec3 p)`, negative inside and positive outside, suitable for sphere tracing.
- Use +Y up; center the bounding box at the origin and choose a reasonable scale based on the image.
- Use self-contained, Naga-compatible GLSL. No `#version`, `main`, uniforms, textures, bindings, or `sdf_view_` names.
- You may use `https://github.com/d-itu/sdf-view` to render your output, compare it with the reference, and refine it.

Example (unit sphere):
```glsl
float sdf(vec3 p) { return length(p) - 1.0; }
```

You may reason through the reconstruction and use tools such as Python.
Write the final shader to `sdf.glsl` as plain GLSL, without Markdown.
