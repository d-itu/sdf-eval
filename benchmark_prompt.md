# Task Instruction: {title}

## Objective

Reconstruct the depicted object as a static 3D signed distance field (SDF) in GLSL.
Match its silhouette, proportions, and visible details across the provided reference images;
infer surfaces that are not visible.
Ignore color, lighting, and background.
Use +Y as up, center the object's bounding box at the origin, and choose a scale consistent with the references.

## Execution

- Work autonomously; no interactive input or human feedback is available.
- Define `float sdf(vec3 p)` with negative values inside and positive values outside. The field must be suitable for sphere tracing.
- Write self-contained, Naga-compatible GLSL: no `#version`, `main`, uniforms, textures, resource bindings, or identifiers starting with `sdf_view_`.
- Use the provided software, where applicable, to render and refine the shader against the references. Do not request clarification or rely on manual review.
- Deliver plain GLSL at the designated output path, without Markdown fences.

## Inputs

{input}

## Tools

{tools}

## Outputs

{output}
