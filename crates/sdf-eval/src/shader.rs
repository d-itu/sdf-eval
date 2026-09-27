//! GLSL compute wrappers. Each input snippet is compiled in a separate module.

const HEADER: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std140) uniform sdf_eval_Parameters {
    vec4 sdf_eval_lower;
    vec4 sdf_eval_upper;
    uvec4 sdf_eval_grid; // resolution, offset, batch count, volume count
};
layout(set = 0, binding = 1, std430) buffer sdf_eval_Occupancy {
    uint sdf_eval_occupancy[];
};
layout(set = 0, binding = 2, std430) buffer sdf_eval_Partials {
    uvec4 sdf_eval_partials[];
};
layout(set = 0, binding = 3, std430) buffer sdf_eval_Totals {
    uvec4 sdf_eval_totals[];
};

vec3 sdf_eval_position(uint index) {
    uint n = sdf_eval_grid.x;
    vec3 fraction;
    if (index < sdf_eval_grid.w) {
        fraction = (vec3(index % n, (index / n) % n, index / (n * n)) + 0.5) / float(n);
    } else {
        uint boundary = index - sdf_eval_grid.w;
        uint face = boundary / (n * n);
        uint cell = boundary % (n * n);
        vec2 uv = (vec2(cell % n, cell / n) + 0.5) / float(n);
        float side = float(face % 2u);
        if (face < 2u) fraction = vec3(side, uv);
        else if (face < 4u) fraction = vec3(uv.x, side, uv.y);
        else fraction = vec3(uv, side);
    }
    return sdf_eval_lower.xyz + fraction * (sdf_eval_upper.xyz - sdf_eval_lower.xyz);
}
"#;

const REFERENCE: &str = r#"
void main() {
    uint local = gl_GlobalInvocationID.x;
    if (local >= sdf_eval_grid.z) return;
    float value = sdf(sdf_eval_position(sdf_eval_grid.y + local));
    sdf_eval_occupancy[local] = (isnan(value) || isinf(value)) ? 2u : uint(value < 0.0);
}
"#;

const PREDICTION: &str = r#"
shared uvec4 sdf_eval_a[256];
shared uvec4 sdf_eval_b[256];
void main() {
    uint local = gl_GlobalInvocationID.x;
    uint lane = gl_LocalInvocationID.x;
    uvec4 a = uvec4(0u);
    uvec4 b = uvec4(0u);
    if (local < sdf_eval_grid.z) {
        uint index = sdf_eval_grid.y + local;
        float value = sdf(sdf_eval_position(index));
        bool invalid = isnan(value) || isinf(value);
        uint reference = sdf_eval_occupancy[local];
        bool ref_inside = reference == 1u;
        bool pred_inside = !invalid && value < 0.0;
        bool volume = index < sdf_eval_grid.w;
        if (volume) {
            a = uvec4(uint(ref_inside && pred_inside), uint(ref_inside || pred_inside),
                      uint(ref_inside), uint(pred_inside));
        }
        b = uvec4(uint(reference == 2u), uint(invalid),
                  uint(!volume && ref_inside), uint(!volume && pred_inside));
    }
    sdf_eval_a[lane] = a;
    sdf_eval_b[lane] = b;
    barrier();
    for (uint stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            sdf_eval_a[lane] += sdf_eval_a[lane + stride];
            sdf_eval_b[lane] += sdf_eval_b[lane + stride];
        }
        barrier();
    }
    if (lane == 0u) {
        sdf_eval_partials[2u * gl_WorkGroupID.x] = sdf_eval_a[0];
        sdf_eval_partials[2u * gl_WorkGroupID.x + 1u] = sdf_eval_b[0];
    }
}
"#;

const REDUCE: &str = r#"
shared uvec4 sdf_eval_a[256];
shared uvec4 sdf_eval_b[256];
void main() {
    uint lane = gl_LocalInvocationID.x;
    uint groups = (sdf_eval_grid.z + 255u) / 256u;
    uvec4 a = uvec4(0u);
    uvec4 b = uvec4(0u);
    for (uint i = lane; i < groups; i += 256u) {
        a += sdf_eval_partials[2u * i];
        b += sdf_eval_partials[2u * i + 1u];
    }
    sdf_eval_a[lane] = a;
    sdf_eval_b[lane] = b;
    barrier();
    for (uint stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            sdf_eval_a[lane] += sdf_eval_a[lane + stride];
            sdf_eval_b[lane] += sdf_eval_b[lane + stride];
        }
        barrier();
    }
    if (lane == 0u) {
        sdf_eval_totals[0] = sdf_eval_a[0];
        sdf_eval_totals[1] = sdf_eval_b[0];
    }
}
"#;

pub fn reference(snippet: &str) -> String {
    format!("{HEADER}\n#line 1 1\n{snippet}\n#line 1 0\n{REFERENCE}")
}

pub fn prediction(snippet: &str) -> String {
    format!("{HEADER}\n#line 1 1\n{snippet}\n#line 1 0\n{PREDICTION}")
}

pub fn reduce() -> String {
    format!("{HEADER}\n{REDUCE}")
}
