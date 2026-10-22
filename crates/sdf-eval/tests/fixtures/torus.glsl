float sdf(vec3 p) {
    return length(vec2(length(p.xy) - 0.9, p.z)) - 0.3;
}
