float sdf(vec3 p) {
    p -= vec3(0.5, 0.0, 0.0);
    return max(max(abs(p.x), abs(p.y)), abs(p.z)) - 0.75;
}
