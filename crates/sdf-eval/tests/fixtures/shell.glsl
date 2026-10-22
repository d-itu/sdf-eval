float sdf(vec3 p) {
    return abs(length(p) - 0.8) - 0.2;
}
