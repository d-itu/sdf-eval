float sdf(vec3 p) { return max(max(abs(p.x), abs(p.y)), abs(p.z)) - 0.75; }
