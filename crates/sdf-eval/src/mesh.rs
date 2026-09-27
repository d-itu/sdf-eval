//! Zero-isosurface extraction and area-weighted sampling.
//! Six consistently oriented tetrahedra per cell avoid ambiguous cube cases.

use anyhow::{Result, ensure};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::array;

pub type Point = [f64; 3];
type Triangle = [Point; 3];

pub fn sub(a: Point, b: Point) -> Point {
    array::from_fn(|i| a[i] - b[i])
}
fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn area(t: &Triangle) -> f64 {
    cross(sub(t[1], t[0]), sub(t[2], t[0]))
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        * 0.5
}

#[derive(Serialize)]
pub struct MeshInfo {
    pub triangles: usize,
    pub area: f64,
    pub boundary_occupied: bool,
}

pub struct Mesh {
    triangles: Vec<Triangle>,
    cumulative_area: Vec<f64>,
    pub info: MeshInfo,
}

impl Mesh {
    fn push(&mut self, triangle: Triangle) {
        let area = area(&triangle);
        if area > 0.0 {
            self.info.area += area;
            self.triangles.push(triangle);
            self.cumulative_area.push(self.info.area);
        }
    }

    pub fn extract(values: &[f32], bounds: [f32; 6], n: u32) -> Result<Self> {
        ensure!(values.iter().all(|v| v.is_finite()), "non_finite_values");
        let mut mesh = Self {
            triangles: Vec::new(),
            cumulative_area: Vec::new(),
            info: MeshInfo {
                triangles: 0,
                area: 0.0,
                boundary_occupied: false,
            },
        };
        let side = n as usize + 1;
        let at = |x: usize, y: usize, z: usize| values[(z * side + y) * side + x];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    if (x == 0
                        || y == 0
                        || z == 0
                        || x + 1 == side
                        || y + 1 == side
                        || z + 1 == side)
                        && at(x, y, z) <= 0.0
                    {
                        mesh.info.boundary_occupied = true;
                    }
                }
            }
        }
        const OFFSETS: [[usize; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ];
        const TETS: [[usize; 4]; 6] = [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ];
        for z in 0..n as usize {
            for y in 0..n as usize {
                for x in 0..n as usize {
                    let v: [f32; 8] = array::from_fn(|i| {
                        at(x + OFFSETS[i][0], y + OFFSETS[i][1], z + OFFSETS[i][2])
                    });
                    if v.iter().all(|v| *v < 0.0) || v.iter().all(|v| *v >= 0.0) {
                        continue;
                    }
                    let p: [Point; 8] = array::from_fn(|i| {
                        let cell = [x + OFFSETS[i][0], y + OFFSETS[i][1], z + OFFSETS[i][2]];
                        // Match the FP32 GPU lattice before interpolating in FP64.
                        array::from_fn(|a| {
                            f64::from(
                                bounds[a] + cell[a] as f32 / n as f32 * (bounds[a + 3] - bounds[a]),
                            )
                        })
                    });
                    for tet in TETS {
                        let mut inside = [0; 4];
                        let mut outside = [0; 4];
                        let (mut ni, mut no) = (0, 0);
                        for i in tet {
                            if v[i] < 0.0 {
                                inside[ni] = i;
                                ni += 1;
                            } else {
                                outside[no] = i;
                                no += 1;
                            }
                        }
                        let edge = |a: usize, b: usize| {
                            let t = f64::from(v[a]) / (f64::from(v[a]) - f64::from(v[b]));
                            array::from_fn(|i| p[a][i] + t * (p[b][i] - p[a][i]))
                        };
                        match ni {
                            1 => mesh.push([
                                edge(inside[0], outside[0]),
                                edge(inside[0], outside[1]),
                                edge(inside[0], outside[2]),
                            ]),
                            3 => mesh.push([
                                edge(outside[0], inside[0]),
                                edge(outside[0], inside[1]),
                                edge(outside[0], inside[2]),
                            ]),
                            2 => {
                                let a = edge(inside[0], outside[0]);
                                let b = edge(inside[0], outside[1]);
                                let c = edge(inside[1], outside[0]);
                                let d = edge(inside[1], outside[1]);
                                mesh.push([a, b, c]);
                                mesh.push([b, d, c]);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        ensure!(!mesh.triangles.is_empty(), "no_surface_extracted");
        mesh.info.triangles = mesh.triangles.len();
        Ok(mesh)
    }

    pub fn sample(&self, count: usize, seed: u64) -> Vec<Point> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let target = rng.random::<f64>() * self.info.area;
                let index = self
                    .cumulative_area
                    .partition_point(|area| *area <= target)
                    .min(self.triangles.len() - 1);
                let [a, b, c] = self.triangles[index];
                let u = rng.random::<f64>().sqrt();
                let v = rng.random::<f64>();
                array::from_fn(|i| (1.0 - u) * a[i] + u * (1.0 - v) * b[i] + u * v * c[i])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_weighting_and_barycentric_centroid() {
        let mut mesh = Mesh {
            triangles: vec![],
            cumulative_area: vec![],
            info: MeshInfo {
                triangles: 0,
                area: 0.0,
                boundary_occupied: false,
            },
        };
        mesh.push([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
        mesh.push([[10., 0., 0.], [12., 0., 0.], [10., 2., 0.]]);
        let points = mesh.sample(100_000, 42);
        let small: Vec<_> = points.iter().filter(|p| p[0] < 5.0).collect();
        assert!((small.len() as f64 / points.len() as f64 - 0.2).abs() < 0.005);
        for axis in 0..2 {
            let mean = small.iter().map(|p| p[axis]).sum::<f64>() / small.len() as f64;
            assert!((mean - 1.0 / 3.0).abs() < 0.006);
        }
        assert_eq!(points, mesh.sample(100_000, 42));
    }

    #[test]
    fn sphere_mesh_converges_with_resolution() {
        use std::f64::consts::PI;
        let error = |n: u32| {
            let side = n + 1;
            let field: Vec<f32> = (0..side.pow(3))
                .map(|i| {
                    let p = [i % side, (i / side) % side, i / (side * side)]
                        .map(|v| -1.5 + v as f32 / n as f32 * 3.0);
                    p.iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0
                })
                .collect();
            let mesh = Mesh::extract(&field, [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], n).unwrap();
            assert!(!mesh.info.boundary_occupied);
            let radial_error = mesh
                .sample(2000, 42)
                .iter()
                .map(|p| (p.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs())
                .sum::<f64>()
                / 2000.0;
            ((mesh.info.area - 4.0 * PI).abs(), radial_error)
        };
        let coarse = error(16);
        let fine = error(32);
        assert!(fine.0 < coarse.0 * 0.4);
        assert!(fine.1 < coarse.1 * 0.4);
    }

    #[test]
    fn extracted_plane_has_correct_area_without_caps() {
        let n = 7;
        let values: Vec<f32> = (0..(n + 1) * (n + 1) * (n + 1))
            .map(|i| (i % (n + 1)) as f32 / n as f32 - 0.43)
            .collect();
        let mesh = Mesh::extract(&values, [0., 0., 0., 1., 1., 1.], n).unwrap();
        assert!((mesh.info.area - 1.0).abs() < 1e-6);
        assert!(mesh.info.boundary_occupied);
        for p in mesh.sample(100, 1) {
            assert!((p[0] - 0.43).abs() < 1e-6);
        }
    }
}
