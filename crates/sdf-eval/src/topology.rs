//! Homology ranks of a finite union of closed axis-aligned voxel cubes.
use std::collections::VecDeque;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{gpu::Gpu, iou::Settings, surface};

#[derive(Debug, Serialize)]
pub struct Topology {
    pub betti: [u64; 3],
    pub euler_characteristic: i64,
    pub occupied_voxels: usize,
    pub boundary_occupied: bool,
}

pub fn calculate(occupied: &[bool], n: usize) -> Topology {
    assert_eq!(occupied.len(), n * n * n);
    let index = |x: usize, y: usize, z: usize| (z * n + y) * n + x;
    let mut seen = vec![false; occupied.len()];
    let mut queue = VecDeque::new();
    let mut components = 0;
    let mut cavities = 0;
    let mut boundary_occupied = false;
    for start in 0..occupied.len() {
        if seen[start] {
            continue;
        }
        let foreground = occupied[start];
        let mut touches_boundary = false;
        seen[start] = true;
        queue.push_back(start);
        while let Some(i) = queue.pop_front() {
            let x = i % n;
            let y = (i / n) % n;
            let z = i / (n * n);
            touches_boundary |= [x, y, z].iter().any(|&v| v == 0 || v + 1 == n);
            // Closed cubes use 26-connectivity; their complement uses 6-connectivity.
            for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let length = dx.abs() + dy.abs() + dz.abs();
                        if length == 0 || (!foreground && length != 1) {
                            continue;
                        }
                        let (xx, yy, zz) = (x as isize + dx, y as isize + dy, z as isize + dz);
                        if xx < 0
                            || yy < 0
                            || zz < 0
                            || xx >= n as isize
                            || yy >= n as isize
                            || zz >= n as isize
                        {
                            continue;
                        }
                        let j = index(xx as usize, yy as usize, zz as usize);
                        if !seen[j] && occupied[j] == foreground {
                            seen[j] = true;
                            queue.push_back(j);
                        }
                    }
                }
            }
        }
        if foreground {
            components += 1;
            boundary_occupied |= touches_boundary;
        } else if !touches_boundary {
            cavities += 1;
        }
    }

    // Count each vertex and its positive-axis edges, faces, and cube once.
    let mut euler = 0i64;
    for z in 0..=n {
        for y in 0..=n {
            for x in 0..=n {
                let mut mask = 0u8;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (xx, yy, zz) = (x + dx, y + dy, z + dz);
                            if xx > 0
                                && yy > 0
                                && zz > 0
                                && xx <= n
                                && yy <= n
                                && zz <= n
                                && occupied[index(xx - 1, yy - 1, zz - 1)]
                            {
                                mask |= 1 << (dx + 2 * dy + 4 * dz);
                            }
                        }
                    }
                }
                euler += i64::from(mask != 0);
                euler -= [0xaa, 0xcc, 0xf0]
                    .iter()
                    .filter(|&&m| mask & m != 0)
                    .count() as i64;
                euler += [0x88, 0xa0, 0xc0]
                    .iter()
                    .filter(|&&m| mask & m != 0)
                    .count() as i64;
                euler -= i64::from(mask & 0x80 != 0);
            }
        }
    }
    // Alexander duality gives beta_2 from bounded complement components.
    // A finite subset of R^3 has beta_3 = 0, hence chi = b0 - b1 + b2.
    let tunnels = components + cavities - euler;
    assert!(tunnels >= 0);
    Topology {
        betti: [components as u64, tunnels as u64, cavities as u64],
        euler_characteristic: euler,
        occupied_voxels: occupied.iter().filter(|&&v| v).count(),
        boundary_occupied,
    }
}

pub fn evaluate(
    reference: &str,
    prediction: &str,
    settings: Settings,
    report: &mut Value,
) -> Result<Value> {
    report["stage"] = json!("adapter_initialization");
    let gpu = Gpu::new().context("adapter_initialization_failed")?;
    report["adapter"] = json!(gpu.adapter);
    let mut results = Vec::with_capacity(2);
    for (name, source) in [("reference", reference), ("prediction", prediction)] {
        report["stage"] = json!(format!("{name}_compile"));
        let pipeline = gpu
            .pipeline(
                name,
                format!(
                    "{}\n#line 1 1\n{source}\n#line 1 0\n{}",
                    surface::HEADER,
                    surface::MAIN
                ),
            )
            .with_context(|| format!("{name}_compile_failed"))?;
        report["stage"] = json!(format!("{name}_field"));
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = surface::field(&gpu, &pipeline, settings, true);
        if let Some(error) = futures::executor::block_on(scope.pop()) {
            return Err(anyhow::Error::from(error).context("gpu_execution_failed"));
        }
        let (values, batch) = result?;
        report["effective_batch_size"] = json!(batch);
        let occupied: Vec<bool> = values.iter().map(|&v| v < 0.0).collect();
        drop(values);
        report["stage"] = json!(format!("{name}_topology"));
        let topology = calculate(&occupied, settings.resolution as usize);
        report[format!("{name}_topology")] = json!(topology);
        results.push(topology);
    }
    let boundary = results.iter().any(|t| t.boundary_occupied);
    report["boundary_occupied"] = json!(boundary);
    ensure!(
        results.iter().all(|t| t.occupied_voxels > 0),
        "empty_occupancy"
    );
    ensure!(!boundary, "boundary_occupied");
    let r = results[0].betti;
    let p = results[1].betti;
    report["stage"] = json!("complete");
    Ok(json!({"status": "ok", "metrics": {
        "betti_absolute_difference": [r[0].abs_diff(p[0]), r[1].abs_diff(p[1]), r[2].abs_diff(p[2])],
        "betti_exact_match": r == p,
    }}))
}

#[cfg(test)]
mod tests {
    use super::calculate;

    fn shape(n: usize, inside: impl Fn(usize, usize, usize) -> bool) -> super::Topology {
        let values: Vec<_> = (0..n * n * n)
            .map(|i| inside(i % n, (i / n) % n, i / (n * n)))
            .collect();
        calculate(&values, n)
    }

    #[test]
    fn solids_tunnels_cavities_and_contacts() {
        assert_eq!(shape(5, |_, _, _| false).betti, [0, 0, 0]);
        let full = shape(5, |_, _, _| true);
        assert_eq!(full.betti, [1, 0, 0]);
        assert!(full.boundary_occupied);
        let shell = shape(5, |x, y, z| {
            [x, y, z].iter().all(|&v| (1..=3).contains(&v)) && [x, y, z] != [2, 2, 2]
        });
        assert_eq!(shell.betti, [1, 0, 1]);
        assert!(!shell.boundary_occupied);
        assert_eq!(
            shape(5, |x, y, z| z == 2
                && (1..=3).contains(&x)
                && (1..=3).contains(&y)
                && (x != 2 || y != 2))
            .betti,
            [1, 1, 0]
        );
        assert_eq!(
            shape(5, |x, y, z| [x, y, z] == [1, 1, 1]
                || [x, y, z] == [3, 3, 3])
            .betti,
            [2, 0, 0]
        );
        for other in [[2, 1, 1], [2, 2, 1], [2, 2, 2]] {
            assert_eq!(
                shape(5, |x, y, z| [x, y, z] == [1, 1, 1] || [x, y, z] == other).betti,
                [1, 0, 0]
            );
        }
        assert_eq!(shape(1, |_, _, _| true).betti, [1, 0, 0]);
    }
}
