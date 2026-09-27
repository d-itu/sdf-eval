//! Exact nearest-neighbor distances between independently sampled surfaces.
use crate::mesh::Point;
use anyhow::Result;
use kdtree::{KdTree, distance::squared_euclidean};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct FScore {
    pub threshold: f64,
    pub precision: f64,
    pub recall: f64,
    pub f_score: f64,
}

#[derive(Debug, Serialize)]
pub struct Metrics {
    pub chamfer_l2_mean: f64,
    pub chamfer_l2_squared_mean: f64,
    pub prediction_to_reference_mean: f64,
    pub reference_to_prediction_mean: f64,
    pub f_scores: Vec<FScore>,
}

fn distances(source: &[Point], target: &[Point]) -> Result<Vec<f64>> {
    let mut tree = KdTree::new(3);
    for (i, p) in target.iter().enumerate() {
        tree.add(*p, i)?;
    }
    source
        .iter()
        .map(|p| Ok(tree.nearest(p, 1, &squared_euclidean)?[0].0.sqrt()))
        .collect()
}

pub fn compare(reference: &[Point], prediction: &[Point], thresholds: &[f64]) -> Result<Metrics> {
    let pr = distances(prediction, reference)?;
    let rp = distances(reference, prediction)?;
    let mean = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;
    let squared_mean = |d: &[f64]| d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64;
    let prediction_to_reference_mean = mean(&pr);
    let reference_to_prediction_mean = mean(&rp);
    Ok(Metrics {
        chamfer_l2_mean: (prediction_to_reference_mean + reference_to_prediction_mean) * 0.5,
        chamfer_l2_squared_mean: (squared_mean(&pr) + squared_mean(&rp)) * 0.5,
        prediction_to_reference_mean,
        reference_to_prediction_mean,
        f_scores: thresholds
            .iter()
            .map(|&threshold| {
                let precision =
                    pr.iter().filter(|&&d| d <= threshold).count() as f64 / pr.len() as f64;
                let recall =
                    rp.iter().filter(|&&d| d <= threshold).count() as f64 / rp.len() as f64;
                FScore {
                    threshold,
                    precision,
                    recall,
                    f_score: if precision + recall == 0.0 {
                        0.0
                    } else {
                        2.0 * precision * recall / (precision + recall)
                    },
                }
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kd_tree_matches_brute_force() {
        let a: Vec<Point> = (0..53)
            .map(|i| [(i as f64).sin(), (i as f64 * 0.3).cos(), i as f64 / 20.0])
            .collect();
        let b: Vec<Point> = (0..71)
            .map(|i| [(i as f64 * 0.7).cos(), i as f64 / 10.0, (i as f64).sin()])
            .collect();
        for (p, d) in a.iter().zip(distances(&a, &b).unwrap()) {
            let expected = b
                .iter()
                .map(|q| squared_euclidean(p, q))
                .fold(f64::INFINITY, f64::min)
                .sqrt();
            assert_eq!(d, expected);
        }
    }
    #[test]
    fn metric_definitions_and_threshold_inclusion() {
        let r = [[0., 0., 0.], [2., 0., 0.]];
        let p = [[0., 0., 0.], [4., 0., 0.], [6., 0., 0.]];
        let result = compare(&r, &p, &[0.0, 2.0]).unwrap();
        assert_eq!(result.chamfer_l2_mean, 1.5);
        assert!((result.chamfer_l2_squared_mean - 13.0 / 3.0).abs() < 1e-14);
        assert_eq!(result.f_scores[0].precision, 1.0 / 3.0);
        assert_eq!(result.f_scores[0].recall, 0.5);
        assert!((result.f_scores[0].f_score - 0.4).abs() < 1e-15);
        assert_eq!(result.f_scores[1].precision, 2.0 / 3.0);
        assert_eq!(result.f_scores[1].recall, 1.0);
    }
}
