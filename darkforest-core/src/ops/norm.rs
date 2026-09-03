//! Normalization ops — batch_norm, instance_norm, group_norm, rms_norm.

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};

const EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
// ---------------------------------------------------------------------------
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor> {
    let data = x.to_vec();
    let features = *x.shape.last().unwrap();
    let batch = x.numel() / features;
    let g = gamma.to_vec();

    let mut out = vec![0.0f32; data.len()];
    for b in 0..batch {
        let offset = b * features;
        let rms = (data[offset..offset + features]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            / features as f32
            + eps)
            .sqrt();
        for f in 0..features {
            out[offset + f] = (data[offset + f] / rms) * g[f];
        }
    }
    Tensor::from_vec_device(out, x.shape.clone(), x.device.clone())
}

// ---------------------------------------------------------------------------
// GroupNorm: group_norm(x, num_groups, gamma, beta, eps)
//   x shape: [N, C, *] — C must be divisible by num_groups
// ---------------------------------------------------------------------------
pub fn group_norm(
    x: &Tensor,
    num_groups: usize,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if x.ndim() < 2 {
        return Err(anyhow!("group_norm: input must be at least 2D"));
    }
    let n = x.shape[0];
    let c = x.shape[1];
    if c % num_groups != 0 {
        return Err(anyhow!(
            "group_norm: C={} not divisible by num_groups={}",
            c,
            num_groups
        ));
    }
    let group_size = c / num_groups;
    let spatial: usize = x.shape[2..].iter().product::<usize>().max(1);
    let elements_per_group = group_size * spatial;

    let src = x.to_vec();
    let g = gamma.to_vec();
    let b = beta.to_vec();
    let mut out = vec![0.0f32; src.len()];

    for batch in 0..n {
        for group in 0..num_groups {
            let start_c = group * group_size;
            // Compute mean and var over group
            let mut mean = 0.0f32;
            for gc in 0..group_size {
                let ch = start_c + gc;
                for s in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + s;
                    mean += src[idx];
                }
            }
            mean /= elements_per_group as f32;

            let mut var = 0.0f32;
            for gc in 0..group_size {
                let ch = start_c + gc;
                for s in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + s;
                    var += (src[idx] - mean).powi(2);
                }
            }
            var /= elements_per_group as f32;
            let rstd = 1.0 / (var + eps).sqrt();

            for gc in 0..group_size {
                let ch = start_c + gc;
                for s in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + s;
                    out[idx] = (src[idx] - mean) * rstd * g[ch] + b[ch];
                }
            }
        }
    }
    Tensor::from_vec_device(out, x.shape.clone(), x.device.clone())
}

// ---------------------------------------------------------------------------
// InstanceNorm: normalize each (N, C) channel independently
//   x shape: [N, C, *]
// ---------------------------------------------------------------------------
pub fn instance_norm(
    x: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    if x.ndim() < 3 {
        return Err(anyhow!(
            "instance_norm: input must be at least 3D [N, C, ...]"
        ));
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();

    let src = x.to_vec();
    let mut out = vec![0.0f32; src.len()];

    for batch in 0..n {
        for ch in 0..c {
            let offset = batch * c * spatial + ch * spatial;
            let slice = &src[offset..offset + spatial];
            let mean: f32 = slice.iter().sum::<f32>() / spatial as f32;
            let var: f32 = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / spatial as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for s in 0..spatial {
                let norm = (slice[s] - mean) * rstd;
                let scale = gamma.map(|g| g.get(ch)).unwrap_or(1.0);
                let shift = beta.map(|b| b.get(ch)).unwrap_or(0.0);
                out[offset + s] = norm * scale + shift;
            }
        }
    }
    Tensor::from_vec_device(out, x.shape.clone(), x.device.clone())
}

// ---------------------------------------------------------------------------
// BatchNorm: normalize across the batch dimension for each channel
//   Training mode: uses batch statistics
//   Inference mode: uses running_mean / running_var
// ---------------------------------------------------------------------------
pub struct BatchNormStats {
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub momentum: f32,
}

pub fn batch_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    running_stats: Option<&mut BatchNormStats>,
    eps: f32,
    training: bool,
) -> Result<(Tensor, Vec<f32>, Vec<f32>)> {
    if x.ndim() < 2 {
        return Err(anyhow!("batch_norm: input must be at least 2D"));
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product::<usize>().max(1);
    let num_per_channel = n * spatial;

    let src = x.to_vec();
    let g = gamma.to_vec();
    let b = beta.to_vec();
    let mut out = vec![0.0f32; src.len()];
    let mut means = vec![0.0f32; c];
    let mut rstds = vec![0.0f32; c];

    for ch in 0..c {
        let mean = if training {
            let mut s = 0.0f32;
            for batch in 0..n {
                for s_idx in 0..spatial {
                    s += src[batch * c * spatial + ch * spatial + s_idx];
                }
            }
            s / num_per_channel as f32
        } else {
            running_stats
                .as_ref()
                .map(|rs| rs.running_mean[ch])
                .unwrap_or(0.0)
        };

        let var = if training {
            let mut v = 0.0f32;
            for batch in 0..n {
                for s_idx in 0..spatial {
                    let d = src[batch * c * spatial + ch * spatial + s_idx] - mean;
                    v += d * d;
                }
            }
            v / num_per_channel as f32
        } else {
            running_stats
                .as_ref()
                .map(|rs| rs.running_var[ch])
                .unwrap_or(1.0)
        };

        let rstd = 1.0 / (var + eps).sqrt();
        means[ch] = mean;
        rstds[ch] = rstd;

        for batch in 0..n {
            for s_idx in 0..spatial {
                let idx = batch * c * spatial + ch * spatial + s_idx;
                out[idx] = (src[idx] - mean) * rstd * g[ch] + b[ch];
            }
        }
    }

    // Update running stats in training mode
    if training {
        if let Some(rs) = running_stats {
            let momentum = rs.momentum;
            for ch in 0..c {
                rs.running_mean[ch] = (1.0 - momentum) * rs.running_mean[ch] + momentum * means[ch];
                // Unbiased variance for running stats (Bessel correction: * N/(N-1))
                let n_f = num_per_channel as f32;
                let unbias = if n_f > 1.0 { n_f / (n_f - 1.0) } else { 1.0 };
                rs.running_var[ch] = (1.0 - momentum) * rs.running_var[ch]
                    + momentum * (1.0 / rstds[ch].powi(2) - eps) * unbias;
            }
        }
    }

    let out_t = Tensor::from_vec_device(out, x.shape.clone(), x.device.clone())?;
    Ok((out_t, means, rstds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_rms_norm_unit() {
        // Input all 1s, gamma all 1s → output all 1s
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let gamma = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let y = rms_norm(&x, &gamma, 1e-5).unwrap();
        for v in y.to_vec() {
            assert!((v - 1.0).abs() < 1e-4, "rms_norm unit failed: {v}");
        }
    }

    #[test]
    fn test_group_norm_shape() {
        let x = Tensor::from_vec(vec![1.0; 8], vec![1, 4, 2]).unwrap();
        let gamma = Tensor::ones(vec![4]);
        let beta = Tensor::zeros(vec![4]);
        let y = group_norm(&x, 2, &gamma, &beta, 1e-5).unwrap();
        assert_eq!(y.shape, vec![1, 4, 2]);
    }

    #[test]
    fn test_instance_norm_zero_mean() {
        // Constant input → instance norm output should be near 0
        let x = Tensor::from_vec(vec![3.0; 6], vec![1, 2, 3]).unwrap();
        let y = instance_norm(&x, None, None, 1e-5).unwrap();
        for v in y.to_vec() {
            assert!(v.abs() < 1e-4, "instance_norm const input failed: {v}");
        }
    }
}
