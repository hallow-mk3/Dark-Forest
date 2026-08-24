//! Layer Normalization forward and backward.
//!
//! LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
//!
//! Applied over the last dimension (features).
//! gamma (scale) and beta (bias) are learned parameters.

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};

pub const LN_EPS: f32 = 1e-5;

/// LayerNorm forward.
///
/// x:     shape [batch..., features]
/// gamma: shape [features]
/// beta:  shape [features]
///
/// Returns (output, mean, rstd) — mean and rstd are saved for backward.
pub fn layernorm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Vec<f32>, Vec<f32>)> {
    let features = *x
        .shape
        .last()
        .ok_or_else(|| anyhow!("layernorm: empty tensor"))?;
    if gamma.shape != vec![features] || beta.shape != vec![features] {
        return Err(anyhow!(
            "layernorm: gamma/beta shape mismatch. Expected [{features}], got {:?}/{:?}",
            gamma.shape,
            beta.shape
        ));
    }
    let batch: usize = x.shape[..x.ndim() - 1].iter().product::<usize>().max(1);

    let x_data = x.to_vec();
    let g_data = gamma.to_vec();
    let b_data = beta.to_vec();
    let mut out = vec![0.0f32; x.numel()];
    let mut means = vec![0.0f32; batch];
    let mut rstds = vec![0.0f32; batch];

    for bi in 0..batch {
        let off = bi * features;
        let row = &x_data[off..off + features];

        let mean = row.iter().sum::<f32>() / features as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / features as f32;
        let rstd = 1.0 / (var + LN_EPS).sqrt();

        means[bi] = mean;
        rstds[bi] = rstd;

        for (i, &xi) in row.iter().enumerate() {
            let xnorm = (xi - mean) * rstd;
            out[off + i] = g_data[i] * xnorm + b_data[i];
        }
    }

    let out_tensor = Tensor::from_vec_device(out, x.shape.clone(), x.device.clone())?;
    Ok((out_tensor, means, rstds))
}

/// LayerNorm backward.
///
/// Returns (grad_x, grad_gamma, grad_beta).
pub fn layernorm_backward(
    grad_out: &[f32],
    x_data: &[f32],
    gamma: &[f32],
    means: &[f32],
    rstds: &[f32],
    batch: usize,
    features: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut grad_x = vec![0.0f32; batch * features];
    let mut grad_gamma = vec![0.0f32; features];
    let mut grad_beta = vec![0.0f32; features];

    for bi in 0..batch {
        let off = bi * features;
        let mean = means[bi];
        let rstd = rstds[bi];

        let xhat: Vec<f32> = (0..features)
            .map(|i| (x_data[off + i] - mean) * rstd)
            .collect();
        let dout = &grad_out[off..off + features];

        // Accumulate grad_gamma, grad_beta
        for i in 0..features {
            grad_gamma[i] += dout[i] * xhat[i];
            grad_beta[i] += dout[i];
        }

        // grad_x via closed-form (see e.g. Karpathy's layernorm backward derivation)
        // dx = (1/N) * rstd * (N * dout_scaled - sum(dout_scaled) - xhat * sum(dout_scaled * xhat))
        let dout_scaled: Vec<f32> = (0..features).map(|i| dout[i] * gamma[i]).collect();
        let sum_ds: f32 = dout_scaled.iter().sum();
        let sum_dsx: f32 = dout_scaled
            .iter()
            .zip(xhat.iter())
            .map(|(d, x)| d * x)
            .sum();

        for i in 0..features {
            grad_x[off + i] = rstd / features as f32
                * (features as f32 * dout_scaled[i] - sum_ds - xhat[i] * sum_dsx);
        }
    }

    (grad_x, grad_gamma, grad_beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_normalizes() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let gamma = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let beta = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3]).unwrap();
        let (out, means, rstds) = layernorm(&x, &gamma, &beta).unwrap();
        let data = out.to_vec();
        // Each row should have mean≈0, std≈1
        for bi in 0..2 {
            let off = bi * 3;
            let m: f32 = data[off..off + 3].iter().sum::<f32>() / 3.0;
            assert!(m.abs() < 1e-5, "row {bi} mean = {m}");
        }
    }

    #[test]
    fn test_layernorm_backward_shape() {
        let (grad_x, grad_g, grad_b) = layernorm_backward(
            &vec![1.0; 6],
            &vec![1.0; 6],
            &vec![1.0; 3],
            &vec![2.0; 2],
            &vec![0.5; 2],
            2,
            3,
        );
        assert_eq!(grad_x.len(), 6);
        assert_eq!(grad_g.len(), 3);
        assert_eq!(grad_b.len(), 3);
    }
}
