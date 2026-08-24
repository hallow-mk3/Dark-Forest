//! Numerically-stable row-wise softmax.
//!
//! softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
//!
//! The backward pass uses the Jacobian identity:
//!   ds/dx_i = s_i * (delta_ij - s_j)  →  grad_x = s * (grad_out - dot(grad_out, s))

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};

/// Row-wise softmax over the last dimension.
///
/// Input shape: [batch..., seq, vocab] (any prefix is treated as batch).
/// Output shape: same.
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let ndim = x.ndim();
    if ndim == 0 {
        return Err(anyhow!("softmax: cannot apply to scalar"));
    }
    let vocab = x.shape[ndim - 1];
    let batch: usize = x.shape[..ndim - 1].iter().product::<usize>().max(1);

    let x_vec = x.to_vec();
    let mut out = vec![0.0f32; x.numel()];

    for b in 0..batch {
        let off = b * vocab;
        let row = &x_vec[off..off + vocab];

        // numerically stable: subtract max first
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        for (i, e) in exps.iter().enumerate() {
            out[off + i] = e / sum;
        }
    }

    Tensor::from_vec_device(out, x.shape.clone(), x.device.clone())
}

/// Backward of row-wise softmax.
///
/// `softmax_out`: the saved forward output (shape = input shape).
/// `grad_output`: upstream gradient (same shape).
///
/// Returns `grad_input` of the same shape.
pub fn softmax_backward(softmax_out: &[f32], grad_output: &[f32], vocab: usize) -> Vec<f32> {
    let batch = softmax_out.len() / vocab;
    let mut grad_in = vec![0.0f32; softmax_out.len()];

    for b in 0..batch {
        let off = b * vocab;
        let s = &softmax_out[off..off + vocab];
        let g = &grad_output[off..off + vocab];

        // dot(grad_out, s) for this row
        let dot: f32 = s.iter().zip(g.iter()).map(|(si, gi)| si * gi).sum();

        for i in 0..vocab {
            grad_in[off + i] = s[i] * (g[i] - dot);
        }
    }

    grad_in
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = softmax(&x).unwrap();
        let data = s.to_vec();
        // Each row should sum to 1
        let row0: f32 = data[..3].iter().sum();
        let row1: f32 = data[3..].iter().sum();
        assert!((row0 - 1.0).abs() < 1e-6, "row0 sum = {row0}");
        assert!((row1 - 1.0).abs() < 1e-6, "row1 sum = {row1}");
    }

    #[test]
    fn test_softmax_monotone() {
        // Larger logit → larger probability
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let s = softmax(&x).unwrap();
        let d = s.to_vec();
        assert!(d[0] < d[1] && d[1] < d[2], "{:?}", d);
    }

    #[test]
    fn test_softmax_backward_shape() {
        let s = vec![0.1, 0.7, 0.2];
        let g = vec![1.0, 0.0, -1.0];
        let grad = softmax_backward(&s, &g, 3);
        assert_eq!(grad.len(), 3);
        // sum of grad_in for a given softmax should be ~0 (because of Jacobian structure)
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-5, "sum = {sum}");
    }
}
