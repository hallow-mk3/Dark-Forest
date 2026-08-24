//! Matrix multiplication: C = A × B
//!
//! Supports batched matmul: A shape [... M K], B shape [... K N] → C shape [... M N].
//! Uses a naive triple-loop for correctness; CUDA uses cuBLAS.

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};

/// Core 2D matmul: A [M×K], B [K×N] → C [M×N].
pub fn matmul2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
    c
}

/// Batched matmul:
///   A shape: [batch..., M, K]
///   B shape: [batch..., K, N]
///   C shape: [batch..., M, N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let ndim_a = a.ndim();
    let ndim_b = b.ndim();
    if ndim_a < 2 || ndim_b < 2 {
        return Err(anyhow!("matmul: inputs must be at least 2D"));
    }
    let m = a.shape[ndim_a - 2];
    let k = a.shape[ndim_a - 1];
    let k2 = b.shape[ndim_b - 2];
    let n = b.shape[ndim_b - 1];
    if k != k2 {
        return Err(anyhow!("matmul: inner dims must match, got {k} vs {k2}"));
    }
    // Compute batch size from all dims except last two
    let batch_a: usize = a.shape[..ndim_a - 2].iter().product::<usize>().max(1);
    let batch_b: usize = b.shape[..ndim_b - 2].iter().product::<usize>().max(1);
    if batch_a != batch_b {
        return Err(anyhow!(
            "matmul: batch dims mismatch: {batch_a} vs {batch_b}"
        ));
    }
    let batch = batch_a;

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut out_data = Vec::with_capacity(batch * m * n);
    for bi in 0..batch {
        let a_slice = &a_vec[bi * m * k..(bi + 1) * m * k];
        let b_slice = &b_vec[bi * k * n..(bi + 1) * k * n];
        out_data.extend_from_slice(&matmul2d(a_slice, b_slice, m, k, n));
    }

    let mut out_shape = a.shape[..ndim_a - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    Tensor::from_vec_device(out_data, out_shape, a.device.clone())
}

/// Backward of batched matmul.
///
/// Given grad_output of shape [batch, M, N]:
///   grad_A = grad_output × B^T  → shape [batch, M, K]
///   grad_B = A^T × grad_output  → shape [batch, K, N]
pub fn matmul_backward(
    grad_out: &[f32],
    a_data: &[f32],
    b_data: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut grad_a = vec![0.0f32; batch * m * k];
    let mut grad_b = vec![0.0f32; batch * k * n];

    for bi in 0..batch {
        let g_off = bi * m * n;
        let a_off = bi * m * k;
        let b_off = bi * k * n;

        let g = &grad_out[g_off..g_off + m * n];
        let a = &a_data[a_off..a_off + m * k];
        let b = &b_data[b_off..b_off + k * n];
        let ga = &mut grad_a[a_off..a_off + m * k];
        let gb = &mut grad_b[b_off..b_off + k * n];

        // grad_A = grad_out × B^T  [M×N] × [N×K] = [M×K]
        for i in 0..m {
            for j in 0..k {
                let mut s = 0.0f32;
                for l in 0..n {
                    s += g[i * n + l] * b[j * n + l];
                }
                ga[i * k + j] = s;
            }
        }

        // grad_B = A^T × grad_out  [K×M] × [M×N] = [K×N]
        for i in 0..k {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..m {
                    s += a[l * k + i] * g[l * n + j];
                }
                gb[i * n + j] = s;
            }
        }
    }

    (grad_a, grad_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        // [2×3] × [3×2] → [2×2]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        for (a, b) in c.to_vec().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn test_matmul_backward_shapes() {
        let (ga, gb) = matmul_backward(
            &vec![1.0; 2 * 2],
            &vec![1.0; 2 * 3],
            &vec![1.0; 3 * 2],
            1,
            2,
            3,
            2,
        );
        assert_eq!(ga.len(), 2 * 3);
        assert_eq!(gb.len(), 3 * 2);
    }
}
