//! Element-wise addition (with broadcasting along batch dims).

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};

/// Element-wise addition: out = a + b (must have identical shapes).
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(anyhow!(
            "add: shape mismatch {:?} vs {:?}",
            a.shape,
            b.shape
        ));
    }
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let out_data: Vec<f32> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x + y).collect();
    Tensor::from_vec_device(out_data, a.shape.clone(), a.device.clone())
}

/// Backward pass for addition.
///
/// grad_output has the same shape as the forward output.
/// Both input gradients equal grad_output (addition is linear).
///
/// Returns (grad_a, grad_b).
pub fn add_backward(grad_output: &[f32]) -> (Vec<f32>, Vec<f32>) {
    (grad_output.to_vec(), grad_output.to_vec())
}

/// Add a bias vector (shape [features]) to every row of input (shape [batch, features]).
/// Used for linear layer bias.
pub fn add_bias(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if input.ndim() < 1 || bias.ndim() != 1 {
        return Err(anyhow!("add_bias: input must be ≥1D and bias must be 1D"));
    }
    let features = *input.shape.last().unwrap();
    if bias.shape[0] != features {
        return Err(anyhow!(
            "add_bias: last dim {} != bias len {}",
            features,
            bias.shape[0]
        ));
    }
    let n = input.numel();
    let in_vec = input.to_vec();
    let b_vec = bias.to_vec();
    let mut out = vec![0.0f32; n];
    for (i, val) in in_vec.iter().enumerate() {
        out[i] = val + b_vec[i % features];
    }
    Tensor::from_vec_device(out, input.shape.clone(), input.device.clone())
}

/// Backward of add_bias: grad_input = grad_output, grad_bias = sum over batch.
pub fn add_bias_backward(
    grad_output: &[f32],
    _batch_size: usize,
    features: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut grad_bias = vec![0.0f32; features];
    for (i, &g) in grad_output.iter().enumerate() {
        grad_bias[i % features] += g;
    }
    (grad_output.to_vec(), grad_bias)
}

/// Scalar multiply: out = alpha * a.
pub fn scale(a: &Tensor, alpha: f32) -> Result<Tensor> {
    let data: Vec<f32> = a.to_vec().iter().map(|x| x * alpha).collect();
    Tensor::from_vec_device(data, a.shape.clone(), a.device.clone())
}

/// Element-wise multiply: out = a * b.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(anyhow!(
            "mul: shape mismatch {:?} vs {:?}",
            a.shape,
            b.shape
        ));
    }
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let data: Vec<f32> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x * y).collect();
    Tensor::from_vec_device(data, a.shape.clone(), a.device.clone())
}

/// Element-wise multiply backward: grad_a = grad_out * b, grad_b = grad_out * a.
pub fn mul_backward(grad_output: &[f32], a_data: &[f32], b_data: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let grad_a: Vec<f32> = grad_output
        .iter()
        .zip(b_data.iter())
        .map(|(g, b)| g * b)
        .collect();
    let grad_b: Vec<f32> = grad_output
        .iter()
        .zip(a_data.iter())
        .map(|(g, a)| g * a)
        .collect();
    (grad_a, grad_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_forward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_backward() {
        let grad = vec![1.0, 2.0, 3.0];
        let (ga, gb) = add_backward(&grad);
        assert_eq!(ga, grad);
        assert_eq!(gb, grad);
    }
}
