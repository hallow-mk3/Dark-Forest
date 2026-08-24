//! Activation functions — full PyTorch parity.
//!
//! Includes: GELU, ReLU, Sigmoid, Tanh, SiLU/Swish, LeakyReLU, ELU,
//! SELU, Hardswish, Mish, Softplus, LogSigmoid — each with a backward pass.

use crate::tensor::Tensor;
use anyhow::Result;

/// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// This is the exact variant used in GPT-2.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    use std::f32::consts::PI;
    let c = (2.0f32 / PI).sqrt();
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| {
            let inner = c * (v + 0.044715 * v.powi(3));
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

/// GELU backward (via chain rule on the closed-form approximation).
pub fn gelu_backward(grad_out: &[f32], x_data: &[f32]) -> Vec<f32> {
    use std::f32::consts::PI;
    let c = (2.0f32 / PI).sqrt();
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| {
            let inner = c * (x + 0.044715 * x.powi(3));
            let tanh_v = inner.tanh();
            let sech2 = 1.0 - tanh_v * tanh_v;
            let d_inner_dx = c * (1.0 + 3.0 * 0.044715 * x.powi(2));
            let dy_dx = 0.5 * (1.0 + tanh_v) + 0.5 * x * sech2 * d_inner_dx;
            g * dy_dx
        })
        .collect()
}

/// ReLU activation.
pub fn relu(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x.to_vec().iter().map(|&v| v.max(0.0)).collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn relu_backward(grad_out: &[f32], x_data: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
        .collect()
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

/// sigmoid backward: g * σ(x) * (1 - σ(x))
pub fn sigmoid_backward(grad_out: &[f32], output: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(output.iter())
        .map(|(&g, &s)| g * s * (1.0 - s))
        .collect()
}

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------
pub fn tanh_act(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x.to_vec().iter().map(|&v| v.tanh()).collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

/// tanh backward: g * (1 - tanh²(x))
pub fn tanh_backward(grad_out: &[f32], output: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(output.iter())
        .map(|(&g, &t)| g * (1.0 - t * t))
        .collect()
}

// ---------------------------------------------------------------------------
// SiLU / Swish: x * sigmoid(x)
// ---------------------------------------------------------------------------
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| v / (1.0 + (-v).exp()))
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

/// silu backward: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
pub fn silu_backward(grad_out: &[f32], x_data: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| {
            let s = 1.0 / (1.0 + (-x).exp());
            g * (s + x * s * (1.0 - s))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LeakyReLU: max(0, x) + negative_slope * min(0, x)
// ---------------------------------------------------------------------------
pub fn leaky_relu(x: &Tensor, negative_slope: f32) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| if v >= 0.0 { v } else { negative_slope * v })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn leaky_relu_backward(grad_out: &[f32], x_data: &[f32], negative_slope: f32) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| if x >= 0.0 { g } else { g * negative_slope })
        .collect()
}

// ---------------------------------------------------------------------------
// ELU: x if x >= 0, else alpha * (exp(x) - 1)
// ---------------------------------------------------------------------------
pub fn elu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn elu_backward(grad_out: &[f32], x_data: &[f32], alpha: f32) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| {
            if x >= 0.0 {
                g
            } else {
                g * alpha * x.exp()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SELU: scale * ELU(x, alpha) with fixed constants
// ---------------------------------------------------------------------------
const SELU_SCALE: f32 = 1.0507009873554804934193349852946;
const SELU_ALPHA: f32 = 1.6732631921767541478979700699423;

pub fn selu(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| {
            if v >= 0.0 {
                SELU_SCALE * v
            } else {
                SELU_SCALE * SELU_ALPHA * (v.exp() - 1.0)
            }
        })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn selu_backward(grad_out: &[f32], x_data: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| {
            if x >= 0.0 {
                g * SELU_SCALE
            } else {
                g * SELU_SCALE * SELU_ALPHA * x.exp()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hardswish: 0 if x <= -3, x if x >= 3, x * (x + 3) / 6 otherwise
// ---------------------------------------------------------------------------
pub fn hardswish(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| {
            if v <= -3.0 {
                0.0
            } else if v >= 3.0 {
                v
            } else {
                v * (v + 3.0) / 6.0
            }
        })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn hardswish_backward(grad_out: &[f32], x_data: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| {
            if x <= -3.0 {
                0.0
            } else if x >= 3.0 {
                g
            } else {
                g * (2.0 * x + 3.0) / 6.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
// ---------------------------------------------------------------------------
pub fn mish(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| {
            let sp = (1.0 + v.exp()).ln();
            v * sp.tanh()
        })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn mish_backward(grad_out: &[f32], x_data: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| {
            let sp = (1.0 + x.exp()).ln();
            let tanh_sp = sp.tanh();
            let sigma = 1.0 / (1.0 + (-x).exp());
            let sech2 = 1.0 - tanh_sp * tanh_sp;
            g * (tanh_sp + x * sech2 * sigma)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Softplus: ln(1 + exp(x))  (numerically stable)
// ---------------------------------------------------------------------------
pub fn softplus(x: &Tensor, beta: f32, threshold: f32) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| {
            if v * beta > threshold {
                v
            } else {
                (1.0 + (beta * v).exp()).ln() / beta
            }
        })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn softplus_backward(grad_out: &[f32], x_data: &[f32], beta: f32, threshold: f32) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| {
            if x * beta > threshold {
                g
            } else {
                g / (1.0 + (-beta * x).exp())
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LogSigmoid: log(sigmoid(x)) = -softplus(-x)
// ---------------------------------------------------------------------------
pub fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x
        .to_vec()
        .iter()
        .map(|&v| {
            if v >= 0.0 {
                -(-v).exp().ln_1p()
            } else {
                v - (1.0 + v.exp()).ln()
            }
        })
        .collect();
    Tensor::from_vec_device(data, x.shape.clone(), x.device.clone())
}

pub fn log_sigmoid_backward(grad_out: &[f32], x_data: &[f32]) -> Vec<f32> {
    grad_out
        .iter()
        .zip(x_data.iter())
        .map(|(&g, &x)| g * (1.0 / (1.0 + x.exp())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_positive_zero() {
        let x = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let y = gelu(&x).unwrap();
        // gelu(0) should be 0
        assert!((y.get(0)).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        // gelu is positive for positive inputs
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let y = gelu(&x).unwrap();
        for v in y.to_vec() {
            assert!(v > 0.0, "gelu({v}) should be positive");
        }
    }

    #[test]
    fn test_relu_backward() {
        let g = vec![1.0, 1.0, 1.0];
        let x = vec![-1.0, 0.0, 1.0];
        let grad = relu_backward(&g, &x);
        assert_eq!(grad, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_range() {
        let x = Tensor::from_vec(vec![-10.0, 0.0, 10.0], vec![3]).unwrap();
        let y = sigmoid(&x).unwrap();
        let v = y.to_vec();
        assert!(v[0] > 0.0 && v[0] < 0.01);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!(v[2] > 0.99 && v[2] < 1.0);
    }

    #[test]
    fn test_tanh_range() {
        let x = Tensor::from_vec(vec![-5.0, 0.0, 5.0], vec![3]).unwrap();
        let y = tanh_act(&x).unwrap();
        let v = y.to_vec();
        assert!(v[0] < -0.99);
        assert!(v[1].abs() < 1e-6);
        assert!(v[2] > 0.99);
    }

    #[test]
    fn test_silu_positive_input() {
        let x = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let y = silu(&x).unwrap();
        assert!((y.get(0) - 0.7310).abs() < 1e-3);
    }

    #[test]
    fn test_elu_negative() {
        let x = Tensor::from_vec(vec![-1.0], vec![1]).unwrap();
        let y = elu(&x, 1.0).unwrap();
        assert!((y.get(0) - ((-1.0f32).exp() - 1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_mish_zero() {
        let x = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let y = mish(&x).unwrap();
        assert!(y.get(0).abs() < 1e-6);
    }
}
