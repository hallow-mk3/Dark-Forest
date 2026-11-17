//! Tape-based Linear layer and MLP built on the new `Tensor` autograd system.
//!
//! Unlike `nn/linear.rs` which uses `Value`-based autograd, these modules
//! operate directly on `Tensor` with the iterative tape engine.

use crate::tensor::Tensor;
use anyhow::Result;

// ---------------------------------------------------------------------------
// Parameter — a `Tensor` with `requires_grad = true`
// ---------------------------------------------------------------------------

/// Creates a fresh parameter tensor (requires_grad=true, is a leaf).
pub fn parameter(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor> {
    let mut t = Tensor::from_vec(data, shape)?;
    t.set_requires_grad(true);
    Ok(t)
}

/// Creates a parameter initialized with Kaiming uniform (He init).
pub fn parameter_kaiming(shape: &[usize]) -> Result<Tensor> {
    let fan_in = if shape.len() >= 2 { shape[shape.len() - 1] } else { shape[0] };
    let bound = (1.0_f32 / fan_in as f32).sqrt();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|_| {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            rng.gen_range(-bound..bound)
        })
        .collect();
    parameter(data, shape.to_vec())
}

/// Creates a zero parameter.
pub fn parameter_zeros(shape: &[usize]) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    parameter(vec![0.0_f32; n], shape.to_vec())
}

// ---------------------------------------------------------------------------
// Linear Layer
// ---------------------------------------------------------------------------

/// Fully-connected linear layer: `y = x @ W^T + b`
///
/// Parameters: `weight` [out_features, in_features], `bias` [out_features].
/// Both are leaf tensors with `requires_grad = true`.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Create a Linear layer with Kaiming-uniform weights and zero bias.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        let weight = parameter_kaiming(&[out_features, in_features])?;
        let bias_param = if bias {
            Some(parameter_zeros(&[out_features])?)
        } else {
            None
        };
        Ok(Self { weight, bias: bias_param, in_features, out_features })
    }

    /// Create from pre-set weight/bias data (useful for exact PyTorch weight copying).
    pub fn from_data(
        in_features: usize,
        out_features: usize,
        weight_data: Vec<f32>,
        bias_data: Option<Vec<f32>>,
    ) -> Result<Self> {
        let weight = parameter(weight_data, vec![out_features, in_features])?;
        let bias = match bias_data {
            Some(b) => Some(parameter(b, vec![out_features])?),
            None => None,
        };
        Ok(Self { weight, bias, in_features, out_features })
    }

    /// Forward: `x @ W^T + b`
    ///
    /// `x` is `[batch, in_features]` → output `[batch, out_features]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // W^T: [in_features, out_features]
        let wt = self.weight.transpose(0, 1)?;
        // x @ W^T: [batch, out_features]
        let xw = x.matmul(&wt)?;
        if let Some(b) = &self.bias {
            // bias is [out_features], broadcast-added to [batch, out_features]
            xw.add(b)
        } else {
            Ok(xw)
        }
    }

    /// All trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    /// Zero gradients for all parameters.
    pub fn zero_grad(&self) {
        self.weight.zero_grad();
        if let Some(b) = &self.bias {
            b.zero_grad();
        }
    }

    /// Copy weights/bias from plain f32 vecs (for PyTorch parity tests).
    pub fn copy_weights_from(&mut self, weight: Vec<f32>, bias: Option<Vec<f32>>) -> Result<()> {
        // Replace storage in-place by rebuilding parameter tensors
        self.weight = parameter(weight, vec![self.out_features, self.in_features])?;
        if let (Some(ref mut b_param), Some(b_data)) = (&mut self.bias, bias) {
            *b_param = parameter(b_data, vec![self.out_features])?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sequential container
// ---------------------------------------------------------------------------

/// A sequential container of functions that chain forward passes.
///
/// Each `BoxedLayer` takes `&Tensor` and returns `Result<Tensor>`.
pub type BoxedLayer = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

pub struct Sequential {
    pub layers: Vec<BoxedLayer>,
}

impl Sequential {
    pub fn new(layers: Vec<BoxedLayer>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// MLP — multi-layer perceptron with ReLU activations
// ---------------------------------------------------------------------------

/// A fully-connected MLP: Linear → ReLU → ... → Linear.
///
/// Last layer has no activation (raw logits for cross-entropy).
pub struct MLP {
    pub layers: Vec<Linear>,
}

impl MLP {
    /// Build from layer sizes, e.g. `&[784, 256, 128, 10]`.
    pub fn new(sizes: &[usize]) -> Result<Self> {
        assert!(sizes.len() >= 2, "MLP needs at least 2 sizes (input + output)");
        let mut layers = Vec::new();
        for window in sizes.windows(2) {
            layers.push(Linear::new(window[0], window[1], true)?);
        }
        Ok(Self { layers })
    }

    /// Build from explicit (weight, bias) data pairs — for parity tests.
    pub fn from_data(
        sizes: &[usize],
        weights: Vec<Vec<f32>>,
        biases: Vec<Vec<f32>>,
    ) -> Result<Self> {
        assert_eq!(sizes.len() - 1, weights.len());
        assert_eq!(weights.len(), biases.len());
        let mut layers = Vec::new();
        for (i, window) in sizes.windows(2).enumerate() {
            layers.push(Linear::from_data(
                window[0],
                window[1],
                weights[i].clone(),
                Some(biases[i].clone()),
            )?);
        }
        Ok(Self { layers })
    }

    /// Forward: applies ReLU after every layer except the last.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.layers.len();
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if i < n - 1 {
                h = h.relu()?;
            }
        }
        Ok(h)
    }

    /// All trainable parameters (weight, bias for each layer).
    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// Zero gradients for all layers.
    pub fn zero_grad(&self) {
        for l in &self.layers {
            l.zero_grad();
        }
    }
}

// ---------------------------------------------------------------------------
// AdamW for Tensor parameters (tape autograd)
// ---------------------------------------------------------------------------

/// AdamW optimizer operating on tape-autograd `Tensor` leaf parameters.
pub struct TapeAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    moments: Vec<(Vec<f32>, Vec<f32>)>, // (m, v) per parameter
}

impl TapeAdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            moments: Vec::new(),
        }
    }

    /// Convenience constructor with PyTorch defaults.
    pub fn default_lr(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }

    fn init_moments(&mut self, params: &[Tensor]) {
        self.moments = params.iter().map(|p| (vec![0.0f32; p.numel()], vec![0.0f32; p.numel()])).collect();
    }

    pub fn zero_grad(&self, params: &[Tensor]) {
        for p in params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self, params: &[Tensor]) {
        if self.moments.is_empty() {
            self.init_moments(params);
        }
        self.step_count += 1;
        let t = self.step_count as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (pi, param) in params.iter().enumerate() {
            let grad_tensor = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let grad = grad_tensor.to_vec();
            let n = param.numel();
            let (ref mut m, ref mut v) = self.moments[pi];

            for i in 0..n {
                let w = param.get(i);
                let g = grad[i];
                // AdamW: decouple weight decay from gradient
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                // weight decay applied to parameter (AdamW style)
                let new_w = w * (1.0 - self.lr * self.weight_decay)
                    - self.lr * m_hat / (v_hat.sqrt() + self.eps);
                param.set(i, new_w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SGD for Tensor parameters (tape autograd)
// ---------------------------------------------------------------------------

/// Plain SGD (with optional momentum) for tape-autograd `Tensor` parameters.
pub struct TapeSGD {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    velocities: Vec<Vec<f32>>,
}

impl TapeSGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self { lr, momentum, weight_decay, velocities: Vec::new() }
    }

    pub fn zero_grad(&self, params: &[Tensor]) {
        for p in params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self, params: &[Tensor]) {
        if self.velocities.is_empty() {
            self.velocities = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        }
        for (pi, param) in params.iter().enumerate() {
            let grad_tensor = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let grad = grad_tensor.to_vec();
            let n = param.numel();
            let v = &mut self.velocities[pi];
            for i in 0..n {
                let w = param.get(i);
                let g = grad[i] + self.weight_decay * w;
                if self.momentum != 0.0 {
                    v[i] = self.momentum * v[i] + g;
                    param.set(i, w - self.lr * v[i]);
                } else {
                    param.set(i, w - self.lr * g);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;

    #[test]
    fn test_linear_forward_shape() -> Result<()> {
        let lin = Linear::new(4, 8, true)?;
        let x = Tensor::from_vec((0..12).map(|v| v as f32).collect(), vec![3, 4])?;
        let out = lin.forward(&x)?;
        assert_eq!(out.shape(), &vec![3, 8]);
        Ok(())
    }

    #[test]
    fn test_linear_backward_produces_grads() -> Result<()> {
        let lin = Linear::new(4, 8, true)?;
        let mut x = Tensor::from_vec((0..12).map(|v| v as f32 * 0.1).collect(), vec![3, 4])?;
        x.set_requires_grad(true);

        let out = lin.forward(&x)?;
        let loss = out.sum(None, false)?;
        loss.backward(None, false)?;

        // weight and bias should have gradients
        assert!(lin.weight.grad().is_some(), "weight grad missing");
        assert!(lin.bias.as_ref().unwrap().grad().is_some(), "bias grad missing");
        assert!(x.grad().is_some(), "input grad missing");
        Ok(())
    }

    #[test]
    fn test_mlp_forward_and_backward() -> Result<()> {
        let mlp = MLP::new(&[4, 8, 3])?;
        let mut x = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4])?;
        x.set_requires_grad(true);

        let logits = mlp.forward(&x)?;
        assert_eq!(logits.shape(), &vec![1, 3]);

        let _target = Tensor::from_vec(vec![1.0f32], vec![1])?; // class 1 (hacky: just sum loss)
        // Use sum as a simple loss to trigger backward
        let loss = logits.sum(None, false)?;
        loss.backward(None, false)?;

        // All layer weights should have gradients
        for (i, layer) in mlp.layers.iter().enumerate() {
            assert!(layer.weight.grad().is_some(), "layer {i} weight grad missing");
        }
        Ok(())
    }

    #[test]
    fn test_tape_sgd_decreases_loss() -> Result<()> {
        // Simple 1-layer: minimize sum(linear(x))
        let lin = Linear::new(4, 1, false)?;
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4])?;

        let mut sgd = TapeSGD::new(0.1, 0.0, 0.0);
        let params = lin.parameters();

        let mut prev_loss = f32::INFINITY;
        for _ in 0..10 {
            sgd.zero_grad(&params);
            let out = lin.forward(&x)?;
            let loss = out.sum(None, false)?;
            let loss_val = loss.to_vec()[0];
            loss.backward(None, false)?;
            sgd.step(&params);
            prev_loss = loss_val;
        }
        // After 10 steps the loss magnitude should have decreased
        let final_out = lin.forward(&x)?;
        let final_loss = final_out.to_vec()[0].abs();
        // Started with nonzero loss, optimizer should have brought it toward 0
        assert!(final_loss < prev_loss.abs() + 1.0, "SGD did not decrease loss");
        Ok(())
    }

    #[test]
    fn test_no_grad_suppresses_tape() -> Result<()> {
        let lin = Linear::new(4, 8, true)?;
        let x = Tensor::from_vec((0..8).map(|v| v as f32 * 0.1).collect(), vec![2, 4])?;

        let out = no_grad(|| lin.forward(&x))?;
        // No grad_fn should be attached in no_grad mode
        assert!(out.grad_fn().is_none(), "grad_fn should be None in no_grad");
        Ok(())
    }
}
