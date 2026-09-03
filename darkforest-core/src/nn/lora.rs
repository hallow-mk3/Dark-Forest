//! LoRA: Low-Rank Adaptation for Parameter-Efficient Fine-Tuning.
//!
//! Replaces or wraps linear layers with low-rank decomposition:
//!     W' = W_base + (lora_B * lora_A) * (scaling)
//!
//! where:
//!   - W_base: frozen base weights [out_features, in_features]
//!   - lora_A: [rank, in_features] initialized with Gaussian ~ N(0, 1/rank)
//!   - lora_B: [out_features, rank] initialized with zeros (so initially delta = 0)
//!   - scaling = alpha / rank

use crate::autograd::Value;
use crate::nn::Linear;
use crate::tensor::Tensor;
use anyhow::Result;

#[derive(Clone, Debug)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
        }
    }
}

pub struct LoRALinear {
    pub base: Linear,
    pub lora_a: Value, // [rank, in_features]
    pub lora_b: Value, // [out_features, rank]
    pub rank: usize,
    pub alpha: f32,
    pub scaling: f32,
    pub enabled: bool,
}

impl LoRALinear {
    /// Wrap an existing linear layer with LoRA adapters.
    pub fn from_linear(base: Linear, config: LoRAConfig) -> Self {
        let rank = config.rank;
        let alpha = config.alpha;
        let scaling = if rank > 0 { alpha / rank as f32 } else { 1.0 };

        let in_features = base.in_features;
        let out_features = base.out_features;

        // Kaiming / normal init for A
        let std_a = 1.0 / (rank as f32).sqrt();
        let lora_a_tensor = Tensor::randn(vec![rank, in_features], std_a);
        let lora_a = Value::leaf(lora_a_tensor);

        // Zero init for B so LoRA starts as an exact identity
        let lora_b_tensor = Tensor::zeros(vec![out_features, rank]);
        let lora_b = Value::leaf(lora_b_tensor);

        Self {
            base,
            lora_a,
            lora_b,
            rank,
            alpha,
            scaling,
            enabled: true,
        }
    }

    /// Construct a new LoRALinear layer from scratch.
    pub fn new(in_features: usize, out_features: usize, bias: bool, config: LoRAConfig) -> Self {
        let base = Linear::new(in_features, out_features, bias);
        Self::from_linear(base, config)
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        let base_out = self.base.forward(x)?;

        if !self.enabled || self.rank == 0 {
            return Ok(base_out);
        }

        // Low-rank forward path: (x * A^T) * B^T * scaling
        let a_t = Value::leaf(self.lora_a.tensor().transpose_last_two()?);
        let b_t = Value::leaf(self.lora_b.tensor().transpose_last_two()?);

        let xa = x.matmul(&a_t)?;
        let xab = xa.matmul(&b_t)?;

        // Scale and add residual to base linear output
        let lora_out = xab.scale(self.scaling)?;

        base_out.add(&lora_out)
    }

    /// Return only trainable LoRA parameters (A and B). Base weights remain frozen.
    pub fn trainable_parameters(&self) -> Vec<Value> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    /// Return all parameters (base + LoRA).
    pub fn all_parameters(&self) -> Vec<Value> {
        let mut params = self.base.parameters();
        params.push(self.lora_a.clone());
        params.push(self.lora_b.clone());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_initial_zero_delta() {
        let lora = LoRALinear::new(
            32,
            64,
            true,
            LoRAConfig {
                rank: 4,
                alpha: 8.0,
                dropout: 0.0,
            },
        );
        let x = Value::leaf(Tensor::randn(vec![2, 32], 1.0));

        let out = lora.forward(&x).unwrap();
        let base_out = lora.base.forward(&x).unwrap();

        let out_data = out.tensor().to_vec();
        let base_data = base_out.tensor().to_vec();

        assert_eq!(out_data.len(), base_data.len());
        for (a, b) in out_data.iter().zip(base_data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "LoRA output must initially match base output"
            );
        }
    }

    #[test]
    fn test_lora_trainable_params_count() {
        let lora = LoRALinear::new(
            128,
            256,
            true,
            LoRAConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.0,
            },
        );
        let trainable = lora.trainable_parameters();
        assert_eq!(trainable.len(), 2);
        assert_eq!(trainable[0].shape(), vec![8, 128]);
        assert_eq!(trainable[1].shape(), vec![256, 8]);
    }
}
