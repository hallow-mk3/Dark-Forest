//! Multi-head self-attention (CPU, correctness-first implementation).
//!
//! This implements standard scaled dot-product attention:
//!   Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V
//!
//! The fused FlashAttention-style CUDA kernel replaces this in Milestone 4.

use crate::autograd::Value;
use crate::nn::linear::Linear;
use crate::tensor::Tensor;
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize, // d_model / n_heads
    #[cfg(feature = "cuda")]
    pub cuda_context: Arc<Mutex<Option<Arc<darkforest_cuda::AttentionContext>>>>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_head = d_model / n_heads;
        MultiHeadAttention {
            q_proj: Linear::new(d_model, d_model, true),
            k_proj: Linear::new(d_model, d_model, true),
            v_proj: Linear::new(d_model, d_model, true),
            out_proj: Linear::new(d_model, d_model, true),
            n_heads,
            d_model,
            d_head,
            #[cfg(feature = "cuda")]
            cuda_context: Arc::new(Mutex::new(None)),
        }
    }

    /// Forward: x shape [seq_len, d_model] → output shape [seq_len, d_model]
    ///
    /// For simplicity in Phase 1, we process one sequence at a time (batch=1).
    /// Causal mask (lower triangular) is applied for autoregressive training.
    pub fn forward(&self, x: &Value, seq_len: usize) -> Result<Value> {
        // Project Q, K, V: each [seq_len, d_model]
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // For now, operate on all heads together (no explicit per-head split in Value).
        // We do the attention per-head by reshaping internally.
        let scale = (self.d_head as f32).powf(-0.5);

        #[cfg(feature = "cuda")]
        if false && x.tensor().is_cuda() && self.d_model <= 128 {
            // The custom CUDA attention path still requires additional stabilization
            // in the allocator/GEMM backend. Keep the numerically correct CPU path as
            // the active implementation until the device-side path is proven stable.
            let context = {
                let mut context_guard = self.cuda_context.lock().unwrap();
                *context_guard = Some(Arc::new(darkforest_cuda::AttentionContext::new(
                    1,
                    seq_len,
                    self.d_head,
                )?));

                context_guard.as_ref().unwrap().clone()
            };
            return self.out_proj.forward(&q.cuda_attention(
                &k,
                &v,
                context,
                seq_len,
                self.d_head,
                scale,
                true,
            )?);
        }

        // Compute attention scores: [seq_len, seq_len] per head
        // Then apply causal mask and softmax, then weighted sum V.
        //
        // Full multi-head split: reshape [seq_len, d_model] → [n_heads, seq_len, d_head]
        // This requires a more elaborate reshape/transpose chain.
        // Phase 1 correctness-first: use a single-head equivalent (full d_model as d_k)
        // The fused CUDA kernel (M4) will do proper multi-head.

        // --- single-head attention (Phase 1 CPU baseline) ---
        // scores = Q · K^T / sqrt(d_model): [seq_len, seq_len]
        let k_t = {
            let kt = k.tensor().transpose_last_two()?;
            Value::leaf(kt)
        };
        let scores_raw = q.matmul(&k_t)?;
        let scores_scaled = scores_raw.scale(scale)?;

        // Apply causal mask: set upper triangle to -inf before softmax
        let scores_masked = apply_causal_mask(&scores_scaled, seq_len)?;

        // Softmax over key dimension
        let attn_weights = scores_masked.softmax()?;

        // Weighted sum of V: [seq_len, d_model]
        let attn_out = attn_weights.matmul(&v)?;

        // Output projection
        self.out_proj.forward(&attn_out)
    }

    pub fn to_device(&mut self, device: crate::tensor::Device) -> Result<()> {
        self.q_proj.to_device(device.clone())?;
        self.k_proj.to_device(device.clone())?;
        self.v_proj.to_device(device.clone())?;
        self.out_proj.to_device(device)?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.out_proj.parameters());
        p
    }
}

/// Apply autoregressive causal mask (upper triangle → -1e9) without forcing a GPU->CPU copy.
fn apply_causal_mask(scores: &Value, seq_len: usize) -> Result<Value> {
    let device = scores.tensor().device.clone();
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = -1e9;
        }
    }

    let mask_tensor = Tensor::from_vec_device(mask, vec![seq_len, seq_len], device)?;
    // This is a non-differentiable mask application (constant mask).
    // Gradient flows through the unmasked entries only — correct behavior.
    Ok(scores.add(&Value::leaf(mask_tensor))?)
}
