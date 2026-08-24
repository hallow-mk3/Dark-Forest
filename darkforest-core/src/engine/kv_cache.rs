//! Pre-allocated inference Key-Value cache for autoregressive generative decoding.

use crate::autograd::Value;
use crate::tensor::{Device, Tensor};
use anyhow::{anyhow, Result};

pub struct KVCache {
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub k_cache: Vec<Tensor>, // Per layer: [max_batch_size, n_heads, max_seq_len, d_head]
    pub v_cache: Vec<Tensor>, // Per layer: [max_batch_size, n_heads, max_seq_len, d_head]
    pub seq_lens: Vec<usize>, // Current populated sequence length per batch item
}

impl KVCache {
    pub fn new(
        max_batch_size: usize,
        max_seq_len: usize,
        n_layers: usize,
        n_heads: usize,
        d_head: usize,
        device: Device,
    ) -> Result<Self> {
        let mut k_cache = Vec::with_capacity(n_layers);
        let mut v_cache = Vec::with_capacity(n_layers);
        let shape = vec![max_batch_size, n_heads, max_seq_len, d_head];

        for _ in 0..n_layers {
            k_cache.push(Tensor::zeros_device(shape.clone(), device.clone()));
            v_cache.push(Tensor::zeros_device(shape.clone(), device.clone()));
        }

        Ok(KVCache {
            max_batch_size,
            max_seq_len,
            n_layers,
            n_heads,
            d_head,
            k_cache,
            v_cache,
            seq_lens: vec![0; max_batch_size],
        })
    }

    /// Update the KV cache for a given layer at current token step
    pub fn update(
        &mut self,
        layer_idx: usize,
        batch_idx: usize,
        k_step: &Tensor, // [n_heads, d_head]
        v_step: &Tensor, // [n_heads, d_head]
    ) -> Result<()> {
        if layer_idx >= self.n_layers {
            return Err(anyhow!("layer_idx out of bounds"));
        }
        if batch_idx >= self.max_batch_size {
            return Err(anyhow!("batch_idx out of bounds"));
        }
        let pos = self.seq_lens[batch_idx];
        if pos >= self.max_seq_len {
            return Err(anyhow!("exceeded max_seq_len for KV cache"));
        }

        let k_data = k_step.to_vec();
        let v_data = v_step.to_vec();

        // Stride calculations
        let head_stride = self.max_seq_len * self.d_head;
        let batch_stride = self.n_heads * head_stride;

        for h in 0..self.n_heads {
            let offset = batch_idx * batch_stride + h * head_stride + pos * self.d_head;
            for d in 0..self.d_head {
                let val_k = k_data[h * self.d_head + d];
                let val_v = v_data[h * self.d_head + d];
                self.k_cache[layer_idx].set(offset + d, val_k);
                self.v_cache[layer_idx].set(offset + d, val_v);
            }
        }

        if layer_idx == self.n_layers - 1 {
            self.seq_lens[batch_idx] += 1;
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        for s in &mut self.seq_lens {
            *s = 0;
        }
        for k in &mut self.k_cache {
            k.zero_grad();
        }
        for v in &mut self.v_cache {
            v.zero_grad();
        }
    }
}
