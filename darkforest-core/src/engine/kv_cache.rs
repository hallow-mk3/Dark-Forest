//! Pre-allocated inference Key-Value cache for autoregressive generative decoding.

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
        if max_batch_size == 0 || max_seq_len == 0 || n_layers == 0 || n_heads == 0 || d_head == 0 {
            return Err(anyhow!("KVCache dimensions must all be greater than zero"));
        }
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

    /// Update the KV cache for a given layer at the next sequential token step.
    pub fn update(
        &mut self,
        layer_idx: usize,
        batch_idx: usize,
        k_step: &Tensor, // [n_heads, d_head]
        v_step: &Tensor, // [n_heads, d_head]
    ) -> Result<()> {
        if batch_idx >= self.max_batch_size {
            return Err(anyhow!(
                "batch_idx {} out of bounds ({})",
                batch_idx,
                self.max_batch_size
            ));
        }
        let pos = self.seq_lens[batch_idx];
        self.append_step(layer_idx, batch_idx, k_step, v_step, pos)?;

        if layer_idx == self.n_layers - 1 {
            self.seq_lens[batch_idx] += 1;
        }

        Ok(())
    }

    /// Append single-token Key and Value vectors at an explicit position `pos`.
    pub fn append_step(
        &mut self,
        layer_idx: usize,
        batch_idx: usize,
        k_step: &Tensor,
        v_step: &Tensor,
        pos: usize,
    ) -> Result<()> {
        if layer_idx >= self.n_layers {
            return Err(anyhow!(
                "layer_idx {} out of bounds ({})",
                layer_idx,
                self.n_layers
            ));
        }
        if batch_idx >= self.max_batch_size {
            return Err(anyhow!(
                "batch_idx {} out of bounds ({})",
                batch_idx,
                self.max_batch_size
            ));
        }
        if pos >= self.max_seq_len {
            return Err(anyhow!(
                "pos {} exceeds max_seq_len ({}) for KV cache",
                pos,
                self.max_seq_len
            ));
        }

        let k_data = k_step.to_vec();
        let v_data = v_step.to_vec();
        let expected_numel = self.n_heads * self.d_head;
        if k_data.len() != expected_numel || v_data.len() != expected_numel {
            return Err(anyhow!(
                "step tensor size mismatch: expected [n_heads * d_head = {}], got k={}, v={}",
                expected_numel,
                k_data.len(),
                v_data.len()
            ));
        }

        // Stride calculations
        let head_stride = self.max_seq_len * self.d_head;
        let batch_stride = self.n_heads * head_stride;

        let mut k_guard = self.k_cache[layer_idx].data_mut();
        let mut v_guard = self.v_cache[layer_idx].data_mut();
        match (&mut k_guard.storage, &mut v_guard.storage) {
            (crate::tensor::Storage::Cpu(k_arr), crate::tensor::Storage::Cpu(v_arr)) => {
                for h in 0..self.n_heads {
                    let offset = batch_idx * batch_stride + h * head_stride + pos * self.d_head;
                    let step_offset = h * self.d_head;
                    k_arr[offset..offset + self.d_head]
                        .copy_from_slice(&k_data[step_offset..step_offset + self.d_head]);
                    v_arr[offset..offset + self.d_head]
                        .copy_from_slice(&v_data[step_offset..step_offset + self.d_head]);
                }
            }
            #[cfg(feature = "cuda")]
            _ => {
                drop(k_guard);
                drop(v_guard);
                for h in 0..self.n_heads {
                    let offset = batch_idx * batch_stride + h * head_stride + pos * self.d_head;
                    for d in 0..self.d_head {
                        let val_k = k_data[h * self.d_head + d];
                        let val_v = v_data[h * self.d_head + d];
                        self.k_cache[layer_idx].set(offset + d, val_k);
                        self.v_cache[layer_idx].set(offset + d, val_v);
                    }
                }
            }
        }

        if pos >= self.seq_lens[batch_idx] {
            self.seq_lens[batch_idx] = pos + 1;
        }

        Ok(())
    }

    /// Return current sequence length for batch item `batch_idx`.
    #[inline]
    pub fn current_len(&self, batch_idx: usize) -> Result<usize> {
        self.seq_lens
            .get(batch_idx)
            .copied()
            .ok_or_else(|| anyhow!("batch_idx {} out of bounds ({})", batch_idx, self.max_batch_size))
    }

    /// Return true if no tokens have been cached for `batch_idx`.
    #[inline]
    pub fn is_empty(&self, batch_idx: usize) -> bool {
        self.seq_lens.get(batch_idx).map(|&l| l == 0).unwrap_or(true)
    }

    /// Access the key tensor for `layer_idx`.
    #[inline]
    pub fn k_cache(&self, layer_idx: usize) -> Result<&Tensor> {
        self.k_cache
            .get(layer_idx)
            .ok_or_else(|| anyhow!("layer_idx {} out of bounds ({})", layer_idx, self.n_layers))
    }

    /// Access the value tensor for `layer_idx`.
    #[inline]
    pub fn v_cache(&self, layer_idx: usize) -> Result<&Tensor> {
        self.v_cache
            .get(layer_idx)
            .ok_or_else(|| anyhow!("layer_idx {} out of bounds ({})", layer_idx, self.n_layers))
    }

    /// Extract key cache slice for `batch_idx` across all heads up to `current_len`.
    /// Returns Tensor shaped [n_heads, current_len, d_head].
    pub fn get_k_slice(&self, layer_idx: usize, batch_idx: usize) -> Result<Tensor> {
        self.extract_slice(&self.k_cache[layer_idx], batch_idx)
    }

    /// Extract value cache slice for `batch_idx` across all heads up to `current_len`.
    /// Returns Tensor shaped [n_heads, current_len, d_head].
    pub fn get_v_slice(&self, layer_idx: usize, batch_idx: usize) -> Result<Tensor> {
        self.extract_slice(&self.v_cache[layer_idx], batch_idx)
    }

    fn extract_slice(&self, cache: &Tensor, batch_idx: usize) -> Result<Tensor> {
        let cur_len = self.current_len(batch_idx)?;
        if cur_len == 0 {
            return Err(anyhow!("Cannot extract slice from empty KV cache for batch {}", batch_idx));
        }

        let head_stride = self.max_seq_len * self.d_head;
        let batch_stride = self.n_heads * head_stride;
        let mut out = Vec::with_capacity(self.n_heads * cur_len * self.d_head);

        let guard = cache.data();
        match &guard.storage {
            crate::tensor::Storage::Cpu(v) => {
                for h in 0..self.n_heads {
                    let base = batch_idx * batch_stride + h * head_stride;
                    for s in 0..cur_len {
                        let row_start = base + s * self.d_head;
                        out.extend_from_slice(&v[row_start..row_start + self.d_head]);
                    }
                }
            }
            #[cfg(feature = "cuda")]
            crate::tensor::Storage::Cuda(_) => {
                drop(guard);
                for h in 0..self.n_heads {
                    let base = batch_idx * batch_stride + h * head_stride;
                    for s in 0..cur_len {
                        for d in 0..self.d_head {
                            out.push(cache.get(base + s * self.d_head + d));
                        }
                    }
                }
            }
        }

        Tensor::from_vec(out, vec![self.n_heads, cur_len, self.d_head])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_initialization_and_update() {
        let max_batch = 2;
        let max_seq = 8;
        let n_layers = 2;
        let n_heads = 2;
        let d_head = 4;

        let mut cache = KVCache::new(max_batch, max_seq, n_layers, n_heads, d_head, Device::Cpu).unwrap();
        assert_eq!(cache.current_len(0).unwrap(), 0);
        assert!(cache.is_empty(0));

        let k_step = Tensor::from_vec(vec![1.0; n_heads * d_head], vec![n_heads, d_head]).unwrap();
        let v_step = Tensor::from_vec(vec![2.0; n_heads * d_head], vec![n_heads, d_head]).unwrap();

        // Update layer 0 then layer 1
        cache.update(0, 0, &k_step, &v_step).unwrap();
        assert_eq!(cache.current_len(0).unwrap(), 0); // only increments on last layer

        cache.update(1, 0, &k_step, &v_step).unwrap();
        assert_eq!(cache.current_len(0).unwrap(), 1);
        assert!(!cache.is_empty(0));

        // Check slice extraction
        let k_slice = cache.get_k_slice(0, 0).unwrap();
        assert_eq!(k_slice.shape(), &vec![n_heads, 1, d_head]);
        assert_eq!(k_slice.get(0), 1.0);

        let v_slice = cache.get_v_slice(0, 0).unwrap();
        assert_eq!(v_slice.shape(), &vec![n_heads, 1, d_head]);
        assert_eq!(v_slice.get(0), 2.0);

        // Reset
        cache.reset();
        assert_eq!(cache.current_len(0).unwrap(), 0);
        assert!(cache.is_empty(0));
    }

    #[test]
    fn test_kv_cache_append_step_explicit_position() {
        let mut cache = KVCache::new(1, 4, 1, 1, 2, Device::Cpu).unwrap();
        let k_step = Tensor::from_vec(vec![42.0, 43.0], vec![1, 2]).unwrap();
        let v_step = Tensor::from_vec(vec![99.0, 100.0], vec![1, 2]).unwrap();

        cache.append_step(0, 0, &k_step, &v_step, 2).unwrap();
        assert_eq!(cache.current_len(0).unwrap(), 3);

        let k_slice = cache.get_k_slice(0, 0).unwrap();
        assert_eq!(k_slice.shape(), &vec![1, 3, 2]);
        assert_eq!(k_slice.get(4), 42.0);
        assert_eq!(k_slice.get(5), 43.0);
    }
}
