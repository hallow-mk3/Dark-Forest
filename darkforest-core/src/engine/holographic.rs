//! Holographic Context Compaction & Continuous Associative Memory
//!
//! Provides a fixed-capacity continuous associative memory manifold based on outer-product
//! fast weights (Hebbian / Linear Attention accumulation) for autoregressive models:
//!
//! - **$O(1)$ Constant Memory**: Memory footprint is bounded by $D \times D$, regardless of context length
//! - **Associative Outer-Product Binding**: Context is accumulated into continuous matrix manifolds ($M \leftarrow \lambda M + k^T v$)
//! - **Decay & Recency Scaling**: Exponential forgetting factor $\lambda \in (0, 1]$ prioritizes recent history
//! - **Hybrid Windowing**: Maintains an exact local attention window while accumulating long-range history
//!   into the associative matrix
//!
//! Note: As with all finite-rank outer-product memory manifolds, capacity is mathematically bounded
//! by $D$; older token associations undergo continuous lossy decay.

use anyhow::{bail, Result};

/// Configuration for the Holographic Manifold Cache.
#[derive(Clone, Debug)]
pub struct HolographicConfig {
    pub d_model: usize,
    pub decay: f32,
    pub local_window_size: usize,
}

impl Default for HolographicConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            decay: 0.995,
            local_window_size: 128,
        }
    }
}

/// A continuous associative holographic memory layer.
/// Represents context as a fixed-rank continuous manifold matrix $M \in \mathbb{R}^{D \times D}$.
#[derive(Clone, Debug)]
pub struct HolographicLayerCache {
    pub d_model: usize,
    pub decay: f32,
    /// Continuous associative memory state matrix: $M \in \mathbb{R}^{D \times D}$
    pub manifold: Vec<f32>,
    /// Exact sliding window buffer for recent tokens: shape `[window, d_model]`
    pub local_keys: Vec<f32>,
    pub local_values: Vec<f32>,
    pub window_count: usize,
    pub total_tokens_compacted: usize,
}

impl HolographicLayerCache {
    pub fn new(d_model: usize, decay: f32) -> Self {
        Self {
            d_model,
            decay,
            manifold: vec![0.0f32; d_model * d_model],
            local_keys: Vec::new(),
            local_values: Vec::new(),
            window_count: 0,
            total_tokens_compacted: 0,
        }
    }

    /// Ingests a new key-value pair $(k_t, v_t)$ at timestep $t$.
    /// When the local window overflows, oldest tokens are automatically bound
    /// into the continuous associative holographic manifold via outer product:
    /// $$M \leftarrow \lambda M + k^T v$$
    pub fn push_token(&mut self, key: &[f32], value: &[f32], window_limit: usize) -> Result<()> {
        if key.len() != self.d_model || value.len() != self.d_model {
            bail!("Key/Value length {} does not match d_model {}", key.len(), self.d_model);
        }

        self.local_keys.extend_from_slice(key);
        self.local_values.extend_from_slice(value);
        self.window_count += 1;

        // If local window exceeds threshold, compact the oldest token into holographic manifold
        if self.window_count > window_limit {
            self.compact_oldest_token();
        }

        Ok(())
    }

    fn compact_oldest_token(&mut self) {
        let d = self.d_model;
        let k_old = &self.local_keys[0..d];
        let v_old = &self.local_values[0..d];

        // 1. Decay existing associative manifold: M = decay * M
        for entry in &mut self.manifold {
            *entry *= self.decay;
        }

        // 2. Holographic outer-product update: M[i, j] += k_old[i] * v_old[j]
        for i in 0..d {
            let ki = k_old[i];
            let row_offset = i * d;
            for j in 0..d {
                self.manifold[row_offset + j] += ki * v_old[j];
            }
        }

        // 3. Shift local sliding window
        self.local_keys.drain(0..d);
        self.local_values.drain(0..d);
        self.window_count -= 1;
        self.total_tokens_compacted += 1;
    }

    /// Queries the holographic manifold memory with query vector $q \in \mathbb{R}^{D}$.
    /// Retrieves continuous associative recall:
    /// $$\hat{v} = q \cdot M \in \mathbb{R}^{D}$$
    pub fn recall_holographic(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.d_model {
            bail!("Query dimension {} does not match d_model {}", query.len(), self.d_model);
        }
        let d = self.d_model;
        let mut retrieved = vec![0.0f32; d];

        for i in 0..d {
            let qi = query[i];
            let row_offset = i * d;
            for j in 0..d {
                retrieved[j] += qi * self.manifold[row_offset + j];
            }
        }

        Ok(retrieved)
    }

    /// Returns the exact memory footprint in bytes.
    /// Notice this is STRICTLY CONSTANT after reaching `window_limit` tokens!
    pub fn memory_footprint_bytes(&self) -> usize {
        let manifold_bytes = self.manifold.len() * std::mem::size_of::<f32>();
        let window_bytes = (self.local_keys.len() + self.local_values.len()) * std::mem::size_of::<f32>();
        manifold_bytes + window_bytes
    }
}

/// Multi-layer holographic context manager.
#[derive(Clone, Debug)]
pub struct HolographicCache {
    pub layers: Vec<HolographicLayerCache>,
    pub config: HolographicConfig,
}

impl HolographicCache {
    pub fn new(n_layers: usize, config: HolographicConfig) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(HolographicLayerCache::new(config.d_model, config.decay));
        }
        Self { layers, config }
    }

    pub fn push_layer_token(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        if layer >= self.layers.len() {
            bail!("Layer {} out of range", layer);
        }
        self.layers[layer].push_token(key, value, self.config.local_window_size)
    }

    pub fn recall_layer(&self, layer: usize, query: &[f32]) -> Result<Vec<f32>> {
        if layer >= self.layers.len() {
            bail!("Layer {} out of range", layer);
        }
        self.layers[layer].recall_holographic(query)
    }

    /// Total memory consumption across all transformer layers.
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_footprint_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_memory_scaling() {
        let d_model = 64;
        let window = 16;
        let mut cache = HolographicLayerCache::new(d_model, 0.99);

        // Push 200 tokens (far exceeding the 16-token window)
        for i in 0..200 {
            let key = vec![0.1 * (i as f32); d_model];
            let val = vec![0.05 * (i as f32); d_model];
            cache.push_token(&key, &val, window).unwrap();
        }

        assert_eq!(cache.window_count, window);
        assert_eq!(cache.total_tokens_compacted, 200 - window);

        // Record memory footprint after 200 tokens
        let footprint_200 = cache.memory_footprint_bytes();

        // Push 1,000 more tokens
        for _ in 200..1200 {
            let key = vec![0.1; d_model];
            let val = vec![0.05; d_model];
            cache.push_token(&key, &val, window).unwrap();
        }

        let footprint_1200 = cache.memory_footprint_bytes();

        // FOOTPRINT IS STRICTLY CONSTANT: O(1) Memory!
        assert_eq!(
            footprint_200, footprint_1200,
            "Holographic manifold cache must have O(1) constant memory scaling"
        );
    }

    #[test]
    fn test_holographic_associative_recall() {
        let d_model = 4;
        let mut cache = HolographicLayerCache::new(d_model, 1.0);

        // Bind key [1, 0, 0, 0] to value [2, 3, 4, 5]
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let val = vec![2.0, 3.0, 4.0, 5.0];
        cache.push_token(&key, &val, 0).unwrap(); // immediately compacts to manifold

        // Query with exact key
        let retrieved = cache.recall_holographic(&key).unwrap();
        assert_eq!(retrieved, val);
    }
}
