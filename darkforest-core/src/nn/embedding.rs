//! Token embedding + sinusoidal positional embedding.

use crate::autograd::Value;
use crate::tensor::Tensor;
use anyhow::Result;

pub struct Embedding {
    pub weight: Value, // shape [vocab_size, embed_dim]
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let std = (embed_dim as f32).powf(-0.5);
        let w = Tensor::randn(vec![vocab_size, embed_dim], std);
        Embedding {
            weight: Value::leaf(w),
            vocab_size,
            embed_dim,
        }
    }

    pub fn forward(&self, indices: &[usize]) -> Result<Value> {
        Value::embedding_lookup(&self.weight, indices)
    }

    pub fn to_device(&mut self, device: crate::tensor::Device) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Value> {
        vec![self.weight.clone()]
    }
}

/// Sinusoidal positional embedding (non-learned, Vaswani et al. 2017).
/// Returns [seq_len, embed_dim] tensor.
pub fn sinusoidal_pos_embedding(seq_len: usize, embed_dim: usize) -> Tensor {
    let mut data = vec![0.0f32; seq_len * embed_dim];
    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let angle = pos as f32 / 10000.0f32.powf(2.0 * (i / 2) as f32 / embed_dim as f32);
            data[pos * embed_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::from_vec(data, vec![seq_len, embed_dim]).unwrap()
}

/// Learned positional embedding table.
pub struct PosEmbedding {
    pub weight: Value, // shape [max_seq_len, embed_dim]
    pub max_seq_len: usize,
    pub embed_dim: usize,
}

impl PosEmbedding {
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        let std = (embed_dim as f32).powf(-0.5);
        let w = Tensor::randn(vec![max_seq_len, embed_dim], std);
        PosEmbedding {
            weight: Value::leaf(w),
            max_seq_len,
            embed_dim,
        }
    }

    pub fn forward(&self, seq_len: usize) -> Result<Value> {
        let indices: Vec<usize> = (0..seq_len).collect();
        Value::embedding_lookup(&self.weight, &indices)
    }

    pub fn to_device(&mut self, device: crate::tensor::Device) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Value> {
        vec![self.weight.clone()]
    }
}
