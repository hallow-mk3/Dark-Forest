//! Configuration abstractions for Dark Forest Engine.

use crate::engine::StaticLoRAConfig;
use serde::{Deserialize, Serialize};

/// High-level model and engine hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub seq_len: usize,
    pub lr: f32,
    pub max_grad_norm: Option<f32>,
    pub weight_decay: f32,
    pub lora: Option<StaticLoRAConfig>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::gpt2_124m()
    }
}

impl EngineConfig {
    /// Full GPT-2 (124M parameters) configuration.
    pub fn gpt2_124m() -> Self {
        Self {
            vocab_size: 50257,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            d_ff: 3072,
            seq_len: 128,
            lr: 3e-4,
            max_grad_norm: Some(1.0),
            weight_decay: 0.01,
            lora: None,
        }
    }

    /// Medium Transformer configuration (4 layers, d=256, 4 heads).
    pub fn small() -> Self {
        Self {
            vocab_size: 50257,
            d_model: 256,
            n_heads: 4,
            n_layers: 4,
            d_ff: 1024,
            seq_len: 128,
            lr: 1e-3,
            max_grad_norm: Some(1.0),
            weight_decay: 0.01,
            lora: None,
        }
    }

    /// Micro transformer for character-level prototyping and fast tests.
    pub fn micro(vocab_size: usize, seq_len: usize) -> Self {
        Self {
            vocab_size,
            d_model: 64,
            n_heads: 2,
            n_layers: 2,
            d_ff: 256,
            seq_len,
            lr: 1e-3,
            max_grad_norm: None,
            weight_decay: 0.0,
            lora: None,
        }
    }

    pub fn with_lora(mut self, rank: usize, alpha: f32) -> Self {
        self.lora = Some(StaticLoRAConfig::new(rank, alpha));
        self
    }

    pub fn with_qlora(mut self, rank: usize, alpha: f32) -> Self {
        self.lora = Some(StaticLoRAConfig::new(rank, alpha).with_qlora(true));
        self
    }

    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn with_seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    pub fn with_vocab_size(mut self, vocab: usize) -> Self {
        self.vocab_size = vocab;
        self
    }
}
