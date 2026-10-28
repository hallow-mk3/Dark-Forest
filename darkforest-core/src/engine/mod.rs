//! Complete Static Execution Engine for Dark Forest Runtime.
//!
//! `Engine` provides a unified, production-grade ML runtime interface
//! supporting deterministic training, fast KV-cache decoding, streaming generation,
//! and parameter-efficient fine-tuning (LoRA / QLoRA) with zero-latency weight merging.

pub mod config;
pub mod generate;
pub mod kv_cache;
pub mod lora;
pub mod safetensors;
pub mod static_gpt2;
pub mod tokenizer;

pub use config::EngineConfig;
pub use generate::{
    argmax, generate, generate_static, generate_static_fast, sample_next_token, sample_next_token_slice,
    sample_next_token_slice_with_context, GenerationReport, SamplerConfig, StopReason,
};
pub use kv_cache::KVCache;
pub use lora::{StaticLoRAConfig, StaticLoRAGPT2};
pub use safetensors::{load_gpt2_safetensors, save_gpt2_safetensors, SafeTensorsFile, TensorMeta};
pub use static_gpt2::StaticGPT2;
pub use tokenizer::{BPETokenizer, ByteTokenizer, CharTokenizer, Tokenizer};

use anyhow::Result;
use std::path::Path;

/// Unified, production-grade engine facade supporting both full parameter training
/// and LoRA/QLoRA parameter-efficient adaptation, fast KV-cache decoding, streaming,
/// and zero-latency weight merging.
pub enum Engine {
    Base(StaticGPT2),
    LoRA(StaticLoRAGPT2),
}

impl Engine {
    /// Initialize an engine from an `EngineConfig`.
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        let mut base = StaticGPT2::new(
            cfg.vocab_size,
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.d_ff,
            cfg.seq_len,
            cfg.lr,
        )?;
        if let Some(norm) = cfg.max_grad_norm {
            base.set_max_grad_norm(Some(norm));
        }
        if let Some(lora_cfg) = cfg.lora {
            let lora_model = StaticLoRAGPT2::new(base, lora_cfg)?;
            Ok(Engine::LoRA(lora_model))
        } else {
            Ok(Engine::Base(base))
        }
    }

    /// Load a complete engine from a configuration and checkpoint path.
    pub fn from_checkpoint(cfg: EngineConfig, path: &Path) -> Result<Self> {
        let mut engine = Self::new(cfg)?;
        engine.load_checkpoint(path)?;
        Ok(engine)
    }

    /// Run one training step on input and target tokens. Returns cross-entropy loss.
    pub fn step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        match self {
            Engine::Base(m) => m.step(input_tokens, target_tokens),
            Engine::LoRA(m) => m.step(input_tokens, target_tokens),
        }
    }

    /// Run one training step replayed via CUDA Graphs (or CPU equivalent).
    pub fn step_graph(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        match self {
            Engine::Base(m) => m.step_graph(input_tokens, target_tokens),
            Engine::LoRA(m) => m.step(input_tokens, target_tokens),
        }
    }

    /// Run one evaluation / validation step without parameter updates.
    pub fn eval_step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        match self {
            Engine::Base(m) => m.eval_step(input_tokens, target_tokens),
            Engine::LoRA(m) => m.eval_step(input_tokens, target_tokens),
        }
    }

    /// Run forward inference pass returning unnormalized logits matrix `[seq_len, vocab_size]`.
    pub fn forward(&mut self, input_tokens: &[usize]) -> Result<Vec<f32>> {
        match self {
            Engine::Base(m) => m.forward(input_tokens),
            Engine::LoRA(m) => m.base.forward(input_tokens),
        }
    }

    /// Run forward inference pass returning only the last token position logits `[vocab_size]`.
    pub fn forward_last_logits(&mut self, input_tokens: &[usize]) -> Result<Vec<f32>> {
        match self {
            Engine::Base(m) => m.forward_last_logits(input_tokens),
            Engine::LoRA(m) => m.base.forward_last_logits(input_tokens),
        }
    }

    /// Predict next greedy token index (argmax of last token logits).
    pub fn predict(&mut self, input_tokens: &[usize]) -> Result<usize> {
        match self {
            Engine::Base(m) => m.predict(input_tokens),
            Engine::LoRA(m) => m.base.predict(input_tokens),
        }
    }

    /// Autoregressive text generation.
    pub fn generate(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &SamplerConfig,
    ) -> Result<Vec<usize>> {
        match self {
            Engine::Base(m) => m.generate(prompt_tokens, max_new_tokens, cfg),
            Engine::LoRA(m) => m.generate(prompt_tokens, max_new_tokens, cfg),
        }
    }

    /// Fast autoregressive generation using KV cache and zero-allocation decode loop.
    pub fn generate_fast(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &SamplerConfig,
    ) -> Result<Vec<usize>> {
        match self {
            Engine::Base(m) => m.generate_fast(prompt_tokens, max_new_tokens, cfg),
            Engine::LoRA(m) => m.generate_fast(prompt_tokens, max_new_tokens, cfg),
        }
    }

    /// Streaming autoregressive token generation with per-token callback.
    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &SamplerConfig,
        on_token: F,
    ) -> Result<GenerationReport>
    where
        F: FnMut(usize) -> bool,
    {
        match self {
            Engine::Base(m) => m.generate_stream(prompt_tokens, max_new_tokens, cfg, on_token),
            Engine::LoRA(m) => m.generate_stream(prompt_tokens, max_new_tokens, cfg, on_token),
        }
    }

    /// Save model checkpoint to binary file.
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        match self {
            Engine::Base(m) => m.save_checkpoint(path),
            Engine::LoRA(m) => m.base.save_checkpoint(path),
        }
    }

    /// Load model checkpoint from binary file.
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        match self {
            Engine::Base(m) => m.load_checkpoint(path),
            Engine::LoRA(m) => m.base.load_checkpoint(path),
        }
    }

    /// Load HuggingFace GPT-2 weights from `.safetensors`.
    pub fn load_safetensors(&mut self, path: &Path) -> Result<()> {
        match self {
            Engine::Base(m) => safetensors::load_gpt2_safetensors(m, path),
            Engine::LoRA(m) => safetensors::load_gpt2_safetensors(&mut m.base, path),
        }
    }

    /// Export model weights to a standard `.safetensors` file.
    pub fn save_safetensors(&self, path: &Path) -> Result<()> {
        match self {
            Engine::Base(m) => safetensors::save_gpt2_safetensors(m, path),
            Engine::LoRA(m) => safetensors::save_gpt2_safetensors(&m.base, path),
        }
    }

    /// Save LoRA adapter checkpoint (if running in LoRA mode).
    pub fn save_lora_checkpoint(&self, path: &Path) -> Result<()> {
        match self {
            Engine::Base(_) => Err(anyhow::anyhow!("Cannot save LoRA checkpoint from a base engine")),
            Engine::LoRA(m) => m.save_lora_checkpoint(path),
        }
    }

    /// Load LoRA adapter checkpoint (if running in LoRA mode).
    pub fn load_lora_checkpoint(&mut self, path: &Path) -> Result<()> {
        match self {
            Engine::Base(_) => Err(anyhow::anyhow!("Cannot load LoRA checkpoint into a base engine")),
            Engine::LoRA(m) => m.load_lora_checkpoint(path),
        }
    }

    /// Merge LoRA adapter weights directly into base weights for zero-latency inference.
    pub fn merge_lora(&mut self) -> Result<()> {
        match self {
            Engine::Base(_) => Ok(()),
            Engine::LoRA(m) => m.merge_weights(),
        }
    }

    pub fn is_lora(&self) -> bool {
        matches!(self, Engine::LoRA(_))
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Engine::Base(m) => m.vocab_size(),
            Engine::LoRA(m) => m.base.vocab_size(),
        }
    }

    pub fn d_model(&self) -> usize {
        match self {
            Engine::Base(m) => m.d_model(),
            Engine::LoRA(m) => m.base.d_model(),
        }
    }

    pub fn n_heads(&self) -> usize {
        match self {
            Engine::Base(m) => m.n_heads(),
            Engine::LoRA(m) => m.base.n_heads(),
        }
    }

    pub fn n_layers(&self) -> usize {
        match self {
            Engine::Base(m) => m.n_layers(),
            Engine::LoRA(m) => m.base.n_layers(),
        }
    }

    pub fn seq_len(&self) -> usize {
        match self {
            Engine::Base(m) => m.seq_len(),
            Engine::LoRA(m) => m.base.seq_len(),
        }
    }

    pub fn lr(&self) -> f32 {
        match self {
            Engine::Base(m) => m.lr(),
            Engine::LoRA(m) => m.lr(),
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        match self {
            Engine::Base(m) => m.set_lr(lr),
            Engine::LoRA(m) => m.set_lr(lr),
        }
    }

    pub fn step_count(&self) -> usize {
        match self {
            Engine::Base(m) => m.step_count(),
            Engine::LoRA(m) => m.step_count(),
        }
    }

    /// Print comprehensive engine telemetry and specification.
    pub fn summary(&self) -> String {
        let mode = if self.is_lora() { "LoRA / QLoRA PEFT" } else { "Full Static Dense" };
        format!(
            "Dark Forest Engine [{mode}] | Vocab: {} | d_model: {} | Layers: {} | Heads: {} | Seq: {} | Steps: {}",
            self.vocab_size(),
            self.d_model(),
            self.n_layers(),
            self.n_heads(),
            self.seq_len(),
            self.step_count(),
        )
    }
}
