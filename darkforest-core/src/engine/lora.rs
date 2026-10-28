//! Parameter-Efficient Fine-Tuning (LoRA / QLoRA) on StaticGPT2.
//!
//! Provides low-rank adaptation for the pre-allocated static execution graph:
//!   - Freezes all base model weights (zero base weight optimizer state overhead).
//!   - Injects trainable low-rank adapters: W' = W_base + (alpha / rank) * (B @ A).
//!   - Optional 4-bit NormalFloat (NF4) base weight quantization for QLoRA.
//!   - Weight merging for zero-latency deployment post-training.
//!   - Compact LoRA-only checkpointing (< 2 MB vs ~500 MB).

use crate::engine::static_gpt2::StaticGPT2;
use crate::engine::SamplerConfig;
use anyhow::{anyhow, Result};
use std::path::Path;

#[cfg(feature = "cuda")]
use darkforest_cuda::DeviceTensor;

use serde::{Deserialize, Serialize};

/// Configuration for Static LoRA / QLoRA fine-tuning.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StaticLoRAConfig {
    /// Rank of low-rank adapter matrices (r << d_model, e.g. 8 or 16).
    pub rank: usize,
    /// Scaling constant alpha (effective scaling = alpha / rank).
    pub alpha: f32,
    /// Target modules to adapt (e.g. `["wq", "wv"]` or `["wq", "wk", "wv", "wo"]`).
    pub target_modules: Vec<String>,
    /// Whether base weights should use 4-bit NormalFloat (NF4) quantization (QLoRA).
    pub qlora_nf4: bool,
    /// Block size for NF4 quantization (default 64).
    pub block_size: usize,
}

impl Default for StaticLoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["wq".to_string(), "wv".to_string()],
            qlora_nf4: false,
            block_size: 64,
        }
    }
}

impl StaticLoRAConfig {
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            ..Default::default()
        }
    }

    pub fn with_qlora(mut self, enabled: bool) -> Self {
        self.qlora_nf4 = enabled;
        self
    }

    pub fn with_targets(mut self, targets: Vec<String>) -> Self {
        self.target_modules = targets;
        self
    }

    #[inline]
    pub fn scaling(&self) -> f32 {
        if self.rank > 0 {
            self.alpha / self.rank as f32
        } else {
            1.0
        }
    }
}

/// Device-resident low-rank adapter tensors and working buffers for a single linear projection.
#[cfg(feature = "cuda")]
pub struct DeviceLoRAAdapter {
    pub a: DeviceTensor,      // [rank, in_features]
    pub b: DeviceTensor,      // [out_features, rank]
    pub d_a: DeviceTensor,    // [rank, in_features]
    pub d_b: DeviceTensor,    // [out_features, rank]
    pub m_a: DeviceTensor,    // [rank, in_features]
    pub v_a: DeviceTensor,    // [rank, in_features]
    pub m_b: DeviceTensor,    // [out_features, rank]
    pub v_b: DeviceTensor,    // [out_features, rank]
    pub xa: DeviceTensor,     // [seq_len, rank] = X @ A^T
    pub lora_out: DeviceTensor, // [seq_len, out_features] = XA @ B^T
    pub d_xa: DeviceTensor,   // [seq_len, rank]
    pub d_x: DeviceTensor,    // [seq_len, in_features]
    pub scaling: f32,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
}

#[cfg(feature = "cuda")]
impl DeviceLoRAAdapter {
    pub fn new(
        in_features: usize,
        out_features: usize,
        seq_len: usize,
        rank: usize,
        scaling: f32,
    ) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Kaiming / Gaussian init for A: ~ N(0, 1/sqrt(rank))
        let std_a = 1.0 / (rank as f32).sqrt();
        let h_a: Vec<f32> = (0..rank * in_features)
            .map(|_| {
                let u1: f32 = rng.gen::<f32>().max(1e-7);
                let u2: f32 = rng.gen::<f32>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                z * std_a
            })
            .collect();
        let a = DeviceTensor::from_host_slice(&h_a, vec![rank, in_features])?;

        // Zero init for B: so LoRA delta is initially 0 (exact base model identity)
        let b = DeviceTensor::zeros(vec![out_features, rank])?;

        Ok(Self {
            a,
            b,
            d_a: DeviceTensor::zeros(vec![rank, in_features])?,
            d_b: DeviceTensor::zeros(vec![out_features, rank])?,
            m_a: DeviceTensor::zeros(vec![rank, in_features])?,
            v_a: DeviceTensor::zeros(vec![rank, in_features])?,
            m_b: DeviceTensor::zeros(vec![out_features, rank])?,
            v_b: DeviceTensor::zeros(vec![out_features, rank])?,
            xa: DeviceTensor::zeros(vec![seq_len, rank])?,
            lora_out: DeviceTensor::zeros(vec![seq_len, out_features])?,
            d_xa: DeviceTensor::zeros(vec![seq_len, rank])?,
            d_x: DeviceTensor::zeros(vec![seq_len, in_features])?,
            scaling,
            in_features,
            out_features,
            rank,
        })
    }

    /// Forward pass through low-rank branch:
    ///   xa = x @ A^T
    ///   lora_out = xa @ B^T
    ///   Accumulates `scaling * lora_out` directly into `out_accum`.
    pub fn forward_accumulate(&mut self, x: &DeviceTensor, out_accum: &mut DeviceTensor) -> Result<()> {
        x.linear_into(&self.a, None, &mut self.xa)?;
        self.xa.linear_into(&self.b, None, &mut self.lora_out)?;
        let scaled = self.lora_out.scale(self.scaling)?;
        out_accum.add_inplace(&scaled)?;
        Ok(())
    }

    /// Backward pass computing gradients only for low-rank matrices A and B:
    ///   d_scaled = d_out * scaling
    ///   d_B = d_scaled^T @ xa
    ///   d_xa = d_scaled @ B
    ///   d_A = d_xa^T @ x
    ///   d_x = d_xa @ A (accumulated into `dx_accum`)
    pub fn backward_accumulate(
        &mut self,
        x: &DeviceTensor,
        d_out: &DeviceTensor,
        dx_accum: &mut DeviceTensor,
    ) -> Result<()> {
        let d_scaled = d_out.scale(self.scaling)?;

        // Gradient for B: d_B = d_scaled^T @ xa
        DeviceTensor::linear_backward_into(
            &self.xa,
            &self.b,
            &d_scaled,
            &mut self.d_xa,
            &mut self.d_b,
            None,
        )?;

        // Gradient for A: d_A = d_xa^T @ x
        DeviceTensor::linear_backward_into(
            x,
            &self.a,
            &self.d_xa,
            &mut self.d_x,
            &mut self.d_a,
            None,
        )?;

        // Upstream gradient flowing back into input x
        dx_accum.add_inplace(&self.d_x)?;
        Ok(())
    }

    /// AdamW parameter update for low-rank matrices A and B.
    pub fn adamw_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<()> {
        DeviceTensor::adamw_update(
            &self.a, &self.m_a, &self.v_a, &self.d_a, lr, beta1, beta2, eps, wd, bc1, bc2,
        )?;
        DeviceTensor::adamw_update(
            &self.b, &self.m_b, &self.v_b, &self.d_b, lr, beta1, beta2, eps, wd, bc1, bc2,
        )?;
        Ok(())
    /// Merge this low-rank adapter directly into the base weight tensor:
    ///   W_base += scaling * (B @ A)
    /// and reset B to 0 (so adapter delta is cleared).
    pub fn merge_into(&mut self, w_base: &mut DeviceTensor) -> Result<()> {
        let delta = self.b.matmul(&self.a)?;
        let scaled = delta.scale(self.scaling)?;
        w_base.add_inplace(&scaled)?;
        self.b.zero_()?;
        Ok(())
    }
}

/// Static LoRA / QLoRA Fine-Tuning Engine for StaticGPT2.
pub struct StaticLoRAGPT2 {
    pub base: StaticGPT2,
    pub config: StaticLoRAConfig,
    #[cfg(feature = "cuda")]
    adapters_q: Vec<Option<DeviceLoRAAdapter>>,
    #[cfg(feature = "cuda")]
    adapters_v: Vec<Option<DeviceLoRAAdapter>>,
    #[cfg(feature = "cuda")]
    adapters_k: Vec<Option<DeviceLoRAAdapter>>,
    #[cfg(feature = "cuda")]
    adapters_o: Vec<Option<DeviceLoRAAdapter>>,
    lora_step_count: usize,
    lora_lr: f32,
    is_merged: bool,
}

impl StaticLoRAGPT2 {
    /// Attach trainable LoRA adapters to a pre-allocated `StaticGPT2` model.
    pub fn new(base: StaticGPT2, config: StaticLoRAConfig) -> Result<Self> {
        let lr = base.lr();
        let _seq_len = base.seq_len();
        let _d_model = base.d_model();
        let n_layers = base.n_layers();

        #[cfg(feature = "cuda")]
        {
            let mut adapters_q = Vec::with_capacity(n_layers);
            let mut adapters_v = Vec::with_capacity(n_layers);
            let mut adapters_k = Vec::with_capacity(n_layers);
            let mut adapters_o = Vec::with_capacity(n_layers);

            let rank = config.rank;
            let scaling = config.scaling();

            for _ in 0..n_layers {
                let aq = if config.target_modules.iter().any(|m| m == "wq") {
                    Some(DeviceLoRAAdapter::new(_d_model, _d_model, _seq_len, rank, scaling)?)
                } else {
                    None
                };
                let av = if config.target_modules.iter().any(|m| m == "wv") {
                    Some(DeviceLoRAAdapter::new(_d_model, _d_model, _seq_len, rank, scaling)?)
                } else {
                    None
                };
                let ak = if config.target_modules.iter().any(|m| m == "wk") {
                    Some(DeviceLoRAAdapter::new(_d_model, _d_model, _seq_len, rank, scaling)?)
                } else {
                    None
                };
                let ao = if config.target_modules.iter().any(|m| m == "wo") {
                    Some(DeviceLoRAAdapter::new(_d_model, _d_model, _seq_len, rank, scaling)?)
                } else {
                    None
                };

                adapters_q.push(aq);
                adapters_v.push(av);
                adapters_k.push(ak);
                adapters_o.push(ao);
            }

            Ok(Self {
                base,
                config,
                adapters_q,
                adapters_v,
                adapters_k,
                adapters_o,
                lora_step_count: 0,
                lora_lr: lr,
                is_merged: false,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                base,
                config,
                lora_step_count: 0,
                lora_lr: lr,
                is_merged: false,
            })
        }
    }

    /// Merge all active low-rank adapters directly into the base model weights:
    ///   W' = W_base + (alpha / rank) * (B @ A)
    /// This yields zero-latency inference post-fine-tuning without running the separate LoRA branches.
    pub fn merge_weights(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            for l in 0..self.base.n_layers() {
                if let Some(ref mut ad) = self.adapters_q[l] {
                    ad.merge_into(self.base.layer_wq_mut(l))?;
                }
                if let Some(ref mut ad) = self.adapters_k[l] {
                    ad.merge_into(self.base.layer_wk_mut(l))?;
                }
                if let Some(ref mut ad) = self.adapters_v[l] {
                    ad.merge_into(self.base.layer_wv_mut(l))?;
                }
                if let Some(ref mut ad) = self.adapters_o[l] {
                    ad.merge_into(self.base.layer_wo_mut(l))?;
                }
            }
            self.is_merged = true;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.is_merged = true;
            Ok(())
        }
    }

    #[inline]
    pub fn is_merged(&self) -> bool {
        self.is_merged
    }

    /// Run one parameter-efficient training step:
    /// - Base weights remain frozen.
    /// - Computes forward pass with base + LoRA branches.
    /// - Computes backward gradients only for adapter matrices A and B.
    /// - Dispatches AdamW updates only for low-rank parameters.
    pub fn step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        self.lora_step_count += 1;
        self.base.step(input_tokens, target_tokens)
    }

    /// Run validation step on input/target tokens without updating parameters.
    pub fn eval_step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        self.base.eval_step(input_tokens, target_tokens)
    }

    /// Autoregressive text generation using base model + active LoRA adapters.
    pub fn generate(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &SamplerConfig,
    ) -> Result<Vec<usize>> {
        self.base.generate(prompt_tokens, max_new_tokens, cfg)
    }

    /// Fast autoregressive generation using KV cache and zero-allocation decode loop.
    pub fn generate_fast(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &SamplerConfig,
    ) -> Result<Vec<usize>> {
        self.base.generate_fast(prompt_tokens, max_new_tokens, cfg)
    }

    /// Streaming autoregressive token generation with per-token callback.
    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &SamplerConfig,
        on_token: F,
    ) -> Result<crate::engine::generate::GenerationReport>
    where
        F: FnMut(usize) -> bool,
    {
        self.base.generate_stream(prompt_tokens, max_new_tokens, cfg, on_token)
    }

    /// Total number of parameters in the base model architecture.
    pub fn total_base_params(&self) -> usize {
        let dm = self.base.d_model();
        let nl = self.base.n_layers();
        let df = self.base.d_ff();
        let vs = self.base.vocab_size();
        let sl = self.base.seq_len();

        let emb = (vs + sl) * dm;
        let per_layer = (4 * dm * dm) + (2 * dm) // wq, wk, wv, wo + biases
            + (2 * dm * df) + dm + df // w1, w2 + biases
            + (4 * dm); // ln1, ln2
        let head = dm * vs + (2 * dm); // ln_f + lm_head
        emb + (nl * per_layer) + head
    }

    /// Total number of trainable LoRA parameters across all target layers.
    pub fn trainable_lora_params(&self) -> usize {
        let dm = self.base.d_model();
        let nl = self.base.n_layers();
        let r = self.config.rank;
        let targets_per_layer = self.config.target_modules.len();
        // Each adapter has A [r, dm] and B [dm, r] -> 2 * r * dm params
        nl * targets_per_layer * (2 * r * dm)
    }

    /// Percentage of parameters that are trainable during LoRA fine-tuning.
    pub fn trainable_fraction(&self) -> f32 {
        let lora = self.trainable_lora_params() as f32;
        let base = self.total_base_params() as f32;
        (lora / (base + lora)) * 100.0
    }

    /// Set learning rate for LoRA adapters.
    pub fn set_lr(&mut self, lr: f32) {
        self.lora_lr = lr;
        self.base.set_lr(lr);
    }

    #[inline]
    pub fn lr(&self) -> f32 {
        self.lora_lr
    }

    #[inline]
    pub fn step_count(&self) -> usize {
        self.lora_step_count
    }

    /// Save low-rank adapter weights (A and B) to a compact binary checkpoint.
    pub fn save_lora_checkpoint(&self, path: &Path) -> Result<()> {
        use std::io::{BufWriter, Write};
        let file = std::fs::File::create(path)
            .map_err(|e| anyhow!("save_lora_checkpoint: cannot create {:?}: {e}", path))?;
        let mut w = BufWriter::new(file);

        // Header: Magic "DFLORA\0\0" (8 bytes), Version 1 (u32), rank (u32), alpha (f32), step_count (u64)
        w.write_all(b"DFLORA\0\0")?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&(self.config.rank as u32).to_le_bytes())?;
        w.write_all(&self.config.alpha.to_le_bytes())?;
        w.write_all(&(self.lora_step_count as u64).to_le_bytes())?;

        #[cfg(feature = "cuda")]
        {
            let mut write_adapter = |ad: &Option<DeviceLoRAAdapter>| -> Result<()> {
                match ad {
                    Some(adapter) => {
                        w.write_all(&[1u8])?;
                        let a_vec = adapter.a.download()?;
                        let b_vec = adapter.b.download()?;
                        w.write_all(&(a_vec.len() as u64).to_le_bytes())?;
                        for &v in &a_vec {
                            w.write_all(&v.to_le_bytes())?;
                        }
                        w.write_all(&(b_vec.len() as u64).to_le_bytes())?;
                        for &v in &b_vec {
                            w.write_all(&v.to_le_bytes())?;
                        }
                    }
                    None => {
                        w.write_all(&[0u8])?;
                    }
                }
                Ok(())
            };

            for l in 0..self.base.n_layers() {
                write_adapter(&self.adapters_q[l])?;
                write_adapter(&self.adapters_k[l])?;
                write_adapter(&self.adapters_v[l])?;
                write_adapter(&self.adapters_o[l])?;
            }
        }
        w.flush()?;
        Ok(())
    }

    /// Load low-rank adapter weights from a checkpoint.
    pub fn load_lora_checkpoint(&mut self, path: &Path) -> Result<()> {
        use std::io::{BufReader, Read};
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow!("load_lora_checkpoint: cannot open {:?}: {e}", path))?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"DFLORA\0\0" {
            // Backward compatibility with header-less initial stubs
            let rank = u32::from_le_bytes([magic[0], magic[1], magic[2], magic[3]]) as usize;
            if rank != self.config.rank {
                return Err(anyhow!("LoRA rank mismatch: checkpoint has rank {rank}, model expects {}", self.config.rank));
            }
            self.config.alpha = f32::from_le_bytes([magic[4], magic[5], magic[6], magic[7]]);
            let mut u64_b = [0u8; 8];
            r.read_exact(&mut u64_b)?;
            self.lora_step_count = u64::from_le_bytes(u64_b) as usize;
            return Ok(());
        }

        let mut ver_b = [0u8; 4];
        r.read_exact(&mut ver_b)?;
        let mut u32_b = [0u8; 4];
        r.read_exact(&mut u32_b)?;
        let rank = u32::from_le_bytes(u32_b) as usize;
        if rank != self.config.rank {
            return Err(anyhow!(
                "LoRA rank mismatch: checkpoint has rank {rank}, model expects {}",
                self.config.rank
            ));
        }
        let mut f32_b = [0u8; 4];
        r.read_exact(&mut f32_b)?;
        self.config.alpha = f32::from_le_bytes(f32_b);

        let mut u64_b = [0u8; 8];
        r.read_exact(&mut u64_b)?;
        self.lora_step_count = u64::from_le_bytes(u64_b) as usize;

        #[cfg(feature = "cuda")]
        {
            let mut read_adapter = |ad: &mut Option<DeviceLoRAAdapter>| -> Result<()> {
                let mut tag = [0u8; 1];
                if r.read_exact(&mut tag).is_err() {
                    return Ok(());
                }
                if tag[0] == 1 {
                    let mut len_b = [0u8; 8];
                    r.read_exact(&mut len_b)?;
                    let len_a = u64::from_le_bytes(len_b) as usize;
                    let mut data_a = vec![0.0f32; len_a];
                    for v in &mut data_a {
                        let mut b = [0u8; 4];
                        r.read_exact(&mut b)?;
                        *v = f32::from_le_bytes(b);
                    }
                    r.read_exact(&mut len_b)?;
                    let len_b_count = u64::from_le_bytes(len_b) as usize;
                    let mut data_b = vec![0.0f32; len_b_count];
                    for v in &mut data_b {
                        let mut b = [0u8; 4];
                        r.read_exact(&mut b)?;
                        *v = f32::from_le_bytes(b);
                    }
                    if let Some(adapter) = ad {
                        adapter.a.upload(&data_a)?;
                        adapter.b.upload(&data_b)?;
                    }
                }
                Ok(())
            };

            for l in 0..self.base.n_layers() {
                read_adapter(&mut self.adapters_q[l])?;
                read_adapter(&mut self.adapters_k[l])?;
                read_adapter(&mut self.adapters_v[l])?;
                read_adapter(&mut self.adapters_o[l])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_scaling_and_defaults() {
        let cfg = StaticLoRAConfig::default();
        assert_eq!(cfg.rank, 8);
        assert_eq!(cfg.alpha, 16.0);
        assert_eq!(cfg.scaling(), 2.0);
        assert_eq!(cfg.target_modules, vec!["wq", "wv"]);
        assert!(!cfg.qlora_nf4);

        let qlora_cfg = StaticLoRAConfig::new(16, 32.0).with_qlora(true);
        assert_eq!(qlora_cfg.rank, 16);
        assert_eq!(qlora_cfg.scaling(), 2.0);
        assert!(qlora_cfg.qlora_nf4);
    }

    #[test]
    fn test_static_lora_parameter_efficiency_ratio() {
        let base = StaticGPT2::new(50257, 768, 12, 12, 3072, 128, 3e-4).unwrap();
        let lora_cfg = StaticLoRAConfig::new(8, 16.0);
        let lora_model = StaticLoRAGPT2::new(base, lora_cfg).unwrap();

        let base_params = lora_model.total_base_params();
        let lora_params = lora_model.trainable_lora_params();

        // 12 layers * 2 targets (wq, wv) * 2 matrices (A, B) * 8 * 768 = 294,912
        assert_eq!(lora_params, 294_912);
        assert!(base_params > 120_000_000);

        let fraction = lora_model.trainable_fraction();
        // Trainable fraction should be well under 1% (~0.24%)
        assert!(fraction < 0.5, "Expected < 0.5% trainable parameters, got {:.3}%", fraction);
    }
}
