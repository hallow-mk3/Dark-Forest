//! Generation utilities: greedy search, top-k, top-p (nucleus sampling), temperature scaling.

use crate::engine::StaticGPT2;
use crate::nn::GPT2;
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use rand::Rng;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    StopToken(usize),
    Eos,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub tokens: Vec<usize>,
    pub prompt_len: usize,
    pub new_tokens: usize,
    pub prefill_time_ms: f64,
    pub decode_time_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_sec: f64,
    pub ms_per_token: f64,
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub stop_tokens: Vec<usize>,
    pub min_tokens: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            stop_tokens: Vec::new(),
            min_tokens: 0,
        }
    }
}

impl SamplerConfig {
    /// Creates a greedy deterministic sampler (temperature = 0.0, top_k = 1, top_p = 1.0).
    pub fn greedy() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_tokens: Vec::new(),
            min_tokens: 0,
        }
    }

    /// Creates a creative high-diversity sampler.
    pub fn creative(temp: f32, top_p: f32) -> Self {
        SamplerConfig {
            temperature: temp,
            top_k: 40,
            top_p,
            repetition_penalty: 1.15,
            stop_tokens: Vec::new(),
            min_tokens: 0,
        }
    }

    /// Creates a temperature-only sampler without top-k or top-p truncation.
    pub fn temperature_only(temp: f32) -> Self {
        SamplerConfig {
            temperature: temp,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_tokens: Vec::new(),
            min_tokens: 0,
        }
    }

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    pub fn with_stop_tokens(mut self, stop_tokens: Vec<usize>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

/// Find index of maximum value in slice (argmax).
#[inline]
pub fn argmax(slice: &[f32]) -> usize {
    if slice.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut best_val = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Sample the next token index directly from a raw logits slice [vocab_size].
pub fn sample_next_token_slice(logits: &[f32], cfg: &SamplerConfig) -> usize {
    sample_next_token_slice_with_context(logits, &[], cfg)
}

/// Sample the next token index applying context-aware repetition penalty, temperature, top-k, and nucleus top-p.
pub fn sample_next_token_slice_with_context(
    logits: &[f32],
    context: &[usize],
    cfg: &SamplerConfig,
) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return 0;
    }

    // Fast-path: greedy deterministic without repetition penalty
    if (cfg.temperature <= 1e-6 || cfg.top_k == 1) && (cfg.repetition_penalty - 1.0).abs() < 1e-5 {
        return argmax(logits);
    }

    // 1. Copy logits and apply repetition penalty to tokens present in recent context
    let mut modified_logits = logits.to_vec();
    if (cfg.repetition_penalty - 1.0).abs() >= 1e-4 && !context.is_empty() {
        let rep_penalty = cfg.repetition_penalty.max(0.1);
        for &tok in context {
            if tok < vocab_size {
                let v = modified_logits[tok];
                modified_logits[tok] = if v > 0.0 {
                    v / rep_penalty
                } else {
                    v * rep_penalty
                };
            }
        }
    }

    // Greedy on penalized logits if temperature is effectively zero
    if cfg.temperature <= 1e-6 || cfg.top_k == 1 {
        return argmax(&modified_logits);
    }

    // 2. Temperature scaling & numerical stability (subtract max)
    let temp = cfg.temperature.max(1e-5);
    let max_val = modified_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<f32> = modified_logits
        .iter()
        .map(|&v| ((v - max_val) / temp).exp())
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        let inv_sum = 1.0 / sum;
        for v in &mut probs {
            *v *= inv_sum;
        }
    } else {
        return argmax(&modified_logits);
    }

    // 3. Top-K filtering
    let mut indexed_probs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if cfg.top_k > 0 && cfg.top_k < vocab_size {
        indexed_probs.truncate(cfg.top_k);
    }

    // 4. Top-P (nucleus) filtering
    if cfg.top_p > 0.0 && cfg.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = indexed_probs.len();
        for (i, (_, p)) in indexed_probs.iter().enumerate() {
            cumsum += p;
            if cumsum > cfg.top_p {
                cutoff = i + 1;
                break;
            }
        }
        indexed_probs.truncate(cutoff);
    }

    // 5. Renormalize and sample
    let filtered_sum: f32 = indexed_probs.iter().map(|(_, p)| p).sum();
    if filtered_sum <= 0.0 || !filtered_sum.is_finite() {
        return indexed_probs.first().map(|(idx, _)| *idx).unwrap_or(0);
    }

    let mut rng = rand::thread_rng();
    let sample: f32 = rng.gen::<f32>() * filtered_sum;

    let mut accum = 0.0f32;
    for (idx, p) in &indexed_probs {
        accum += p;
        if sample <= accum {
            return *idx;
        }
    }

    indexed_probs.last().map(|(idx, _)| *idx).unwrap_or(0)
}

/// Sample the next token index given output logits Tensor [vocab_size].
pub fn sample_next_token(logits: &Tensor, cfg: &SamplerConfig) -> usize {
    let probs = logits.to_vec();
    sample_next_token_slice(&probs, cfg)
}

/// Autoregressive text generation loop for dynamic autograd `GPT2` model.
pub fn generate(
    model: &GPT2,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    sampler_cfg: &SamplerConfig,
) -> Result<Vec<usize>> {
    if prompt_tokens.is_empty() {
        return Err(anyhow!("generate: prompt_tokens cannot be empty"));
    }
    let mut current_tokens = prompt_tokens.to_vec();

    for _ in 0..max_new_tokens {
        let max_ctx = model.cfg.max_seq_len;
        let start = if current_tokens.len() > max_ctx {
            current_tokens.len() - max_ctx
        } else {
            0
        };
        let ctx = &current_tokens[start..];

        let logits_val = model.forward(ctx)?;
        let logits_t = logits_val.tensor();

        // Extract last position logits [vocab_size]
        let vocab_size = model.cfg.vocab_size;
        let seq_len = ctx.len();
        let last_offset = (seq_len - 1) * vocab_size;
        let full_logits = logits_t.to_vec();
        let step_logits = &full_logits[last_offset..last_offset + vocab_size];

        let next_token = sample_next_token_slice_with_context(step_logits, &current_tokens, sampler_cfg);
        current_tokens.push(next_token);

        if current_tokens.len() - prompt_tokens.len() >= sampler_cfg.min_tokens
            && sampler_cfg.stop_tokens.contains(&next_token)
        {
            break;
        }
    }

    Ok(current_tokens)
}

pub fn generate_static(
    model: &mut StaticGPT2,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    sampler_cfg: &SamplerConfig,
) -> Result<Vec<usize>> {
    model.generate(prompt_tokens, max_new_tokens, sampler_cfg)
}

/// Seriously fast autoregressive generation with KV cache and zero-allocation decode loop.
pub fn generate_static_fast(
    model: &mut StaticGPT2,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    sampler_cfg: &SamplerConfig,
) -> Result<Vec<usize>> {
    model.generate_fast(prompt_tokens, max_new_tokens, sampler_cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let logits = vec![0.1, 2.5, -1.0, 4.2, 3.0];
        assert_eq!(argmax(&logits), 3);
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn test_greedy_sampler() {
        let logits = vec![0.1, 2.5, -1.0, 4.2, 3.0];
        let cfg = SamplerConfig::greedy();
        assert_eq!(sample_next_token_slice(&logits, &cfg), 3);
    }

    #[test]
    fn test_top_k_sampler() {
        let logits = vec![0.0, 10.0, 9.0, 1.0];
        let cfg = SamplerConfig {
            temperature: 0.1,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_tokens: vec![],
            min_tokens: 0,
        };
        let token = sample_next_token_slice(&logits, &cfg);
        assert!(token == 1 || token == 2);
    }

    #[test]
    fn test_top_p_sampler() {
        let logits = vec![100.0, -100.0, -100.0];
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repetition_penalty: 1.0,
            stop_tokens: vec![],
            min_tokens: 0,
        };
        assert_eq!(sample_next_token_slice(&logits, &cfg), 0);
    }

    #[test]
    fn test_repetition_penalty() {
        let logits = vec![1.0, 1.0, 1.0];
        let cfg = SamplerConfig::greedy().with_repetition_penalty(2.0);
        // Token 0 in context gets penalized from 1.0 -> 0.5, so token 1 or 2 is preferred
        let token = sample_next_token_slice_with_context(&logits, &[0], &cfg);
        assert!(token == 1 || token == 2, "Expected token != 0, got {}", token);
    }
}
