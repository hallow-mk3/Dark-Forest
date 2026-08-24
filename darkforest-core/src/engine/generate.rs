//! Generation utilities: greedy search, top-k, top-p (nucleus sampling), temperature scaling.

use crate::autograd::Value;
use crate::nn::GPT2;
use crate::tensor::Tensor;
use anyhow::Result;
use rand::Rng;

pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        }
    }
}

/// Sample the next token index given output logits [vocab_size].
pub fn sample_next_token(logits: &Tensor, cfg: &SamplerConfig) -> usize {
    let mut probs = logits.to_vec();
    let vocab_size = probs.len();

    // 1. Temperature scaling
    let temp = cfg.temperature.max(1e-5);
    for v in &mut probs {
        *v /= temp;
    }

    // 2. Numerical stability (subtract max)
    let max_val = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut probs {
        *v = (*v - max_val).exp();
    }
    let sum: f32 = probs.iter().sum();
    for v in &mut probs {
        *v /= sum;
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

    // Renormalize
    let filtered_sum: f32 = indexed_probs.iter().map(|(_, p)| p).sum();
    let mut rng = rand::thread_rng();
    let sample: f32 = rng.gen::<f32>() * filtered_sum;

    let mut accum = 0.0f32;
    for (idx, p) in indexed_probs {
        accum += p;
        if sample <= accum {
            return idx;
        }
    }

    0
}

/// Autoregressive text generation loop
pub fn generate(
    model: &GPT2,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    sampler_cfg: &SamplerConfig,
) -> Result<Vec<usize>> {
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
        let step_tensor = Tensor::from_vec(step_logits.to_vec(), vec![vocab_size])?;

        let next_token = sample_next_token(&step_tensor, sampler_cfg);
        current_tokens.push(next_token);
    }

    Ok(current_tokens)
}
