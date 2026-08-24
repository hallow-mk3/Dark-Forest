//! TransformerBlock and GPT-2-small model definition.
//!
//! GPT-2 small config:
//!   vocab_size = 50257
//!   d_model    = 768
//!   n_heads    = 12
//!   n_layers   = 12
//!   d_ff       = 3072  (4 * d_model)
//!   max_seq_len = 1024

use crate::autograd::Value;
use crate::nn::{Embedding, Linear, MultiHeadAttention, PosEmbedding};
use crate::tensor::Tensor;
use anyhow::Result;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
}

impl GPT2Config {
    pub fn small() -> Self {
        GPT2Config {
            vocab_size: 50257,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            d_ff: 3072,
            max_seq_len: 1024,
        }
    }

    /// Tiny config for gradient checking and smoke tests.
    pub fn tiny() -> Self {
        GPT2Config {
            vocab_size: 256,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            max_seq_len: 64,
        }
    }

    pub fn param_count(&self) -> usize {
        // Token embedding: vocab_size * d_model
        let emb = self.vocab_size * self.d_model;
        // Pos embedding: max_seq_len * d_model
        let pos = self.max_seq_len * self.d_model;
        // Per layer: 4 attention projections (each d_model * d_model) + 2 LN + 2 MLP linears
        let per_layer = 4 * self.d_model * self.d_model   // QKV + out
            + 2 * self.d_model                              // 2 LN gammas
            + 2 * self.d_model                              // 2 LN betas
            + self.d_model * self.d_ff                     // MLP fc1
            + self.d_ff * self.d_model; // MLP fc2
                                        // Final LN + head
        let head = self.d_model * 2 + self.vocab_size * self.d_model;
        emb + pos + self.n_layers * per_layer + head
    }
}

// ---------------------------------------------------------------------------
// LayerNorm module
// ---------------------------------------------------------------------------
pub struct LayerNorm {
    pub gamma: Value,
    pub beta: Value,
    pub features: usize,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        LayerNorm {
            gamma: Value::leaf(Tensor::ones(vec![features])),
            beta: Value::leaf(Tensor::zeros(vec![features])),
            features,
        }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        x.layernorm(&self.gamma, &self.beta)
    }

    pub fn to_device(&mut self, device: crate::tensor::Device) -> Result<()> {
        self.gamma = self.gamma.to_device(device.clone())?;
        self.beta = self.beta.to_device(device)?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Value> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl TransformerBlock {
    pub fn new(cfg: &GPT2Config) -> Self {
        TransformerBlock {
            ln1: LayerNorm::new(cfg.d_model),
            attn: MultiHeadAttention::new(cfg.d_model, cfg.n_heads),
            ln2: LayerNorm::new(cfg.d_model),
            fc1: Linear::new(cfg.d_model, cfg.d_ff, true),
            fc2: Linear::new(cfg.d_ff, cfg.d_model, true),
        }
    }

    /// Forward: x shape [seq_len, d_model]
    pub fn forward(&self, x: &Value, seq_len: usize) -> Result<Value> {
        // Attention sub-layer with residual
        let ln1_out = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&ln1_out, seq_len)?;
        let res1 = x.add(&attn_out)?;

        // MLP sub-layer with residual
        let ln2_out = self.ln2.forward(&res1)?;
        let mlp_h = self.fc1.forward(&ln2_out)?;
        let mlp_a = mlp_h.gelu()?;
        let mlp_out = self.fc2.forward(&mlp_a)?;
        let res2 = res1.add(&mlp_out)?;

        Ok(res2)
    }

    pub fn to_device(&mut self, device: crate::tensor::Device) -> Result<()> {
        self.ln1.to_device(device.clone())?;
        self.attn.to_device(device.clone())?;
        self.ln2.to_device(device.clone())?;
        self.fc1.to_device(device.clone())?;
        self.fc2.to_device(device)?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.ln1.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.fc1.parameters());
        p.extend(self.fc2.parameters());
        p
    }
}

// ---------------------------------------------------------------------------
// GPT-2 model
// ---------------------------------------------------------------------------
pub struct GPT2 {
    pub cfg: GPT2Config,
    pub tok_emb: Embedding,
    pub pos_emb: PosEmbedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_final: LayerNorm,
    pub lm_head: Linear, // [d_model → vocab_size]
}

impl GPT2 {
    pub fn new(cfg: GPT2Config) -> Self {
        let tok_emb = Embedding::new(cfg.vocab_size, cfg.d_model);
        let pos_emb = PosEmbedding::new(cfg.max_seq_len, cfg.d_model);
        let blocks = (0..cfg.n_layers)
            .map(|_| TransformerBlock::new(&cfg))
            .collect();
        let ln_final = LayerNorm::new(cfg.d_model);
        let lm_head = Linear::new(cfg.d_model, cfg.vocab_size, false);
        GPT2 {
            cfg,
            tok_emb,
            pos_emb,
            blocks,
            ln_final,
            lm_head,
        }
    }

    pub fn to_device(&mut self, device: crate::tensor::Device) -> Result<()> {
        self.tok_emb.to_device(device.clone())?;
        self.pos_emb.to_device(device.clone())?;
        for block in &mut self.blocks {
            block.to_device(device.clone())?;
        }
        self.ln_final.to_device(device.clone())?;
        self.lm_head.to_device(device)?;
        Ok(())
    }

    /// Forward pass.
    ///
    /// `tokens`: input token indices, shape [seq_len]
    /// Returns logits, shape [seq_len, vocab_size]
    pub fn forward(&self, tokens: &[usize]) -> Result<Value> {
        let seq_len = tokens.len();

        // Token + positional embeddings
        let tok = self.tok_emb.forward(tokens)?;
        let pos = self.pos_emb.forward(seq_len)?;
        let mut x = tok.add(&pos)?;

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, seq_len)?;
        }

        // Final layer norm + lm head
        let x_norm = self.ln_final.forward(&x)?;
        let logits = self.lm_head.forward(&x_norm)?;

        Ok(logits)
    }

    /// Compute cross-entropy loss on a batch of tokens.
    ///
    /// `tokens`: shape [seq_len+1], predicts tokens[1..] from tokens[..seq_len]
    pub fn loss(&self, tokens: &[usize]) -> Result<Value> {
        let inputs = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        let logits = self.forward(inputs)?;
        logits.cross_entropy_loss(targets)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.tok_emb.parameters();
        p.extend(self.pos_emb.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.ln_final.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}
