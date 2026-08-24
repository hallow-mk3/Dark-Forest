//! StaticGPT2 — a fully static, pre-allocated GPT-2 training engine.
//!
//! Eliminates all per-step overhead from the dynamic autograd system:
//! - Zero `Arc<Mutex>` allocations
//! - Zero computation graph construction
//! - Zero intermediate tensor heap allocations (all buffers pre-allocated)
//! - Direct CUDA kernel dispatch every step

use anyhow::{anyhow, Result};

#[cfg(feature = "cuda")]
use darkforest_cuda::{AttentionContext, DeviceTensor};

/// Per-layer weight tensors (all device-resident).
#[cfg(feature = "cuda")]
struct LayerWeights {
    // LayerNorm 1
    ln1_gamma: DeviceTensor, // [d_model]
    ln1_beta: DeviceTensor,  // [d_model]
    // Attention projections: weight [d_model, d_model], bias [d_model]
    wq: DeviceTensor, bq: DeviceTensor,
    wk: DeviceTensor, bk: DeviceTensor,
    wv: DeviceTensor, bv: DeviceTensor,
    wo: DeviceTensor, bo: DeviceTensor,
    // LayerNorm 2
    ln2_gamma: DeviceTensor,
    ln2_beta: DeviceTensor,
    // MLP
    w1: DeviceTensor, b1: DeviceTensor, // [d_ff, d_model] / [d_ff]
    w2: DeviceTensor, b2: DeviceTensor, // [d_model, d_ff] / [d_model]
}

/// Per-layer forward activation buffers (pre-allocated, reused every step).
#[cfg(feature = "cuda")]
struct LayerActivations {
    ln1_out: DeviceTensor,   // [seq, d_model]
    ln1_mean: DeviceTensor,  // [seq]
    ln1_rstd: DeviceTensor,  // [seq]
    q: DeviceTensor,         // [seq, d_model]
    k: DeviceTensor,         // [seq, d_model]
    v: DeviceTensor,         // [seq, d_model]
    attn_out: DeviceTensor,  // [seq, d_model]
    proj_out: DeviceTensor,  // [seq, d_model]
    x_mid: DeviceTensor,     // [seq, d_model]  (x after attn residual)
    ln2_out: DeviceTensor,   // [seq, d_model]
    ln2_mean: DeviceTensor,  // [seq]
    ln2_rstd: DeviceTensor,  // [seq]
    mlp_h: DeviceTensor,     // [seq, d_ff]
    mlp_a: DeviceTensor,     // [seq, d_ff]  (after gelu)
    mlp_out: DeviceTensor,   // [seq, d_model]
}

/// Per-layer gradient buffers (pre-allocated, reused every step).
#[cfg(feature = "cuda")]
struct LayerGrads {
    // upstream gradient flowing into this layer from above
    dx: DeviceTensor,           // [seq, d_model]
    // MLP backward
    d_mlp_out: DeviceTensor,    // [seq, d_model]
    d_mlp_a: DeviceTensor,      // [seq, d_ff]
    d_mlp_h: DeviceTensor,      // [seq, d_ff]
    d_w2: DeviceTensor,         // [d_model, d_ff]
    d_b2: DeviceTensor,         // [d_model]
    d_w1: DeviceTensor,         // [d_ff, d_model]
    d_b1: DeviceTensor,         // [d_ff]
    d_ln2_out: DeviceTensor,    // [seq, d_model]
    d_ln2_gamma: DeviceTensor,  // [d_model]
    d_ln2_beta: DeviceTensor,   // [d_model]
    // Attention backward
    d_proj_out: DeviceTensor,   // [seq, d_model]
    d_attn_out: DeviceTensor,   // [seq, d_model]
    d_q: DeviceTensor,          // [seq, d_model]
    d_k: DeviceTensor,          // [seq, d_model]
    d_v: DeviceTensor,          // [seq, d_model]
    d_wo: DeviceTensor,         // [d_model, d_model]
    d_bo: DeviceTensor,         // [d_model]
    d_wq: DeviceTensor,         // [d_model, d_model]
    d_bq: DeviceTensor,         // [d_model]
    d_wk: DeviceTensor,         // [d_model, d_model]
    d_bk: DeviceTensor,         // [d_model]
    d_wv: DeviceTensor,         // [d_model, d_model]
    d_bv: DeviceTensor,         // [d_model]
    d_ln1_out: DeviceTensor,    // [seq, d_model]
    d_ln1_gamma: DeviceTensor,  // [d_model]
    d_ln1_beta: DeviceTensor,   // [d_model]
}

/// AdamW moment buffers for one parameter tensor.
#[cfg(feature = "cuda")]
struct Moments {
    m: DeviceTensor, // first moment
    v: DeviceTensor, // second moment
}

/// A fully static, pre-allocated GPT-2 transformer training engine.
#[cfg(feature = "cuda")]
/// A fully static, pre-allocated GPT-2 transformer training engine.
#[cfg(feature = "cuda")]
pub struct StaticGPT2 {
    // Config
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub seq_len: usize,

    // Persistent device index buffers (zero malloc per step)
    d_input_indices: darkforest_cuda::memory::CudaBuffer,
    d_pos_indices: darkforest_cuda::memory::CudaBuffer,
    d_target_indices: darkforest_cuda::memory::CudaBuffer,

    // Pre-allocated host u32 staging buffers (avoid Vec alloc per step)
    h_input_u32: Vec<u32>,
    h_target_u32: Vec<u32>,

    // Global weights
    tok_emb: DeviceTensor,      // [vocab_size, d_model]
    pos_emb: DeviceTensor,      // [seq_len, d_model]
    ln_f_gamma: DeviceTensor,   // [d_model]
    ln_f_beta: DeviceTensor,    // [d_model]
    lm_head: DeviceTensor,      // [vocab_size, d_model]

    // Per-layer weights
    layers: Vec<LayerWeights>,

    // Forward activation buffers
    tok_out: DeviceTensor,      // [seq, d_model]
    pos_out: DeviceTensor,      // [seq, d_model]
    x_in: Vec<DeviceTensor>,    // [n_layers+1][seq, d_model] — x entering each layer + final
    layer_acts: Vec<LayerActivations>,
    ln_f_out: DeviceTensor,     // [seq, d_model]
    ln_f_mean: DeviceTensor,    // [seq]
    ln_f_rstd: DeviceTensor,    // [seq]
    logits: DeviceTensor,       // [seq, vocab_size]
    loss_tensor: DeviceTensor,  // [1]
    probs: DeviceTensor,        // [seq, vocab_size]
    grad_one: DeviceTensor,     // [1]

    // Pinned host buffer for async loss D2H (no CPU-GPU sync inside step)
    pinned_loss: darkforest_cuda::memory::PinnedBuffer,
    // Gradient buffers
    d_logits: DeviceTensor,     // [seq, vocab_size]
    d_lm_head: DeviceTensor,    // [vocab_size, d_model]
    d_ln_f_out: DeviceTensor,   // [seq, d_model]
    d_ln_f_gamma: DeviceTensor, // [d_model]
    d_ln_f_beta: DeviceTensor,  // [d_model]
    layer_grads: Vec<LayerGrads>,
    d_tok_emb: DeviceTensor,    // [vocab_size, d_model]
    d_pos_emb: DeviceTensor,    // [seq_len, d_model]

    // Reusable dx residual accumulator
    dx_comb: DeviceTensor,      // [seq, d_model]

    // AdamW state
    step_count: usize,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    // moment buffers — one Moments per weight tensor
    m_tok: Moments, m_pos: Moments,
    m_ln_f_gamma: Moments, m_ln_f_beta: Moments,
    m_lm_head: Moments,
    layer_moments: Vec<LayerMoments>,

    // Attention contexts (preallocated flash-attention scratch)
    attn_contexts: Vec<std::sync::Arc<AttentionContext>>,

    // CUDA Graph executable for the step
    graph_exec: Option<darkforest_cuda::CudaGraphExec>,
}

/// AdamW moment buffers for all weights in one transformer layer.
#[cfg(feature = "cuda")]
struct LayerMoments {
    ln1_gamma: Moments, ln1_beta: Moments,
    wq: Moments, bq: Moments,
    wk: Moments, bk: Moments,
    wv: Moments, bv: Moments,
    wo: Moments, bo: Moments,
    ln2_gamma: Moments, ln2_beta: Moments,
    w1: Moments, b1: Moments,
    w2: Moments, b2: Moments,
}

#[cfg(feature = "cuda")]
impl Moments {
    fn for_tensor(t: &DeviceTensor) -> Result<Self> {
        Ok(Moments {
            m: DeviceTensor::zeros(t.shape.clone())?,
            v: DeviceTensor::zeros(t.shape.clone())?,
        })
    }
}

#[cfg(feature = "cuda")]
fn randn_device(shape: Vec<usize>, std: f32) -> Result<DeviceTensor> {
    use rand::distributions::Distribution;
    let n = shape.iter().product::<usize>();
    let normal = rand_distr::Normal::new(0.0f32, std).unwrap();
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    DeviceTensor::from_host(&data, shape)
}

#[cfg(feature = "cuda")]
impl StaticGPT2 {
    /// Build the model with random initial weights and pre-allocate all buffers.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        seq_len: usize,
        lr: f32,
    ) -> Result<Self> {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_head = d_model / n_heads;
        if d_head > 128 {
            return Err(anyhow!(
                "StaticGPT2 head dimension must be <= 128, got {d_head}"
            ));
        }
        let std_emb = (d_model as f32).powf(-0.5);
        let std_proj = (2.0f32 / d_model as f32).sqrt();
        let std_ff = (2.0f32 / d_model as f32).sqrt();

        let tok_emb = randn_device(vec![vocab_size, d_model], std_emb)?;
        let pos_emb = randn_device(vec![seq_len, d_model], std_emb)?;
        let ln_f_gamma = DeviceTensor::ones(vec![d_model])?;
        let ln_f_beta = DeviceTensor::zeros(vec![d_model])?;
        let lm_head = randn_device(vec![vocab_size, d_model], std_proj)?;

        let mut layers = Vec::with_capacity(n_layers);
        let mut layer_acts = Vec::with_capacity(n_layers);
        let mut layer_grads = Vec::with_capacity(n_layers);
        let mut layer_moments = Vec::with_capacity(n_layers);
        let mut attn_contexts = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let wq = randn_device(vec![d_model, d_model], std_proj)?;
            let bq = DeviceTensor::zeros(vec![d_model])?;
            let wk = randn_device(vec![d_model, d_model], std_proj)?;
            let bk = DeviceTensor::zeros(vec![d_model])?;
            let wv = randn_device(vec![d_model, d_model], std_proj)?;
            let bv = DeviceTensor::zeros(vec![d_model])?;
            let wo = randn_device(vec![d_model, d_model], std_proj)?;
            let bo = DeviceTensor::zeros(vec![d_model])?;
            let w1 = randn_device(vec![d_ff, d_model], std_ff)?;
            let b1 = DeviceTensor::zeros(vec![d_ff])?;
            let w2 = randn_device(vec![d_model, d_ff], std_ff)?;
            let b2 = DeviceTensor::zeros(vec![d_model])?;
            let ln1_gamma = DeviceTensor::ones(vec![d_model])?;
            let ln1_beta  = DeviceTensor::zeros(vec![d_model])?;
            let ln2_gamma = DeviceTensor::ones(vec![d_model])?;
            let ln2_beta  = DeviceTensor::zeros(vec![d_model])?;

            layer_moments.push(LayerMoments {
                ln1_gamma: Moments::for_tensor(&ln1_gamma)?,
                ln1_beta:  Moments::for_tensor(&ln1_beta)?,
                wq: Moments::for_tensor(&wq)?,  bq: Moments::for_tensor(&bq)?,
                wk: Moments::for_tensor(&wk)?,  bk: Moments::for_tensor(&bk)?,
                wv: Moments::for_tensor(&wv)?,  bv: Moments::for_tensor(&bv)?,
                wo: Moments::for_tensor(&wo)?,  bo: Moments::for_tensor(&bo)?,
                ln2_gamma: Moments::for_tensor(&ln2_gamma)?,
                ln2_beta:  Moments::for_tensor(&ln2_beta)?,
                w1: Moments::for_tensor(&w1)?,  b1: Moments::for_tensor(&b1)?,
                w2: Moments::for_tensor(&w2)?,  b2: Moments::for_tensor(&b2)?,
            });

            layers.push(LayerWeights {
                ln1_gamma, ln1_beta, wq, bq, wk, bk, wv, bv, wo, bo,
                ln2_gamma, ln2_beta, w1, b1, w2, b2,
            });

            // Pre-allocate activation buffers for this layer
            layer_acts.push(LayerActivations {
                ln1_out:  DeviceTensor::zeros(vec![seq_len, d_model])?,
                ln1_mean: DeviceTensor::zeros(vec![seq_len])?,
                ln1_rstd: DeviceTensor::zeros(vec![seq_len])?,
                q: DeviceTensor::zeros(vec![seq_len, d_model])?,
                k: DeviceTensor::zeros(vec![seq_len, d_model])?,
                v: DeviceTensor::zeros(vec![seq_len, d_model])?,
                attn_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                proj_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                x_mid:    DeviceTensor::zeros(vec![seq_len, d_model])?,
                ln2_out:  DeviceTensor::zeros(vec![seq_len, d_model])?,
                ln2_mean: DeviceTensor::zeros(vec![seq_len])?,
                ln2_rstd: DeviceTensor::zeros(vec![seq_len])?,
                mlp_h:    DeviceTensor::zeros(vec![seq_len, d_ff])?,
                mlp_a:    DeviceTensor::zeros(vec![seq_len, d_ff])?,
                mlp_out:  DeviceTensor::zeros(vec![seq_len, d_model])?,
            });

            // Pre-allocate gradient buffers
            layer_grads.push(LayerGrads {
                dx:          DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_mlp_out:   DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_mlp_a:     DeviceTensor::zeros(vec![seq_len, d_ff])?,
                d_mlp_h:     DeviceTensor::zeros(vec![seq_len, d_ff])?,
                d_w2:        DeviceTensor::zeros(vec![d_model, d_ff])?,
                d_b2:        DeviceTensor::zeros(vec![d_model])?,
                d_w1:        DeviceTensor::zeros(vec![d_ff, d_model])?,
                d_b1:        DeviceTensor::zeros(vec![d_ff])?,
                d_ln2_out:   DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_ln2_gamma: DeviceTensor::zeros(vec![d_model])?,
                d_ln2_beta:  DeviceTensor::zeros(vec![d_model])?,
                d_proj_out:  DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_attn_out:  DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_q: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_k: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_v: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_wo: DeviceTensor::zeros(vec![d_model, d_model])?,
                d_bo: DeviceTensor::zeros(vec![d_model])?,
                d_wq: DeviceTensor::zeros(vec![d_model, d_model])?,
                d_bq: DeviceTensor::zeros(vec![d_model])?,
                d_wk: DeviceTensor::zeros(vec![d_model, d_model])?,
                d_bk: DeviceTensor::zeros(vec![d_model])?,
                d_wv: DeviceTensor::zeros(vec![d_model, d_model])?,
                d_bv: DeviceTensor::zeros(vec![d_model])?,
                d_ln1_out:   DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_ln1_gamma: DeviceTensor::zeros(vec![d_model])?,
                d_ln1_beta:  DeviceTensor::zeros(vec![d_model])?,
            });

            attn_contexts.push(std::sync::Arc::new(
                AttentionContext::new(n_heads, seq_len, d_head)?
            ));
        }

        // x buffers entering each layer + after final ln
        let mut x_in = Vec::with_capacity(n_layers + 1);
        for _ in 0..=n_layers {
            x_in.push(DeviceTensor::zeros(vec![seq_len, d_model])?);
        }

        let allocator = std::sync::Arc::new(darkforest_cuda::memory::GpuAllocator);
        let d_input_indices = darkforest_cuda::memory::CudaBuffer::new(seq_len * std::mem::size_of::<u32>(), allocator.clone())?;
        let d_pos_indices = darkforest_cuda::memory::CudaBuffer::new(seq_len * std::mem::size_of::<u32>(), allocator.clone())?;
        let d_target_indices = darkforest_cuda::memory::CudaBuffer::new(seq_len * std::mem::size_of::<u32>(), allocator)?;

        // Preload pos indices [0, 1, ..., seq_len-1] once
        let pos_data: Vec<u32> = (0..seq_len as u32).collect();
        d_pos_indices.upload_bytes(pos_data.as_ptr() as *const u8, seq_len * std::mem::size_of::<u32>())?;

        let m_tok      = Moments::for_tensor(&tok_emb)?;
        let m_pos      = Moments::for_tensor(&pos_emb)?;
        let m_ln_f_gamma = Moments::for_tensor(&ln_f_gamma)?;
        let m_ln_f_beta  = Moments::for_tensor(&ln_f_beta)?;
        let m_lm_head    = Moments::for_tensor(&lm_head)?;

        Ok(Self {
            vocab_size, d_model, n_heads, n_layers, d_ff, seq_len,
            d_input_indices, d_pos_indices, d_target_indices,
            tok_emb, pos_emb, ln_f_gamma, ln_f_beta, lm_head,
            layers, layer_acts,
            tok_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
            pos_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
            x_in,
            ln_f_out:    DeviceTensor::zeros(vec![seq_len, d_model])?,
            ln_f_mean:   DeviceTensor::zeros(vec![seq_len])?,
            ln_f_rstd:   DeviceTensor::zeros(vec![seq_len])?,
            logits:      DeviceTensor::zeros(vec![seq_len, vocab_size])?,
            loss_tensor: DeviceTensor::zeros(vec![1])?,
            probs:       DeviceTensor::zeros(vec![seq_len, vocab_size])?,
            grad_one:    DeviceTensor::ones(vec![1])?,
            d_logits:    DeviceTensor::zeros(vec![seq_len, vocab_size])?,
            d_lm_head:   DeviceTensor::zeros(vec![vocab_size, d_model])?,
            d_ln_f_out:  DeviceTensor::zeros(vec![seq_len, d_model])?,
            d_ln_f_gamma: DeviceTensor::zeros(vec![d_model])?,
            d_ln_f_beta:  DeviceTensor::zeros(vec![d_model])?,
            layer_grads,
            d_tok_emb:   DeviceTensor::zeros(vec![vocab_size, d_model])?,
            d_pos_emb:   DeviceTensor::zeros(vec![seq_len, d_model])?,
            dx_comb:     DeviceTensor::zeros(vec![seq_len, d_model])?,
            step_count: 0,
            lr, beta1: 0.9, beta2: 0.95, eps: 1e-8, wd: 0.1,
            m_tok, m_pos, m_ln_f_gamma, m_ln_f_beta, m_lm_head,
            layer_moments,
            attn_contexts,
            pinned_loss: darkforest_cuda::memory::PinnedBuffer::new(std::mem::size_of::<f32>())?,
            // Pre-allocated host staging buffers — avoid Vec alloc per step
            h_input_u32:  vec![0u32; seq_len],
            h_target_u32: vec![0u32; seq_len],
            graph_exec: None,
        })
    }

    /// Run one complete training step: forward → loss → backward → AdamW update.
    /// Returns the scalar cross-entropy loss value.
    pub fn step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        assert_eq!(input_tokens.len(), self.seq_len);
        assert_eq!(target_tokens.len(), self.seq_len);
        self.step_count += 1;

        for (i, &tok) in input_tokens.iter().enumerate() { self.h_input_u32[i] = tok as u32; }
        for (i, &tok) in target_tokens.iter().enumerate() { self.h_target_u32[i] = tok as u32; }
        self.d_input_indices.upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;
        self.d_target_indices.upload_bytes(self.h_target_u32.as_ptr() as *const u8, self.seq_len * 4)?;

        self.step_inner()
    }

    pub fn step_graph(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        assert_eq!(input_tokens.len(), self.seq_len);
        assert_eq!(target_tokens.len(), self.seq_len);
        self.step_count += 1;

        for (i, &tok) in input_tokens.iter().enumerate() { self.h_input_u32[i] = tok as u32; }
        for (i, &tok) in target_tokens.iter().enumerate() { self.h_target_u32[i] = tok as u32; }
        self.d_input_indices.upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;
        self.d_target_indices.upload_bytes(self.h_target_u32.as_ptr() as *const u8, self.seq_len * 4)?;

        if self.graph_exec.is_none() {
            if self.step_count == 1 {
                // Warm up on step 1 to initialize cuBLAS handles
                self.step_inner()?;
            } else {
                let graph = darkforest_cuda::CudaGraphExec::capture(|| {
                    self.step_inner().map(|_| ())
                })?;
                self.graph_exec = Some(graph);
            }
        } else {
            self.graph_exec.as_ref().unwrap().launch()?;
        }
        
        Ok(self.pinned_loss.read_f32())
    }

    fn step_inner(&mut self) -> Result<f32> {
        // ---------------------------------------------------------------
        // FORWARD PASS
        // ---------------------------------------------------------------
        let scale = (self.d_model as f32 / self.n_heads as f32).powf(-0.5);

        // Embeddings
        DeviceTensor::embedding_lookup_device_indices(&self.d_input_indices, self.seq_len, &self.tok_emb, &mut self.tok_out)?;
        DeviceTensor::embedding_lookup_device_indices(&self.d_pos_indices, self.seq_len, &self.pos_emb, &mut self.pos_out)?;
        // x[0] = tok + pos
        self.tok_out.add_into(&self.pos_out, &mut self.x_in[0])?;

        // Transformer blocks
        for l in 0..self.n_layers {
            let lw = &self.layers[l];
            let la = &mut self.layer_acts[l];

            // LayerNorm 1
            self.x_in[l].layernorm_into(
                Some(&lw.ln1_gamma),
                Some(&lw.ln1_beta),
                &mut la.ln1_out,
                &mut la.ln1_mean,
                &mut la.ln1_rstd,
            )?;

            // Q, K, V projections
            la.ln1_out.linear_into(&lw.wq, Some(&lw.bq), &mut la.q)?;
            la.ln1_out.linear_into(&lw.wk, Some(&lw.bk), &mut la.k)?;
            la.ln1_out.linear_into(&lw.wv, Some(&lw.bv), &mut la.v)?;

            // Multi-Head Flash Attention
            self.attn_contexts[l].forward_device_into(
                &la.q,
                &la.k,
                &la.v,
                scale,
                true,
                &mut la.attn_out,
            )?;

            // Output projection + first residual
            la.attn_out.linear_into(&lw.wo, Some(&lw.bo), &mut la.proj_out)?;
            self.x_in[l].add_into(&la.proj_out, &mut la.x_mid)?;

            // LayerNorm 2
            la.x_mid.layernorm_into(
                Some(&lw.ln2_gamma),
                Some(&lw.ln2_beta),
                &mut la.ln2_out,
                &mut la.ln2_mean,
                &mut la.ln2_rstd,
            )?;

            // MLP: linear → gelu → linear
            la.ln2_out.linear_into(&lw.w1, Some(&lw.b1), &mut la.mlp_h)?;
            la.mlp_h.gelu_into(&mut la.mlp_a)?;
            la.mlp_a.linear_into(&lw.w2, Some(&lw.b2), &mut la.mlp_out)?;

            // Second residual
            la.x_mid.add_into(&la.mlp_out, &mut self.x_in[l + 1])?;
        }

        // Final LayerNorm + LM head
        self.x_in[self.n_layers].layernorm_into(
            Some(&self.ln_f_gamma),
            Some(&self.ln_f_beta),
            &mut self.ln_f_out,
            &mut self.ln_f_mean,
            &mut self.ln_f_rstd,
        )?;
        self.ln_f_out.linear_into(&self.lm_head, None, &mut self.logits)?;

        // Cross-entropy loss (in-place zero allocation)
        DeviceTensor::cross_entropy_device_targets(
            &self.logits,
            &self.d_target_indices,
            &mut self.probs,
            &mut self.loss_tensor,
        )?;
        self.loss_tensor.async_download_scalar_f32(&self.pinned_loss)?;

        // ---------------------------------------------------------------
        // BACKWARD PASS
        // ---------------------------------------------------------------
        // dlogits from cross-entropy
        DeviceTensor::cross_entropy_backward_device_targets(
            &self.probs,
            &self.d_target_indices,
            &self.grad_one,
            &mut self.d_logits,
        )?;

        // LM head backward: d(ln_f_out), d(lm_head)
        DeviceTensor::linear_backward_into(
            &self.ln_f_out,
            &self.lm_head,
            &self.d_logits,
            &mut self.d_ln_f_out,
            &mut self.d_lm_head,
            None,
        )?;

        // Final LayerNorm backward
        DeviceTensor::layernorm_backward_into(
            &self.d_ln_f_out,
            &self.x_in[self.n_layers],
            Some(&self.ln_f_gamma),
            &self.ln_f_mean,
            &self.ln_f_rstd,
            &mut self.dx_comb,
            Some(&mut self.d_ln_f_gamma),
            Some(&mut self.d_ln_f_beta),
        )?;

        // Transformer blocks backward (reverse order)
        for l in (0..self.n_layers).rev() {
            let lw = &self.layers[l];
            let la = &self.layer_acts[l];
            let lg = &mut self.layer_grads[l];

            // --- MLP backward ---
            lg.d_mlp_out.copy_from(&self.dx_comb)?;
            lg.dx.copy_from(&self.dx_comb)?;
            let d_x_mid_from_above = &mut lg.dx;

            // w2 backward: d(mlp_a), d(w2), d(b2)
            DeviceTensor::linear_backward_into(
                &la.mlp_a,
                &lw.w2,
                &lg.d_mlp_out,
                &mut lg.d_mlp_a,
                &mut lg.d_w2,
                Some(&mut lg.d_b2),
            )?;

            // gelu backward: d(mlp_h)
            la.mlp_h.gelu_backward_into(&lg.d_mlp_a, &mut lg.d_mlp_h)?;

            // w1 backward: d(ln2_out), d(w1), d(b1)
            DeviceTensor::linear_backward_into(
                &la.ln2_out,
                &lw.w1,
                &lg.d_mlp_h,
                &mut lg.d_ln2_out,
                &mut lg.d_w1,
                Some(&mut lg.d_b1),
            )?;

            // LayerNorm 2 backward into d_x_mid (accumulated in dx_comb)
            DeviceTensor::layernorm_backward_into(
                &lg.d_ln2_out,
                &la.x_mid,
                Some(&lw.ln2_gamma),
                &la.ln2_mean,
                &la.ln2_rstd,
                &mut self.dx_comb,
                Some(&mut lg.d_ln2_gamma),
                Some(&mut lg.d_ln2_beta),
            )?;

            // Combine d_x_mid: residual path + MLP path
            d_x_mid_from_above.add_inplace(&self.dx_comb)?;

            // --- Attention backward ---
            lg.d_proj_out.copy_from(d_x_mid_from_above)?;
            // Output projection backward: d(attn_out), d(wo), d(bo)
            DeviceTensor::linear_backward_into(
                &la.attn_out,
                &lw.wo,
                &lg.d_proj_out,
                &mut lg.d_attn_out,
                &mut lg.d_wo,
                Some(&mut lg.d_bo),
            )?;

            // FlashAttention backward: d(q), d(k), d(v)
            self.attn_contexts[l].backward_device_into(
                &la.q,
                &la.k,
                &la.v,
                &lg.d_attn_out,
                scale,
                true,
                &mut lg.d_q,
                &mut lg.d_k,
                &mut lg.d_v,
            )?;

            // Q proj backward into d_ln1_out (beta = 0.0)
            DeviceTensor::linear_backward_into(
                &la.ln1_out,
                &lw.wq,
                &lg.d_q,
                &mut lg.d_ln1_out,
                &mut lg.d_wq,
                Some(&mut lg.d_bq),
            )?;

            // K proj backward accumulating directly into d_ln1_out (beta = 1.0)
            DeviceTensor::linear_backward_accumulate_into(
                &la.ln1_out,
                &lw.wk,
                &lg.d_k,
                &mut lg.d_ln1_out,
                &mut lg.d_wk,
                Some(&mut lg.d_bk),
            )?;

            // V proj backward accumulating directly into d_ln1_out (beta = 1.0)
            DeviceTensor::linear_backward_accumulate_into(
                &la.ln1_out,
                &lw.wv,
                &lg.d_v,
                &mut lg.d_ln1_out,
                &mut lg.d_wv,
                Some(&mut lg.d_bv),
            )?;

            // LayerNorm 1 backward into self.dx_comb
            DeviceTensor::layernorm_backward_into(
                &lg.d_ln1_out,
                &self.x_in[l],
                Some(&lw.ln1_gamma),
                &la.ln1_mean,
                &la.ln1_rstd,
                &mut self.dx_comb,
                Some(&mut lg.d_ln1_gamma),
                Some(&mut lg.d_ln1_beta),
            )?;

            // Combine dx for x_in[l]: dx_comb = d_x_in_from_residual (in lg.dx) + d_x_in_from_ln1
            self.dx_comb.add_inplace(&lg.dx)?;
        }


        // Embedding backward — self.dx_comb holds d(x_in[0]) after transformer loop
        DeviceTensor::embedding_backward_device_indices(
            &self.d_input_indices,
            self.seq_len,
            &self.dx_comb,
            &mut self.d_tok_emb,
            self.vocab_size,
            self.d_model,
        )?;
        DeviceTensor::embedding_backward_device_indices(
            &self.d_pos_indices,
            self.seq_len,
            &self.dx_comb,
            &mut self.d_pos_emb,
            self.seq_len,
            self.d_model,
        )?;

        // ---------------------------------------------------------------
        // ADAMW UPDATE — all on device, no CPU sync
        // ---------------------------------------------------------------
        let t = self.step_count as f32;
        let lr = self.lr;
        let (b1, b2, eps, wd) = (self.beta1, self.beta2, self.eps, self.wd);
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);

        macro_rules! update {
            ($p:expr, $g:expr, $mom:expr) => {
                DeviceTensor::adamw_update(&$p, &$mom.m, &$mom.v, &$g, lr, b1, b2, eps, wd, bc1, bc2)?;
            };
        }

        update!(self.tok_emb,    self.d_tok_emb,    self.m_tok);
        update!(self.pos_emb,    self.d_pos_emb,    self.m_pos);
        update!(self.ln_f_gamma, self.d_ln_f_gamma, self.m_ln_f_gamma);
        update!(self.ln_f_beta,  self.d_ln_f_beta,  self.m_ln_f_beta);
        update!(self.lm_head,    self.d_lm_head,    self.m_lm_head);

        for l in 0..self.n_layers {
            let lw = &mut self.layers[l];
            let lg = &self.layer_grads[l];
            let lm = &mut self.layer_moments[l];
            update!(lw.ln1_gamma, lg.d_ln1_gamma, lm.ln1_gamma);
            update!(lw.ln1_beta,  lg.d_ln1_beta,  lm.ln1_beta);
            update!(lw.wq, lg.d_wq, lm.wq);
            update!(lw.bq, lg.d_bq, lm.bq);
            update!(lw.wk, lg.d_wk, lm.wk);
            update!(lw.bk, lg.d_bk, lm.bk);
            update!(lw.wv, lg.d_wv, lm.wv);
            update!(lw.bv, lg.d_bv, lm.bv);
            update!(lw.wo, lg.d_wo, lm.wo);
            update!(lw.bo, lg.d_bo, lm.bo);
            update!(lw.ln2_gamma, lg.d_ln2_gamma, lm.ln2_gamma);
            update!(lw.ln2_beta,  lg.d_ln2_beta,  lm.ln2_beta);
            update!(lw.w1, lg.d_w1, lm.w1);
            update!(lw.b1, lg.d_b1, lm.b1);
            update!(lw.w2, lg.d_w2, lm.w2);
            update!(lw.b2, lg.d_b2, lm.b2);
        }

        Ok(self.pinned_loss.read_f32())
    }
}


/// Stub for non-CUDA builds (compile-time no-op).
#[cfg(not(feature = "cuda"))]
pub struct StaticGPT2;

#[cfg(not(feature = "cuda"))]
impl StaticGPT2 {
    pub fn new(_vs: usize, _dm: usize, _nh: usize, _nl: usize, _df: usize, _sl: usize, _lr: f32)
        -> Result<Self> { Ok(Self) }
    pub fn step(&mut self, _in: &[usize], _tgt: &[usize]) -> Result<f32> {
        Err(anyhow!("CUDA required for StaticGPT2"))
    }
}
