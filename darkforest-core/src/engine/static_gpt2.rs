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
    wq: DeviceTensor,
    bq: DeviceTensor,
    wk: DeviceTensor,
    bk: DeviceTensor,
    wv: DeviceTensor,
    bv: DeviceTensor,
    wo: DeviceTensor,
    bo: DeviceTensor,
    // LayerNorm 2
    ln2_gamma: DeviceTensor,
    ln2_beta: DeviceTensor,
    // MLP
    w1: DeviceTensor,
    b1: DeviceTensor, // [d_ff, d_model] / [d_ff]
    w2: DeviceTensor,
    b2: DeviceTensor, // [d_model, d_ff] / [d_model]
}

/// Per-layer forward activation buffers (pre-allocated, reused every step).
#[cfg(feature = "cuda")]
struct LayerActivations {
    ln1_out: DeviceTensor,  // [seq, d_model]
    ln1_mean: DeviceTensor, // [seq]
    ln1_rstd: DeviceTensor, // [seq]
    q: DeviceTensor,        // [seq, d_model]
    k: DeviceTensor,        // [seq, d_model]
    v: DeviceTensor,        // [seq, d_model]
    attn_out: DeviceTensor, // [seq, d_model]
    proj_out: DeviceTensor, // [seq, d_model]
    x_mid: DeviceTensor,    // [seq, d_model]  (x after attn residual)
    ln2_out: DeviceTensor,  // [seq, d_model]
    ln2_mean: DeviceTensor, // [seq]
    ln2_rstd: DeviceTensor, // [seq]
    mlp_h: DeviceTensor,    // [seq, d_ff]
    mlp_a: DeviceTensor,    // [seq, d_ff]  (after gelu)
    mlp_out: DeviceTensor,  // [seq, d_model]
}

/// Per-layer gradient buffers (pre-allocated, reused every step).
#[cfg(feature = "cuda")]
struct LayerGrads {
    // upstream gradient flowing into this layer from above
    dx: DeviceTensor, // [seq, d_model]
    // MLP backward
    d_mlp_out: DeviceTensor,   // [seq, d_model]
    d_mlp_a: DeviceTensor,     // [seq, d_ff]
    d_mlp_h: DeviceTensor,     // [seq, d_ff]
    d_w2: DeviceTensor,        // [d_model, d_ff]
    d_b2: DeviceTensor,        // [d_model]
    d_w1: DeviceTensor,        // [d_ff, d_model]
    d_b1: DeviceTensor,        // [d_ff]
    d_ln2_out: DeviceTensor,   // [seq, d_model]
    d_ln2_gamma: DeviceTensor, // [d_model]
    d_ln2_beta: DeviceTensor,  // [d_model]
    // Attention backward
    d_proj_out: DeviceTensor,  // [seq, d_model]
    d_attn_out: DeviceTensor,  // [seq, d_model]
    d_q: DeviceTensor,         // [seq, d_model]
    d_k: DeviceTensor,         // [seq, d_model]
    d_v: DeviceTensor,         // [seq, d_model]
    d_wo: DeviceTensor,        // [d_model, d_model]
    d_bo: DeviceTensor,        // [d_model]
    d_wq: DeviceTensor,        // [d_model, d_model]
    d_bq: DeviceTensor,        // [d_model]
    d_wk: DeviceTensor,        // [d_model, d_model]
    d_bk: DeviceTensor,        // [d_model]
    d_wv: DeviceTensor,        // [d_model, d_model]
    d_bv: DeviceTensor,        // [d_model]
    d_ln1_out: DeviceTensor,   // [seq, d_model]
    d_ln1_gamma: DeviceTensor, // [d_model]
    d_ln1_beta: DeviceTensor,  // [d_model]
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
    tok_emb: DeviceTensor,    // [vocab_size, d_model]
    pos_emb: DeviceTensor,    // [seq_len, d_model]
    ln_f_gamma: DeviceTensor, // [d_model]
    ln_f_beta: DeviceTensor,  // [d_model]
    lm_head: DeviceTensor,    // [vocab_size, d_model]

    // Per-layer weights
    layers: Vec<LayerWeights>,

    // Forward activation buffers
    tok_out: DeviceTensor,   // [seq, d_model]
    pos_out: DeviceTensor,   // [seq, d_model]
    x_in: Vec<DeviceTensor>, // [n_layers+1][seq, d_model] — x entering each layer + final
    layer_acts: Vec<LayerActivations>,
    ln_f_out: DeviceTensor,    // [seq, d_model]
    ln_f_mean: DeviceTensor,   // [seq]
    ln_f_rstd: DeviceTensor,   // [seq]
    logits: DeviceTensor,      // [seq, vocab_size]
    loss_tensor: DeviceTensor, // [1]
    probs: DeviceTensor,       // [seq, vocab_size]
    grad_one: DeviceTensor,    // [1]

    // Pinned host buffer for async loss D2H (no CPU-GPU sync inside step)
    pinned_loss: darkforest_cuda::memory::PinnedBuffer,
    // Gradient buffers
    d_logits: DeviceTensor,     // [seq, vocab_size]
    d_lm_head: DeviceTensor,    // [vocab_size, d_model]
    d_ln_f_out: DeviceTensor,   // [seq, d_model]
    d_ln_f_gamma: DeviceTensor, // [d_model]
    d_ln_f_beta: DeviceTensor,  // [d_model]
    layer_grads: Vec<LayerGrads>,
    d_tok_emb: DeviceTensor, // [vocab_size, d_model]
    d_pos_emb: DeviceTensor, // [seq_len, d_model]

    // Reusable dx residual accumulator
    dx_comb: DeviceTensor, // [seq, d_model]

    // AdamW state
    step_count: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    /// Optional global gradient norm clipping. Set to `Some(max_norm)` to enable.
    max_grad_norm: Option<f32>,
    // moment buffers — one Moments per weight tensor
    m_tok: Moments,
    m_pos: Moments,
    m_ln_f_gamma: Moments,
    m_ln_f_beta: Moments,
    m_lm_head: Moments,
    layer_moments: Vec<LayerMoments>,

    // Attention contexts (preallocated flash-attention scratch)
    attn_contexts: Vec<std::sync::Arc<AttentionContext>>,

    // CUDA Graph executable for the step
    graph_exec: Option<darkforest_cuda::CudaGraphExec>,

    // Pre-allocated Key-Value cache for single-token autoregressive decoding
    decode_k_cache: Vec<DeviceTensor>, // [n_layers][seq_len, d_model]
    decode_v_cache: Vec<DeviceTensor>, // [n_layers][seq_len, d_model]

    // Pre-allocated single-token decode activation tensors (batch = 1)
    decode_tok_out: DeviceTensor,   // [1, d_model]
    decode_pos_out: DeviceTensor,   // [1, d_model]
    decode_x: DeviceTensor,         // [1, d_model]
    decode_ln1_out: DeviceTensor,   // [1, d_model]
    decode_ln1_mean: DeviceTensor,  // [1]
    decode_ln1_rstd: DeviceTensor,  // [1]
    decode_q: DeviceTensor,         // [1, d_model]
    decode_k: DeviceTensor,         // [1, d_model]
    decode_v: DeviceTensor,         // [1, d_model]
    decode_attn_out: DeviceTensor,  // [1, d_model]
    decode_proj_out: DeviceTensor,  // [1, d_model]
    decode_x_mid: DeviceTensor,     // [1, d_model]
    decode_ln2_out: DeviceTensor,   // [1, d_model]
    decode_ln2_mean: DeviceTensor,  // [1]
    decode_ln2_rstd: DeviceTensor,  // [1]
    decode_mlp_h: DeviceTensor,     // [1, d_ff]
    decode_mlp_a: DeviceTensor,     // [1, d_ff]
    decode_mlp_out: DeviceTensor,   // [1, d_model]
    decode_ln_f_out: DeviceTensor,  // [1, d_model]
    decode_ln_f_mean: DeviceTensor, // [1]
    decode_ln_f_rstd: DeviceTensor, // [1]
    decode_logits: DeviceTensor,    // [1, vocab_size]

    // Single-token index buffers and device reduction buffers
    d_single_tok: darkforest_cuda::memory::CudaBuffer,
    d_single_pos: darkforest_cuda::memory::CudaBuffer,
    d_argmax_out: darkforest_cuda::memory::CudaBuffer,

    // Page-locked host buffers for zero-copy D2H transfers
    pinned_argmax: darkforest_cuda::memory::PinnedBuffer,
    pinned_logits: darkforest_cuda::memory::PinnedBuffer,
}


/// AdamW moment buffers for all weights in one transformer layer.
#[cfg(feature = "cuda")]
struct LayerMoments {
    ln1_gamma: Moments,
    ln1_beta: Moments,
    wq: Moments,
    bq: Moments,
    wk: Moments,
    bk: Moments,
    wv: Moments,
    bv: Moments,
    wo: Moments,
    bo: Moments,
    ln2_gamma: Moments,
    ln2_beta: Moments,
    w1: Moments,
    b1: Moments,
    w2: Moments,
    b2: Moments,
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
            let ln1_beta = DeviceTensor::zeros(vec![d_model])?;
            let ln2_gamma = DeviceTensor::ones(vec![d_model])?;
            let ln2_beta = DeviceTensor::zeros(vec![d_model])?;

            layer_moments.push(LayerMoments {
                ln1_gamma: Moments::for_tensor(&ln1_gamma)?,
                ln1_beta: Moments::for_tensor(&ln1_beta)?,
                wq: Moments::for_tensor(&wq)?,
                bq: Moments::for_tensor(&bq)?,
                wk: Moments::for_tensor(&wk)?,
                bk: Moments::for_tensor(&bk)?,
                wv: Moments::for_tensor(&wv)?,
                bv: Moments::for_tensor(&bv)?,
                wo: Moments::for_tensor(&wo)?,
                bo: Moments::for_tensor(&bo)?,
                ln2_gamma: Moments::for_tensor(&ln2_gamma)?,
                ln2_beta: Moments::for_tensor(&ln2_beta)?,
                w1: Moments::for_tensor(&w1)?,
                b1: Moments::for_tensor(&b1)?,
                w2: Moments::for_tensor(&w2)?,
                b2: Moments::for_tensor(&b2)?,
            });

            layers.push(LayerWeights {
                ln1_gamma,
                ln1_beta,
                wq,
                bq,
                wk,
                bk,
                wv,
                bv,
                wo,
                bo,
                ln2_gamma,
                ln2_beta,
                w1,
                b1,
                w2,
                b2,
            });

            // Pre-allocate activation buffers for this layer
            layer_acts.push(LayerActivations {
                ln1_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                ln1_mean: DeviceTensor::zeros(vec![seq_len])?,
                ln1_rstd: DeviceTensor::zeros(vec![seq_len])?,
                q: DeviceTensor::zeros(vec![seq_len, d_model])?,
                k: DeviceTensor::zeros(vec![seq_len, d_model])?,
                v: DeviceTensor::zeros(vec![seq_len, d_model])?,
                attn_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                proj_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                x_mid: DeviceTensor::zeros(vec![seq_len, d_model])?,
                ln2_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                ln2_mean: DeviceTensor::zeros(vec![seq_len])?,
                ln2_rstd: DeviceTensor::zeros(vec![seq_len])?,
                mlp_h: DeviceTensor::zeros(vec![seq_len, d_ff])?,
                mlp_a: DeviceTensor::zeros(vec![seq_len, d_ff])?,
                mlp_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
            });

            // Pre-allocate gradient buffers
            layer_grads.push(LayerGrads {
                dx: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_mlp_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_mlp_a: DeviceTensor::zeros(vec![seq_len, d_ff])?,
                d_mlp_h: DeviceTensor::zeros(vec![seq_len, d_ff])?,
                d_w2: DeviceTensor::zeros(vec![d_model, d_ff])?,
                d_b2: DeviceTensor::zeros(vec![d_model])?,
                d_w1: DeviceTensor::zeros(vec![d_ff, d_model])?,
                d_b1: DeviceTensor::zeros(vec![d_ff])?,
                d_ln2_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_ln2_gamma: DeviceTensor::zeros(vec![d_model])?,
                d_ln2_beta: DeviceTensor::zeros(vec![d_model])?,
                d_proj_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_attn_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
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
                d_ln1_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
                d_ln1_gamma: DeviceTensor::zeros(vec![d_model])?,
                d_ln1_beta: DeviceTensor::zeros(vec![d_model])?,
            });

            attn_contexts.push(std::sync::Arc::new(AttentionContext::new(
                n_heads, seq_len, d_head,
            )?));
        }

        // x buffers entering each layer + after final ln
        let mut x_in = Vec::with_capacity(n_layers + 1);
        for _ in 0..=n_layers {
            x_in.push(DeviceTensor::zeros(vec![seq_len, d_model])?);
        }

        let allocator = std::sync::Arc::new(darkforest_cuda::memory::GpuAllocator);
        let d_input_indices = darkforest_cuda::memory::CudaBuffer::new(
            seq_len * std::mem::size_of::<u32>(),
            allocator.clone(),
        )?;
        let d_pos_indices = darkforest_cuda::memory::CudaBuffer::new(
            seq_len * std::mem::size_of::<u32>(),
            allocator.clone(),
        )?;
        let d_target_indices = darkforest_cuda::memory::CudaBuffer::new(
            seq_len * std::mem::size_of::<u32>(),
            allocator,
        )?;

        // Preload pos indices [0, 1, ..., seq_len-1] once
        let pos_data: Vec<u32> = (0..seq_len as u32).collect();
        d_pos_indices.upload_bytes(
            pos_data.as_ptr() as *const u8,
            seq_len * std::mem::size_of::<u32>(),
        )?;

        let m_tok = Moments::for_tensor(&tok_emb)?;
        let m_pos = Moments::for_tensor(&pos_emb)?;
        let m_ln_f_gamma = Moments::for_tensor(&ln_f_gamma)?;
        let m_ln_f_beta = Moments::for_tensor(&ln_f_beta)?;
        let m_lm_head = Moments::for_tensor(&lm_head)?;

        // Pre-allocate Key-Value cache for autoregressive decoding
        let mut decode_k_cache = Vec::with_capacity(n_layers);
        let mut decode_v_cache = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            decode_k_cache.push(DeviceTensor::zeros(vec![seq_len, d_model])?);
            decode_v_cache.push(DeviceTensor::zeros(vec![seq_len, d_model])?);
        }

        let d_single_tok = darkforest_cuda::memory::CudaBuffer::new(
            std::mem::size_of::<u32>(),
            allocator.clone(),
        )?;
        let d_single_pos = darkforest_cuda::memory::CudaBuffer::new(
            std::mem::size_of::<u32>(),
            allocator.clone(),
        )?;
        let d_argmax_out = darkforest_cuda::memory::CudaBuffer::new(
            std::mem::size_of::<u32>(),
            allocator.clone(),
        )?;
        let pinned_argmax = darkforest_cuda::memory::PinnedBuffer::new(std::mem::size_of::<u32>())?;
        let pinned_logits = darkforest_cuda::memory::PinnedBuffer::new(vocab_size * std::mem::size_of::<f32>())?;

        Ok(Self {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            seq_len,
            d_input_indices,
            d_pos_indices,
            d_target_indices,
            tok_emb,
            pos_emb,
            ln_f_gamma,
            ln_f_beta,
            lm_head,
            layers,
            layer_acts,
            tok_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
            pos_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
            x_in,
            ln_f_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
            ln_f_mean: DeviceTensor::zeros(vec![seq_len])?,
            ln_f_rstd: DeviceTensor::zeros(vec![seq_len])?,
            logits: DeviceTensor::zeros(vec![seq_len, vocab_size])?,
            loss_tensor: DeviceTensor::zeros(vec![1])?,
            probs: DeviceTensor::zeros(vec![seq_len, vocab_size])?,
            grad_one: DeviceTensor::ones(vec![1])?,
            d_logits: DeviceTensor::zeros(vec![seq_len, vocab_size])?,
            d_lm_head: DeviceTensor::zeros(vec![vocab_size, d_model])?,
            d_ln_f_out: DeviceTensor::zeros(vec![seq_len, d_model])?,
            d_ln_f_gamma: DeviceTensor::zeros(vec![d_model])?,
            d_ln_f_beta: DeviceTensor::zeros(vec![d_model])?,
            layer_grads,
            d_tok_emb: DeviceTensor::zeros(vec![vocab_size, d_model])?,
            d_pos_emb: DeviceTensor::zeros(vec![seq_len, d_model])?,
            dx_comb: DeviceTensor::zeros(vec![seq_len, d_model])?,
            step_count: 0,
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            wd: 0.1,
            max_grad_norm: None,
            m_tok,
            m_pos,
            m_ln_f_gamma,
            m_ln_f_beta,
            m_lm_head,
            layer_moments,
            attn_contexts,
            pinned_loss: darkforest_cuda::memory::PinnedBuffer::new(std::mem::size_of::<f32>())?,
            // Pre-allocated host staging buffers — avoid Vec alloc per step
            h_input_u32: vec![0u32; seq_len],
            h_target_u32: vec![0u32; seq_len],
            graph_exec: None,

            // KV-Cache decoding buffers
            decode_k_cache,
            decode_v_cache,
            decode_tok_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_pos_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_x: DeviceTensor::zeros(vec![1, d_model])?,
            decode_ln1_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_ln1_mean: DeviceTensor::zeros(vec![1])?,
            decode_ln1_rstd: DeviceTensor::zeros(vec![1])?,
            decode_q: DeviceTensor::zeros(vec![1, d_model])?,
            decode_k: DeviceTensor::zeros(vec![1, d_model])?,
            decode_v: DeviceTensor::zeros(vec![1, d_model])?,
            decode_attn_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_proj_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_x_mid: DeviceTensor::zeros(vec![1, d_model])?,
            decode_ln2_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_ln2_mean: DeviceTensor::zeros(vec![1])?,
            decode_ln2_rstd: DeviceTensor::zeros(vec![1])?,
            decode_mlp_h: DeviceTensor::zeros(vec![1, d_ff])?,
            decode_mlp_a: DeviceTensor::zeros(vec![1, d_ff])?,
            decode_mlp_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_ln_f_out: DeviceTensor::zeros(vec![1, d_model])?,
            decode_ln_f_mean: DeviceTensor::zeros(vec![1])?,
            decode_ln_f_rstd: DeviceTensor::zeros(vec![1])?,
            decode_logits: DeviceTensor::zeros(vec![1, vocab_size])?,
            d_single_tok,
            d_single_pos,
            d_argmax_out,
            pinned_argmax,
            pinned_logits,
        })
    }


    /// Run one complete training step: forward → loss → backward → AdamW update.
    /// Returns the scalar cross-entropy loss value.
    pub fn step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        if input_tokens.len() != self.seq_len || target_tokens.len() != self.seq_len {
            return Err(anyhow!(
                "StaticGPT2 context length violation in step(): expected exactly {} tokens, got {} input and {} target tokens.",
                self.seq_len,
                input_tokens.len(),
                target_tokens.len()
            ));
        }
        self.step_count += 1;

        for (i, &tok) in input_tokens.iter().enumerate() {
            self.h_input_u32[i] = tok as u32;
        }
        for (i, &tok) in target_tokens.iter().enumerate() {
            self.h_target_u32[i] = tok as u32;
        }
        self.d_input_indices
            .upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;
        self.d_target_indices
            .upload_bytes(self.h_target_u32.as_ptr() as *const u8, self.seq_len * 4)?;

        self.step_inner()
    }

    pub fn step_graph(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        if input_tokens.len() != self.seq_len || target_tokens.len() != self.seq_len {
            return Err(anyhow!(
                "StaticGPT2 context length violation: expected exactly {} tokens (pre-allocated static horizon), got {} input and {} target tokens. Reallocate StaticGPT2 to change sequence horizon.",
                self.seq_len,
                input_tokens.len(),
                target_tokens.len()
            ));
        }
        self.step_count += 1;

        for (i, &tok) in input_tokens.iter().enumerate() {
            self.h_input_u32[i] = tok as u32;
        }
        for (i, &tok) in target_tokens.iter().enumerate() {
            self.h_target_u32[i] = tok as u32;
        }
        self.d_input_indices
            .upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;
        self.d_target_indices
            .upload_bytes(self.h_target_u32.as_ptr() as *const u8, self.seq_len * 4)?;

        if self.graph_exec.is_none() {
            if self.step_count == 1 {
                // Warm up on step 1 to initialize cuBLAS handles
                self.step_inner()?;
            } else {
                let graph =
                    darkforest_cuda::CudaGraphExec::capture(|| self.step_inner().map(|_| ()))?;
                self.graph_exec = Some(graph);
            }
        } else {
            self.graph_exec.as_ref().unwrap().launch()?;
        }

        Ok(self.pinned_loss.read_f32())
    }

    /// Run one validation / evaluation step: forward pass and cross-entropy loss computation.
    ///
    /// Computes the exact validation loss without materializing backward gradients or
    /// executing AdamW weight updates.
    pub fn eval_step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        if input_tokens.len() != self.seq_len || target_tokens.len() != self.seq_len {
            return Err(anyhow!(
                "StaticGPT2 context length violation in eval_step(): expected exactly {} tokens, got {} input and {} target tokens.",
                self.seq_len,
                input_tokens.len(),
                target_tokens.len()
            ));
        }

        for (i, &tok) in input_tokens.iter().enumerate() {
            self.h_input_u32[i] = tok as u32;
        }
        for (i, &tok) in target_tokens.iter().enumerate() {
            self.h_target_u32[i] = tok as u32;
        }
        self.d_input_indices
            .upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;
        self.d_target_indices
            .upload_bytes(self.h_target_u32.as_ptr() as *const u8, self.seq_len * 4)?;

        self.forward_inner()?;

        DeviceTensor::cross_entropy_device_targets(
            &self.logits,
            &self.d_target_indices,
            &mut self.probs,
            &mut self.loss_tensor,
        )?;
        self.loss_tensor
            .async_download_scalar_f32(&self.pinned_loss)?;
        crate::cuda_sync()?;

        Ok(self.pinned_loss.read_f32())
    }

    /// Run the forward inference pass for the input tokens.
    ///
    /// Returns the full unnormalized output logits matrix of shape `[seq_len, vocab_size]`
    /// as a flat `Vec<f32>` in row-major order.
    pub fn forward(&mut self, input_tokens: &[usize]) -> Result<Vec<f32>> {
        if input_tokens.len() != self.seq_len {
            return Err(anyhow!(
                "StaticGPT2 forward length violation: expected exactly {} tokens, got {}.",
                self.seq_len,
                input_tokens.len()
            ));
        }
        for (i, &tok) in input_tokens.iter().enumerate() {
            self.h_input_u32[i] = tok as u32;
        }
        self.d_input_indices
            .upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;

        self.forward_inner()?;
        crate::cuda_sync()?;
        self.logits.download()
    }

    /// Extract unnormalized prediction logits `[vocab_size]` for the final token position in the context.
    ///
    /// Optimized to download only the final row slice (vocab_size floats = ~200 KB) instead of the
    /// full sequence logits tensor (up to 205 MB), eliminating PCIe bus saturation.
    pub fn forward_last_logits(&mut self, input_tokens: &[usize]) -> Result<Vec<f32>> {
        if input_tokens.len() != self.seq_len {
            return Err(anyhow!(
                "StaticGPT2 forward length violation: expected exactly {} tokens, got {}.",
                self.seq_len,
                input_tokens.len()
            ));
        }
        for (i, &tok) in input_tokens.iter().enumerate() {
            self.h_input_u32[i] = tok as u32;
        }
        self.d_input_indices
            .upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;

        self.forward_inner()?;
        crate::cuda_sync()?;
        self.logits
            .download_slice((self.seq_len - 1) * self.vocab_size, self.vocab_size)
    }

    /// Predict the greedy next-token index for the given input context (argmax over final logits).
    ///
    /// Executes device-side argmax reduction to minimize PCIe transfer to a single 4-byte scalar.
    pub fn predict(&mut self, input_tokens: &[usize]) -> Result<usize> {
        let last_logits = self.forward_last_logits(input_tokens)?;
        Ok(crate::engine::argmax(&last_logits))
    }

    /// Single-token autoregressive decoding step at position `pos`.
    ///
    /// Executes the forward pass for 1 token:
    /// - Embeddings: tok + pos
    /// - 12 Transformer blocks with KV-cache attention
    /// - Final LayerNorm + LM Head GEMV
    /// Logits reside in `self.decode_logits` [1, vocab_size].
    pub fn decode_step(&mut self, token: usize, pos: usize) -> Result<()> {
        if pos >= self.seq_len {
            return Err(anyhow!(
                "decode_step: pos {} exceeds max seq_len {}",
                pos,
                self.seq_len
            ));
        }
        let scale = (self.d_model as f32 / self.n_heads as f32).powf(-0.5);

        // Upload single token index and single position index
        let tok_u32 = token as u32;
        let pos_u32 = pos as u32;
        self.d_single_tok
            .upload_bytes(&tok_u32 as *const u32 as *const u8, 4)?;
        self.d_single_pos
            .upload_bytes(&pos_u32 as *const u32 as *const u8, 4)?;

        // Embeddings: 1 token lookup from tok_emb, 1 position lookup from pos_emb
        DeviceTensor::embedding_lookup_device_indices(
            &self.d_single_tok,
            1,
            &self.tok_emb,
            &mut self.decode_tok_out,
        )?;
        DeviceTensor::embedding_lookup_device_indices(
            &self.d_single_pos,
            1,
            &self.pos_emb,
            &mut self.decode_pos_out,
        )?;

        // x = tok + pos
        self.decode_tok_out.add_into(&self.decode_pos_out, &mut self.decode_x)?;

        // Transformer blocks
        for l in 0..self.n_layers {
            let lw = &self.layers[l];

            // 1. LayerNorm 1
            self.decode_x.layernorm_into(
                Some(&lw.ln1_gamma),
                Some(&lw.ln1_beta),
                &mut self.decode_ln1_out,
                &mut self.decode_ln1_mean,
                &mut self.decode_ln1_rstd,
            )?;

            // 2. Q, K, V projections via high-throughput GEMV
            self.decode_ln1_out.gemv_into(&lw.wq, Some(&lw.bq), &mut self.decode_q)?;
            self.decode_ln1_out.gemv_into(&lw.wk, Some(&lw.bk), &mut self.decode_k)?;
            self.decode_ln1_out.gemv_into(&lw.wv, Some(&lw.bv), &mut self.decode_v)?;

            // 3. Store new K and V into KV-cache at index `pos`
            self.decode_k.copy_row_into(&mut self.decode_k_cache[l], pos)?;
            self.decode_v.copy_row_into(&mut self.decode_v_cache[l], pos)?;

            // 4. KV-Cache Single-Token Attention
            self.attn_contexts[l].decode_kv_into(
                &self.decode_q,
                &self.decode_k_cache[l],
                &self.decode_v_cache[l],
                pos + 1,
                scale,
                &mut self.decode_attn_out,
            )?;

            // 5. Output projection GEMV + residual add
            self.decode_attn_out.gemv_into(&lw.wo, Some(&lw.bo), &mut self.decode_proj_out)?;
            self.decode_x.add_into(&self.decode_proj_out, &mut self.decode_x_mid)?;

            // 6. LayerNorm 2
            self.decode_x_mid.layernorm_into(
                Some(&lw.ln2_gamma),
                Some(&lw.ln2_beta),
                &mut self.decode_ln2_out,
                &mut self.decode_ln2_mean,
                &mut self.decode_ln2_rstd,
            )?;

            // 7. MLP: GEMV w1 -> GELU -> GEMV w2
            self.decode_ln2_out.gemv_into(&lw.w1, Some(&lw.b1), &mut self.decode_mlp_h)?;
            self.decode_mlp_h.gelu_into(&mut self.decode_mlp_a)?;
            self.decode_mlp_a.gemv_into(&lw.w2, Some(&lw.b2), &mut self.decode_mlp_out)?;

            // 8. Second residual add
            self.decode_x_mid.add_into(&self.decode_mlp_out, &mut self.decode_x)?;
        }

        // Final LayerNorm
        self.decode_x.layernorm_into(
            Some(&self.ln_f_gamma),
            Some(&self.ln_f_beta),
            &mut self.decode_ln_f_out,
            &mut self.decode_ln_f_mean,
            &mut self.decode_ln_f_rstd,
        )?;

        // LM Head GEMV: [1, d_model] @ [vocab_size, d_model] -> [1, vocab_size]
        self.decode_ln_f_out.gemv_into(&self.lm_head, None, &mut self.decode_logits)?;

        Ok(())
    }

    pub(crate) fn layer_wq_mut(&mut self, l: usize) -> &mut DeviceTensor {
        &mut self.layers[l].wq
    }
    pub(crate) fn layer_wk_mut(&mut self, l: usize) -> &mut DeviceTensor {
        &mut self.layers[l].wk
    }
    pub(crate) fn layer_wv_mut(&mut self, l: usize) -> &mut DeviceTensor {
        &mut self.layers[l].wv
    }
    pub(crate) fn layer_wo_mut(&mut self, l: usize) -> &mut DeviceTensor {
        &mut self.layers[l].wo
    }

    /// Seriously fast autoregressive generation using KV cache and zero-allocation decode loop.
    ///
    /// Computes prompt prefill in O(P) steps and subsequent tokens in O(1) single-token steps.
    /// Eliminates full-sequence recomputation (O(S^2) -> O(S)) and minimizes PCIe traffic
    /// down to 4 bytes per token for greedy decoding.
    pub fn generate_fast(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &crate::engine::SamplerConfig,
    ) -> Result<Vec<usize>> {
        let report = self.generate_stream(prompt_tokens, max_new_tokens, cfg, |_| true)?;
        Ok(report.tokens)
    }

    /// Streaming autoregressive token generation with per-token callback.
    ///
    /// Invokes `on_token(token)` as each token is generated. If `on_token` returns `false`,
    /// generation terminates early. Returns a complete `GenerationReport` with metrics.
    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &crate::engine::SamplerConfig,
        mut on_token: F,
    ) -> Result<crate::engine::generate::GenerationReport>
    where
        F: FnMut(usize) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err(anyhow!("generate_stream: prompt_tokens cannot be empty"));
        }
        if prompt_tokens.len() + max_new_tokens > self.seq_len {
            return Err(anyhow!(
                "generate_stream: prompt ({}) + max_new_tokens ({}) exceeds max seq_len ({})",
                prompt_tokens.len(),
                max_new_tokens,
                self.seq_len
            ));
        }

        let t_start = std::time::Instant::now();

        // 1. Prefill prompt tokens into the KV cache
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            self.decode_step(tok, pos)?;
        }
        let prefill_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let mut output_tokens = prompt_tokens.to_vec();
        let is_greedy_unpenalized = (cfg.temperature <= 1e-6 || cfg.top_k == 1)
            && (cfg.repetition_penalty - 1.0).abs() < 1e-5;

        let t_decode_start = std::time::Instant::now();
        let mut stop_reason = crate::engine::generate::StopReason::MaxTokens;
        let mut generated_count = 0;

        // 2. Autoregressive decoding loop
        for step in 0..max_new_tokens {
            let next_tok = if is_greedy_unpenalized {
                // Zero-overhead GPU argmax + 4-byte async D2H read
                self.decode_logits.argmax_device(&self.d_argmax_out)?;
                self.d_argmax_out.async_download_to_pinned(&self.pinned_argmax, 0, 4)?;
                crate::cuda_sync()?;
                self.pinned_argmax.read_u32() as usize
            } else {
                // Download 1 row of logits to pinned host buffer and sample with context
                self.decode_logits.async_download_slice_to_pinned(&self.pinned_logits, 0, self.vocab_size)?;
                crate::cuda_sync()?;
                let logits_slice = self.pinned_logits.as_f32_slice(self.vocab_size);
                crate::engine::generate::sample_next_token_slice_with_context(
                    logits_slice,
                    &output_tokens,
                    cfg,
                )
            };

            output_tokens.push(next_tok);
            generated_count += 1;

            if !on_token(next_tok) {
                stop_reason = crate::engine::generate::StopReason::Cancelled;
                break;
            }

            if generated_count >= cfg.min_tokens && cfg.stop_tokens.contains(&next_tok) {
                stop_reason = crate::engine::generate::StopReason::StopToken(next_tok);
                break;
            }

            let next_pos = prompt_tokens.len() + step;
            if step + 1 < max_new_tokens && next_pos < self.seq_len {
                self.decode_step(next_tok, next_pos)?;
            }
        }

        let decode_ms = t_decode_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let tok_s = if decode_ms > 0.0 {
            (generated_count as f64 / decode_ms) * 1000.0
        } else {
            0.0
        };
        let ms_per_tok = if generated_count > 0 {
            decode_ms / generated_count as f64
        } else {
            0.0
        };

        Ok(crate::engine::generate::GenerationReport {
            tokens: output_tokens,
            prompt_len: prompt_tokens.len(),
            new_tokens: generated_count,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            total_time_ms: total_ms,
            tokens_per_sec: tok_s,
            ms_per_token: ms_per_tok,
            stop_reason,
        })
    }

    /// Autoregressive text generation.
    ///
    /// Automatically routes through the accelerated KV-cache engine (`generate_fast`) when the total
    /// sequence fits within the context window, falling back to sliding-window context evaluation
    /// if generating beyond `seq_len`.
    pub fn generate(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &crate::engine::SamplerConfig,
    ) -> Result<Vec<usize>> {
        if prompt_tokens.is_empty() {
            return Err(anyhow!("generate: prompt_tokens cannot be empty"));
        }
        if prompt_tokens.len() + max_new_tokens <= self.seq_len {
            return self.generate_fast(prompt_tokens, max_new_tokens, cfg);
        }

        // Sliding context fallback for extreme sequence lengths beyond static seq_len
        let mut current_tokens = prompt_tokens.to_vec();
        for _ in 0..max_new_tokens {
            let mut window = vec![0usize; self.seq_len];
            let n = current_tokens.len();
            if n <= self.seq_len {
                window[self.seq_len - n..].copy_from_slice(&current_tokens);
            } else {
                window.copy_from_slice(&current_tokens[n - self.seq_len..]);
            }

            let last_logits = self.forward_last_logits(&window)?;
            let next_tok = crate::engine::generate::sample_next_token_slice_with_context(
                &last_logits,
                &current_tokens,
                cfg,
            );
            current_tokens.push(next_tok);

            if current_tokens.len() - prompt_tokens.len() >= cfg.min_tokens
                && cfg.stop_tokens.contains(&next_tok)
            {
                break;
            }
        }
        Ok(current_tokens)
    }


    /// Forward kernel dispatch pass across embeddings, all transformer blocks, final layernorm,
    /// and lm_head projection. Resulting logits reside on device in `self.logits`.
    fn forward_inner(&mut self) -> Result<()> {
        let scale = (self.d_model as f32 / self.n_heads as f32).powf(-0.5);

        // Embeddings
        DeviceTensor::embedding_lookup_device_indices(
            &self.d_input_indices,
            self.seq_len,
            &self.tok_emb,
            &mut self.tok_out,
        )?;
        DeviceTensor::embedding_lookup_device_indices(
            &self.d_pos_indices,
            self.seq_len,
            &self.pos_emb,
            &mut self.pos_out,
        )?;
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
            la.attn_out
                .linear_into(&lw.wo, Some(&lw.bo), &mut la.proj_out)?;
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
            la.ln2_out
                .linear_into(&lw.w1, Some(&lw.b1), &mut la.mlp_h)?;
            la.mlp_h.gelu_into(&mut la.mlp_a)?;
            la.mlp_a
                .linear_into(&lw.w2, Some(&lw.b2), &mut la.mlp_out)?;

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
        self.ln_f_out
            .linear_into(&self.lm_head, None, &mut self.logits)?;

        Ok(())
    }

    fn step_inner(&mut self) -> Result<f32> {
        // ---------------------------------------------------------------
        // FORWARD PASS
        // ---------------------------------------------------------------
        self.forward_inner()?;

        // Cross-entropy loss (in-place zero allocation)
        DeviceTensor::cross_entropy_device_targets(
            &self.logits,
            &self.d_target_indices,
            &mut self.probs,
            &mut self.loss_tensor,
        )?;
        self.loss_tensor
            .async_download_scalar_f32(&self.pinned_loss)?;

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
        // GRADIENT NORM CLIPPING (optional)
        // Downloads gradient tensors to host, computes global L2 norm, then
        // scales every gradient DeviceTensor by clip_scale if needed.
        // This is a one-time CPU sync per step and only fires when enabled.
        // ---------------------------------------------------------------
        if let Some(max_norm) = self.max_grad_norm {
            // Collect all gradient DeviceTensor references into a scratch list
            // so we can compute the global squared norm.
            let mut sq_sum = 0.0f32;
            macro_rules! acc_norm {
                ($g:expr) => {
                    let host = $g.download()?;
                    sq_sum += host.iter().map(|v| v * v).sum::<f32>();
                };
            }
            acc_norm!(self.d_tok_emb);
            acc_norm!(self.d_pos_emb);
            acc_norm!(self.d_ln_f_gamma);
            acc_norm!(self.d_ln_f_beta);
            acc_norm!(self.d_lm_head);
            for lg in &self.layer_grads {
                acc_norm!(lg.d_ln1_gamma);
                acc_norm!(lg.d_ln1_beta);
                acc_norm!(lg.d_wq);
                acc_norm!(lg.d_bq);
                acc_norm!(lg.d_wk);
                acc_norm!(lg.d_bk);
                acc_norm!(lg.d_wv);
                acc_norm!(lg.d_bv);
                acc_norm!(lg.d_wo);
                acc_norm!(lg.d_bo);
                acc_norm!(lg.d_ln2_gamma);
                acc_norm!(lg.d_ln2_beta);
                acc_norm!(lg.d_w1);
                acc_norm!(lg.d_b1);
                acc_norm!(lg.d_w2);
                acc_norm!(lg.d_b2);
            }
            let global_norm = sq_sum.sqrt();
            if global_norm > max_norm {
                let clip_scale = max_norm / (global_norm + 1e-8);
                macro_rules! clip_grad {
                    ($g:expr) => {{
                        let scaled = $g.scale(clip_scale)?;
                        $g.copy_from(&scaled)?;
                    }};
                }
                clip_grad!(self.d_tok_emb);
                clip_grad!(self.d_pos_emb);
                clip_grad!(self.d_ln_f_gamma);
                clip_grad!(self.d_ln_f_beta);
                clip_grad!(self.d_lm_head);
                for lg in &mut self.layer_grads {
                    clip_grad!(lg.d_ln1_gamma);
                    clip_grad!(lg.d_ln1_beta);
                    clip_grad!(lg.d_wq);
                    clip_grad!(lg.d_bq);
                    clip_grad!(lg.d_wk);
                    clip_grad!(lg.d_bk);
                    clip_grad!(lg.d_wv);
                    clip_grad!(lg.d_bv);
                    clip_grad!(lg.d_wo);
                    clip_grad!(lg.d_bo);
                    clip_grad!(lg.d_ln2_gamma);
                    clip_grad!(lg.d_ln2_beta);
                    clip_grad!(lg.d_w1);
                    clip_grad!(lg.d_b1);
                    clip_grad!(lg.d_w2);
                    clip_grad!(lg.d_b2);
                }
            }
        }

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
                DeviceTensor::adamw_update(
                    &$p, &$mom.m, &$mom.v, &$g, lr, b1, b2, eps, wd, bc1, bc2,
                )?;
            };
        }

        update!(self.tok_emb, self.d_tok_emb, self.m_tok);
        update!(self.pos_emb, self.d_pos_emb, self.m_pos);
        update!(self.ln_f_gamma, self.d_ln_f_gamma, self.m_ln_f_gamma);
        update!(self.ln_f_beta, self.d_ln_f_beta, self.m_ln_f_beta);
        update!(self.lm_head, self.d_lm_head, self.m_lm_head);

        for l in 0..self.n_layers {
            let lw = &mut self.layers[l];
            let lg = &self.layer_grads[l];
            let lm = &mut self.layer_moments[l];
            update!(lw.ln1_gamma, lg.d_ln1_gamma, lm.ln1_gamma);
            update!(lw.ln1_beta, lg.d_ln1_beta, lm.ln1_beta);
            update!(lw.wq, lg.d_wq, lm.wq);
            update!(lw.bq, lg.d_bq, lm.bq);
            update!(lw.wk, lg.d_wk, lm.wk);
            update!(lw.bk, lg.d_bk, lm.bk);
            update!(lw.wv, lg.d_wv, lm.wv);
            update!(lw.bv, lg.d_bv, lm.bv);
            update!(lw.wo, lg.d_wo, lm.wo);
            update!(lw.bo, lg.d_bo, lm.bo);
            update!(lw.ln2_gamma, lg.d_ln2_gamma, lm.ln2_gamma);
            update!(lw.ln2_beta, lg.d_ln2_beta, lm.ln2_beta);
            update!(lw.w1, lg.d_w1, lm.w1);
            update!(lw.b1, lg.d_b1, lm.b1);
            update!(lw.w2, lg.d_w2, lm.w2);
            update!(lw.b2, lg.d_b2, lm.b2);
        }

        Ok(self.pinned_loss.read_f32())
    }

    // -----------------------------------------------------------------------
    // HYPER-PARAMETER SETTERS  (safe to call between steps)
    // -----------------------------------------------------------------------

    /// Update the AdamW learning rate at runtime (e.g. from a LR scheduler).
    #[inline]
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Enable or disable global gradient norm clipping.
    /// Pass `Some(max_norm)` to clip, `None` to disable.
    #[inline]
    pub fn set_max_grad_norm(&mut self, max_norm: Option<f32>) {
        self.max_grad_norm = max_norm;
    }

    /// Return the current training step count.
    #[inline]
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Return the current AdamW learning rate.
    #[inline]
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Return the current maximum gradient clipping norm, if configured.
    #[inline]
    pub fn max_grad_norm(&self) -> Option<f32> {
        self.max_grad_norm
    }

    /// Vocabulary size configured for this model.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embedding dimension (d_model).
    #[inline]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Number of attention heads.
    #[inline]
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Number of transformer layers.
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Feed-forward hidden dimension (d_ff).
    #[inline]
    pub fn d_ff(&self) -> usize {
        self.d_ff
    }

    /// Fixed static sequence length context horizon.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset AdamW moment accumulators (m and v) to zero across all weights.
    pub fn reset_moments(&mut self) -> Result<()> {
        macro_rules! zero_mom {
            ($m:expr) => {{
                $m.m.fill(0.0)?;
                $m.v.fill(0.0)?;
            }};
        }
        zero_mom!(self.m_tok);
        zero_mom!(self.m_pos);
        zero_mom!(self.m_ln_f_gamma);
        zero_mom!(self.m_ln_f_beta);
        zero_mom!(self.m_lm_head);
        for lm in &mut self.layer_moments {
            zero_mom!(lm.ln1_gamma);
            zero_mom!(lm.ln1_beta);
            zero_mom!(lm.wq);
            zero_mom!(lm.bq);
            zero_mom!(lm.wk);
            zero_mom!(lm.bk);
            zero_mom!(lm.wv);
            zero_mom!(lm.bv);
            zero_mom!(lm.wo);
            zero_mom!(lm.bo);
            zero_mom!(lm.ln2_gamma);
            zero_mom!(lm.ln2_beta);
            zero_mom!(lm.w1);
            zero_mom!(lm.b1);
            zero_mom!(lm.w2);
            zero_mom!(lm.b2);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // WEIGHT GETTERS & SETTERS (SafeTensors & Interoperability)
    // -----------------------------------------------------------------------
    pub fn get_tok_emb(&self) -> Result<Vec<f32>> { self.tok_emb.download() }
    pub fn get_pos_emb(&self) -> Result<Vec<f32>> { self.pos_emb.download() }
    pub fn get_ln_f_gamma(&self) -> Result<Vec<f32>> { self.ln_f_gamma.download() }
    pub fn get_ln_f_beta(&self) -> Result<Vec<f32>> { self.ln_f_beta.download() }
    pub fn get_lm_head(&self) -> Result<Vec<f32>> { self.lm_head.download() }

    pub fn get_layer_ln1_gamma(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.ln1_gamma.download())
    }
    pub fn get_layer_ln1_beta(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.ln1_beta.download())
    }
    pub fn get_layer_wq(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.wq.download())
    }
    pub fn get_layer_wk(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.wk.download())
    }
    pub fn get_layer_wv(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.wv.download())
    }
    pub fn get_layer_wo(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.wo.download())
    }
    pub fn get_layer_bq(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.bq.download())
    }
    pub fn get_layer_bk(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.bk.download())
    }
    pub fn get_layer_bv(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.bv.download())
    }
    pub fn get_layer_bo(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.bo.download())
    }
    pub fn get_layer_ln2_gamma(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.ln2_gamma.download())
    }
    pub fn get_layer_ln2_beta(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.ln2_beta.download())
    }
    pub fn get_layer_w1(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.w1.download())
    }
    pub fn get_layer_b1(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.b1.download())
    }
    pub fn get_layer_w2(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.w2.download())
    }
    pub fn get_layer_b2(&self, l: usize) -> Result<Vec<f32>> {
        self.layers.get(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds")).and_then(|lw| lw.b2.download())
    }

    pub fn set_tok_emb(&mut self, data: &[f32]) -> Result<()> { self.tok_emb.upload(data) }
    pub fn set_pos_emb(&mut self, data: &[f32]) -> Result<()> { self.pos_emb.upload(data) }
    pub fn set_ln_f_gamma(&mut self, data: &[f32]) -> Result<()> { self.ln_f_gamma.upload(data) }
    pub fn set_ln_f_beta(&mut self, data: &[f32]) -> Result<()> { self.ln_f_beta.upload(data) }
    pub fn set_lm_head(&mut self, data: &[f32]) -> Result<()> { self.lm_head.upload(data) }

    pub fn set_layer_ln1_gamma(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.ln1_gamma.upload(data)
    }
    pub fn set_layer_ln1_beta(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.ln1_beta.upload(data)
    }
    pub fn set_layer_wq(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.wq.upload(data)
    }
    pub fn set_layer_wk(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.wk.upload(data)
    }
    pub fn set_layer_wv(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.wv.upload(data)
    }
    pub fn set_layer_wo(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.wo.upload(data)
    }
    pub fn set_layer_bq(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.bq.upload(data)
    }
    pub fn set_layer_bk(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.bk.upload(data)
    }
    pub fn set_layer_bv(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.bv.upload(data)
    }
    pub fn set_layer_bo(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.bo.upload(data)
    }
    pub fn set_layer_ln2_gamma(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.ln2_gamma.upload(data)
    }
    pub fn set_layer_ln2_beta(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.ln2_beta.upload(data)
    }
    pub fn set_layer_w1(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.w1.upload(data)
    }
    pub fn set_layer_b1(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.b1.upload(data)
    }
    pub fn set_layer_w2(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.w2.upload(data)
    }
    pub fn set_layer_b2(&mut self, l: usize, data: &[f32]) -> Result<()> {
        self.layers.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?.b2.upload(data)
    }

    // -----------------------------------------------------------------------
    // CHECKPOINT SAVE / LOAD
    // -----------------------------------------------------------------------
    /// Serialise all weight tensors and AdamW moment buffers to `path`.
    ///
    /// Format: a flat sequence of (u64 length, f32 data…) blocks, one per tensor,
    /// followed by a footer containing step_count, lr, beta1, beta2, eps, wd.
    /// This is a simple custom binary format — use `load_checkpoint` to restore.
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        use std::io::{BufWriter, Write};
        let file = std::fs::File::create(path)
            .map_err(|e| anyhow!("save_checkpoint: cannot create {:?}: {e}", path))?;
        let mut w = BufWriter::new(file);

        macro_rules! write_tensor {
            ($t:expr) => {{
                let data = $t.download()?;
                let n = data.len() as u64;
                w.write_all(&n.to_le_bytes())?;
                for v in &data {
                    w.write_all(&v.to_le_bytes())?;
                }
            }};
        }
        macro_rules! write_moments {
            ($m:expr) => {{
                write_tensor!($m.m);
                write_tensor!($m.v);
            }};
        }

        // Global weights
        write_tensor!(self.tok_emb);
        write_tensor!(self.pos_emb);
        write_tensor!(self.ln_f_gamma);
        write_tensor!(self.ln_f_beta);
        write_tensor!(self.lm_head);
        // Per-layer weights
        for lw in &self.layers {
            write_tensor!(lw.ln1_gamma);
            write_tensor!(lw.ln1_beta);
            write_tensor!(lw.wq);
            write_tensor!(lw.bq);
            write_tensor!(lw.wk);
            write_tensor!(lw.bk);
            write_tensor!(lw.wv);
            write_tensor!(lw.bv);
            write_tensor!(lw.wo);
            write_tensor!(lw.bo);
            write_tensor!(lw.ln2_gamma);
            write_tensor!(lw.ln2_beta);
            write_tensor!(lw.w1);
            write_tensor!(lw.b1);
            write_tensor!(lw.w2);
            write_tensor!(lw.b2);
        }
        // AdamW moments (global)
        write_moments!(self.m_tok);
        write_moments!(self.m_pos);
        write_moments!(self.m_ln_f_gamma);
        write_moments!(self.m_ln_f_beta);
        write_moments!(self.m_lm_head);
        // AdamW moments (per-layer)
        for lm in &self.layer_moments {
            write_moments!(lm.ln1_gamma);
            write_moments!(lm.ln1_beta);
            write_moments!(lm.wq);
            write_moments!(lm.bq);
            write_moments!(lm.wk);
            write_moments!(lm.bk);
            write_moments!(lm.wv);
            write_moments!(lm.bv);
            write_moments!(lm.wo);
            write_moments!(lm.bo);
            write_moments!(lm.ln2_gamma);
            write_moments!(lm.ln2_beta);
            write_moments!(lm.w1);
            write_moments!(lm.b1);
            write_moments!(lm.w2);
            write_moments!(lm.b2);
        }
        // Footer: step_count (u64) + lr, beta1, beta2, eps, wd (f32 each)
        w.write_all(&(self.step_count as u64).to_le_bytes())?;
        w.write_all(&self.lr.to_le_bytes())?;
        w.write_all(&self.beta1.to_le_bytes())?;
        w.write_all(&self.beta2.to_le_bytes())?;
        w.write_all(&self.eps.to_le_bytes())?;
        w.write_all(&self.wd.to_le_bytes())?;
        w.flush()?;
        Ok(())
    }

    /// Restore weights and AdamW state from a checkpoint written by `save_checkpoint`.
    ///
    /// The model must have been constructed with the *same* hyper-parameters
    /// (vocab_size, d_model, n_heads, n_layers, d_ff, seq_len).
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
        use std::io::{BufReader, Read};
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow!("load_checkpoint: cannot open {:?}: {e}", path))?;
        let mut r = BufReader::new(file);

        let mut read_vec = |r: &mut BufReader<std::fs::File>| -> Result<Vec<f32>> {
            let mut len_bytes = [0u8; 8];
            r.read_exact(&mut len_bytes)?;
            let n = u64::from_le_bytes(len_bytes) as usize;
            let mut data = vec![0.0f32; n];
            for v in &mut data {
                let mut buf = [0u8; 4];
                r.read_exact(&mut buf)?;
                *v = f32::from_le_bytes(buf);
            }
            Ok(data)
        };

        macro_rules! load_tensor {
            ($t:expr) => {{
                let data = read_vec(&mut r)?;
                if data.len() != $t.numel() {
                    return Err(anyhow!(
                        "load_checkpoint: tensor size mismatch: file={} model={}",
                        data.len(),
                        $t.numel()
                    ));
                }
                $t.upload(&data)?;
            }};
        }
        macro_rules! load_moments {
            ($m:expr) => {{
                load_tensor!($m.m);
                load_tensor!($m.v);
            }};
        }

        // Global weights
        load_tensor!(self.tok_emb);
        load_tensor!(self.pos_emb);
        load_tensor!(self.ln_f_gamma);
        load_tensor!(self.ln_f_beta);
        load_tensor!(self.lm_head);
        // Per-layer weights
        for lw in &mut self.layers {
            load_tensor!(lw.ln1_gamma);
            load_tensor!(lw.ln1_beta);
            load_tensor!(lw.wq);
            load_tensor!(lw.bq);
            load_tensor!(lw.wk);
            load_tensor!(lw.bk);
            load_tensor!(lw.wv);
            load_tensor!(lw.bv);
            load_tensor!(lw.wo);
            load_tensor!(lw.bo);
            load_tensor!(lw.ln2_gamma);
            load_tensor!(lw.ln2_beta);
            load_tensor!(lw.w1);
            load_tensor!(lw.b1);
            load_tensor!(lw.w2);
            load_tensor!(lw.b2);
        }
        // AdamW moments (global)
        load_moments!(self.m_tok);
        load_moments!(self.m_pos);
        load_moments!(self.m_ln_f_gamma);
        load_moments!(self.m_ln_f_beta);
        load_moments!(self.m_lm_head);
        // AdamW moments (per-layer)
        for lm in &mut self.layer_moments {
            load_moments!(lm.ln1_gamma);
            load_moments!(lm.ln1_beta);
            load_moments!(lm.wq);
            load_moments!(lm.bq);
            load_moments!(lm.wk);
            load_moments!(lm.bk);
            load_moments!(lm.wv);
            load_moments!(lm.bv);
            load_moments!(lm.wo);
            load_moments!(lm.bo);
            load_moments!(lm.ln2_gamma);
            load_moments!(lm.ln2_beta);
            load_moments!(lm.w1);
            load_moments!(lm.b1);
            load_moments!(lm.w2);
            load_moments!(lm.b2);
        }
        // Footer
        let mut u64_buf = [0u8; 8];
        r.read_exact(&mut u64_buf)?;
        self.step_count = u64::from_le_bytes(u64_buf) as usize;
        let mut f32_buf = [0u8; 4];
        r.read_exact(&mut f32_buf)?; self.lr = f32::from_le_bytes(f32_buf);
        r.read_exact(&mut f32_buf)?; self.beta1 = f32::from_le_bytes(f32_buf);
        r.read_exact(&mut f32_buf)?; self.beta2 = f32::from_le_bytes(f32_buf);
        r.read_exact(&mut f32_buf)?; self.eps = f32::from_le_bytes(f32_buf);
        r.read_exact(&mut f32_buf)?; self.wd = f32::from_le_bytes(f32_buf);
        // Invalidate any captured CUDA graph — the weights have changed
        self.graph_exec = None;
        Ok(())
    }

    /// Diagnostic: run one step with `cudaDeviceSynchronize` after EVERY kernel call
    /// so we can isolate which exact operation takes the most wall-clock time.
    /// Do NOT call this during training — it is purely for profiling.
    pub fn profile_step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<()> {
        use darkforest_cuda::DeviceTensor;
        use std::time::Instant;

        assert_eq!(input_tokens.len(), self.seq_len);
        assert_eq!(target_tokens.len(), self.seq_len);
        self.step_count += 1;

        for (i, &tok) in input_tokens.iter().enumerate() {
            self.h_input_u32[i] = tok as u32;
        }
        for (i, &tok) in target_tokens.iter().enumerate() {
            self.h_target_u32[i] = tok as u32;
        }

        let t = Instant::now();
        self.d_input_indices
            .upload_bytes(self.h_input_u32.as_ptr() as *const u8, self.seq_len * 4)?;
        self.d_target_indices
            .upload_bytes(self.h_target_u32.as_ptr() as *const u8, self.seq_len * 4)?;
        crate::cuda_sync()?;
        println!(
            "  {:50} {:7.3} ms",
            "H2D upload indices",
            t.elapsed().as_secs_f64() * 1000.0
        );

        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                let t = Instant::now();
                $body;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    $label,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }};
        }

        let scale = (self.d_model as f32 / self.n_heads as f32).powf(-0.5);

        prof!("embedding lookup tok", {
            DeviceTensor::embedding_lookup_device_indices(
                &self.d_input_indices,
                self.seq_len,
                &self.tok_emb,
                &mut self.tok_out,
            )?;
        });
        prof!("embedding lookup pos", {
            DeviceTensor::embedding_lookup_device_indices(
                &self.d_pos_indices,
                self.seq_len,
                &self.pos_emb,
                &mut self.pos_out,
            )?;
        });
        prof!("tok + pos add", {
            self.tok_out.add_into(&self.pos_out, &mut self.x_in[0])?;
        });

        for l in 0..self.n_layers {
            let lw = &self.layers[l];
            let la = &mut self.layer_acts[l];
            let label_ln1 = format!("L{l} layernorm1");
            let label_q = format!("L{l} Q proj linear");
            let label_k = format!("L{l} K proj linear");
            let label_v = format!("L{l} V proj linear");
            let label_attn = format!("L{l} attention fwd");
            let label_op = format!("L{l} O proj + residual");
            let label_ln2 = format!("L{l} layernorm2");
            let label_mu = format!("L{l} MLP up linear");
            let label_mg = format!("L{l} MLP gelu");
            let label_md = format!("L{l} MLP down + residual");

            {
                let t = Instant::now();
                self.x_in[l].layernorm_into(
                    Some(&lw.ln1_gamma),
                    Some(&lw.ln1_beta),
                    &mut la.ln1_out,
                    &mut la.ln1_mean,
                    &mut la.ln1_rstd,
                )?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_ln1,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.ln1_out.linear_into(&lw.wq, Some(&lw.bq), &mut la.q)?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_q,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.ln1_out.linear_into(&lw.wk, Some(&lw.bk), &mut la.k)?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_k,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.ln1_out.linear_into(&lw.wv, Some(&lw.bv), &mut la.v)?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_v,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                self.attn_contexts[l].forward_device_into(
                    &la.q,
                    &la.k,
                    &la.v,
                    scale,
                    true,
                    &mut la.attn_out,
                )?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_attn,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.attn_out
                    .linear_into(&lw.wo, Some(&lw.bo), &mut la.proj_out)?;
                self.x_in[l].add_into(&la.proj_out, &mut la.x_mid)?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_op,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.x_mid.layernorm_into(
                    Some(&lw.ln2_gamma),
                    Some(&lw.ln2_beta),
                    &mut la.ln2_out,
                    &mut la.ln2_mean,
                    &mut la.ln2_rstd,
                )?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_ln2,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.ln2_out
                    .linear_into(&lw.w1, Some(&lw.b1), &mut la.mlp_h)?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_mu,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.mlp_h.gelu_into(&mut la.mlp_a)?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_mg,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            {
                let t = Instant::now();
                la.mlp_a
                    .linear_into(&lw.w2, Some(&lw.b2), &mut la.mlp_out)?;
                la.x_mid.add_into(&la.mlp_out, &mut self.x_in[l + 1])?;
                crate::cuda_sync()?;
                println!(
                    "  {:50} {:7.3} ms",
                    label_md,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        prof!("final layernorm", {
            self.x_in[self.n_layers].layernorm_into(
                Some(&self.ln_f_gamma),
                Some(&self.ln_f_beta),
                &mut self.ln_f_out,
                &mut self.ln_f_mean,
                &mut self.ln_f_rstd,
            )?;
        });
        prof!("lm head linear", {
            self.ln_f_out
                .linear_into(&self.lm_head, None, &mut self.logits)?;
        });
        prof!("cross-entropy forward", {
            DeviceTensor::cross_entropy_device_targets(
                &self.logits,
                &self.d_target_indices,
                &mut self.probs,
                &mut self.loss_tensor,
            )?;
            self.loss_tensor
                .async_download_scalar_f32(&self.pinned_loss)?;
        });

        // --- BACKWARD ---
        prof!("cross-entropy backward", {
            DeviceTensor::cross_entropy_backward_device_targets(
                &self.probs,
                &self.d_target_indices,
                &self.grad_one,
                &mut self.d_logits,
            )?;
        });
        prof!("lm head backward", {
            DeviceTensor::linear_backward_into(
                &self.ln_f_out,
                &self.lm_head,
                &self.d_logits,
                &mut self.d_ln_f_out,
                &mut self.d_lm_head,
                None,
            )?;
        });
        prof!("final layernorm backward", {
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
        });

        for l in (0..self.n_layers).rev() {
            let label = format!("L{l} block backward (all)");
            let t = Instant::now();
            let lw = &self.layers[l];
            let la = &self.layer_acts[l];
            let lg = &mut self.layer_grads[l];

            lg.d_mlp_out.copy_from(&self.dx_comb)?;
            lg.dx.copy_from(&self.dx_comb)?;
            let d_x_mid_from_above = &mut lg.dx;

            DeviceTensor::linear_backward_into(
                &la.mlp_a,
                &lw.w2,
                &lg.d_mlp_out,
                &mut lg.d_mlp_a,
                &mut lg.d_w2,
                Some(&mut lg.d_b2),
            )?;
            la.mlp_h.gelu_backward_into(&lg.d_mlp_a, &mut lg.d_mlp_h)?;
            DeviceTensor::linear_backward_into(
                &la.ln2_out,
                &lw.w1,
                &lg.d_mlp_h,
                &mut lg.d_ln2_out,
                &mut lg.d_w1,
                Some(&mut lg.d_b1),
            )?;
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
            d_x_mid_from_above.add_inplace(&self.dx_comb)?;
            lg.d_proj_out.copy_from(d_x_mid_from_above)?;
            DeviceTensor::linear_backward_into(
                &la.attn_out,
                &lw.wo,
                &lg.d_proj_out,
                &mut lg.d_attn_out,
                &mut lg.d_wo,
                Some(&mut lg.d_bo),
            )?;
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
            DeviceTensor::linear_backward_into(
                &la.ln1_out,
                &lw.wq,
                &lg.d_q,
                &mut lg.d_ln1_out,
                &mut lg.d_wq,
                Some(&mut lg.d_bq),
            )?;
            DeviceTensor::linear_backward_accumulate_into(
                &la.ln1_out,
                &lw.wk,
                &lg.d_k,
                &mut lg.d_ln1_out,
                &mut lg.d_wk,
                Some(&mut lg.d_bk),
            )?;
            DeviceTensor::linear_backward_accumulate_into(
                &la.ln1_out,
                &lw.wv,
                &lg.d_v,
                &mut lg.d_ln1_out,
                &mut lg.d_wv,
                Some(&mut lg.d_bv),
            )?;
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
            self.dx_comb.add_inplace(&lg.dx)?;

            crate::cuda_sync()?;
            println!(
                "  {:50} {:7.3} ms",
                label,
                t.elapsed().as_secs_f64() * 1000.0
            );
        }

        prof!("embedding backward", {
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
        });

        // AdamW (just time one full update pass)
        let t_adamw = Instant::now();
        let t_val = self.step_count as f32;
        let lr = self.lr;
        let (b1, b2, eps, wd) = (self.beta1, self.beta2, self.eps, self.wd);
        let bc1 = 1.0 - b1.powf(t_val);
        let bc2 = 1.0 - b2.powf(t_val);
        macro_rules! update {
            ($p:expr, $g:expr, $mom:expr) => {
                DeviceTensor::adamw_update(
                    &$p, &$mom.m, &$mom.v, &$g, lr, b1, b2, eps, wd, bc1, bc2,
                )?;
            };
        }
        update!(self.tok_emb, self.d_tok_emb, self.m_tok);
        update!(self.pos_emb, self.d_pos_emb, self.m_pos);
        update!(self.ln_f_gamma, self.d_ln_f_gamma, self.m_ln_f_gamma);
        update!(self.ln_f_beta, self.d_ln_f_beta, self.m_ln_f_beta);
        update!(self.lm_head, self.d_lm_head, self.m_lm_head);
        for l in 0..self.n_layers {
            let lw = &mut self.layers[l];
            let lg = &self.layer_grads[l];
            let lm = &mut self.layer_moments[l];
            update!(lw.ln1_gamma, lg.d_ln1_gamma, lm.ln1_gamma);
            update!(lw.ln1_beta, lg.d_ln1_beta, lm.ln1_beta);
            update!(lw.wq, lg.d_wq, lm.wq);
            update!(lw.bq, lg.d_bq, lm.bq);
            update!(lw.wk, lg.d_wk, lm.wk);
            update!(lw.bk, lg.d_bk, lm.bk);
            update!(lw.wv, lg.d_wv, lm.wv);
            update!(lw.bv, lg.d_bv, lm.bv);
            update!(lw.wo, lg.d_wo, lm.wo);
            update!(lw.bo, lg.d_bo, lm.bo);
            update!(lw.ln2_gamma, lg.d_ln2_gamma, lm.ln2_gamma);
            update!(lw.ln2_beta, lg.d_ln2_beta, lm.ln2_beta);
            update!(lw.w1, lg.d_w1, lm.w1);
            update!(lw.b1, lg.d_b1, lm.b1);
            update!(lw.w2, lg.d_w2, lm.w2);
            update!(lw.b2, lg.d_b2, lm.b2);
        }
        crate::cuda_sync()?;
        println!(
            "  {:50} {:7.3} ms",
            "AdamW update (all params)",
            t_adamw.elapsed().as_secs_f64() * 1000.0
        );

        Ok(())
    }
}

/// CPU-backed transformer execution engine when CUDA feature is disabled.
#[cfg(not(feature = "cuda"))]
pub struct StaticGPT2 {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    seq_len: usize,
    lr: f32,
    max_grad_norm: Option<f32>,
    step_count: usize,
    tok_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    ln_f_gamma: Vec<f32>,
    ln_f_beta: Vec<f32>,
    lm_head: Vec<f32>,
    pub(crate) layer_wq: Vec<Vec<f32>>,
    layer_bq: Vec<Vec<f32>>,
    pub(crate) layer_wk: Vec<Vec<f32>>,
    layer_bk: Vec<Vec<f32>>,
    pub(crate) layer_wv: Vec<Vec<f32>>,
    layer_bv: Vec<Vec<f32>>,
    pub(crate) layer_wo: Vec<Vec<f32>>,
    layer_bo: Vec<Vec<f32>>,
    layer_ln1_g: Vec<Vec<f32>>,
    layer_ln1_b: Vec<Vec<f32>>,
    layer_ln2_g: Vec<Vec<f32>>,
    layer_ln2_b: Vec<Vec<f32>>,
    layer_w1: Vec<Vec<f32>>,
    layer_b1: Vec<Vec<f32>>,
    layer_w2: Vec<Vec<f32>>,
    layer_b2: Vec<Vec<f32>>,
}

#[cfg(not(feature = "cuda"))]
impl StaticGPT2 {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        seq_len: usize,
        lr: f32,
    ) -> Result<Self> {
        let std_dev = 0.02f32;
        let mut rng = rand::thread_rng();
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0f32, std_dev).unwrap();

        let mut sample_vec = |len: usize| -> Vec<f32> {
            (0..len).map(|_| normal.sample(&mut rng)).collect()
        };

        let tok_emb = sample_vec(vocab_size * d_model);
        let pos_emb = sample_vec(seq_len * d_model);
        let ln_f_gamma = vec![1.0f32; d_model];
        let ln_f_beta = vec![0.0f32; d_model];
        let lm_head = sample_vec(vocab_size * d_model);

        let mut layer_wq = Vec::with_capacity(n_layers);
        let mut layer_bq = Vec::with_capacity(n_layers);
        let mut layer_wk = Vec::with_capacity(n_layers);
        let mut layer_bk = Vec::with_capacity(n_layers);
        let mut layer_wv = Vec::with_capacity(n_layers);
        let mut layer_bv = Vec::with_capacity(n_layers);
        let mut layer_wo = Vec::with_capacity(n_layers);
        let mut layer_bo = Vec::with_capacity(n_layers);
        let mut layer_ln1_g = Vec::with_capacity(n_layers);
        let mut layer_ln1_b = Vec::with_capacity(n_layers);
        let mut layer_ln2_g = Vec::with_capacity(n_layers);
        let mut layer_ln2_b = Vec::with_capacity(n_layers);
        let mut layer_w1 = Vec::with_capacity(n_layers);
        let mut layer_b1 = Vec::with_capacity(n_layers);
        let mut layer_w2 = Vec::with_capacity(n_layers);
        let mut layer_b2 = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            layer_wq.push(sample_vec(d_model * d_model));
            layer_bq.push(vec![0.0f32; d_model]);
            layer_wk.push(sample_vec(d_model * d_model));
            layer_bk.push(vec![0.0f32; d_model]);
            layer_wv.push(sample_vec(d_model * d_model));
            layer_bv.push(vec![0.0f32; d_model]);
            layer_wo.push(sample_vec(d_model * d_model));
            layer_bo.push(vec![0.0f32; d_model]);
            layer_ln1_g.push(vec![1.0f32; d_model]);
            layer_ln1_b.push(vec![0.0f32; d_model]);
            layer_ln2_g.push(vec![1.0f32; d_model]);
            layer_ln2_b.push(vec![0.0f32; d_model]);
            layer_w1.push(sample_vec(d_ff * d_model));
            layer_b1.push(vec![0.0f32; d_ff]);
            layer_w2.push(sample_vec(d_model * d_ff));
            layer_b2.push(vec![0.0f32; d_model]);
        }

        Ok(Self {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            seq_len,
            lr,
            max_grad_norm: None,
            step_count: 0,
            tok_emb,
            pos_emb,
            ln_f_gamma,
            ln_f_beta,
            lm_head,
            layer_wq,
            layer_bq,
            layer_wk,
            layer_bk,
            layer_wv,
            layer_bv,
            layer_wo,
            layer_bo,
            layer_ln1_g,
            layer_ln1_b,
            layer_ln2_g,
            layer_ln2_b,
            layer_w1,
            layer_b1,
            layer_w2,
            layer_b2,
        })
    }

    pub(crate) fn layer_wq_mut(&mut self, l: usize) -> &mut Vec<f32> {
        &mut self.layer_wq[l]
    }
    pub(crate) fn layer_wk_mut(&mut self, l: usize) -> &mut Vec<f32> {
        &mut self.layer_wk[l]
    }
    pub(crate) fn layer_wv_mut(&mut self, l: usize) -> &mut Vec<f32> {
        &mut self.layer_wv[l]
    }
    pub(crate) fn layer_wo_mut(&mut self, l: usize) -> &mut Vec<f32> {
        &mut self.layer_wo[l]
    }

    fn cpu_layernorm(x: &[f32], gamma: &[f32], beta: &[f32], d_model: usize) -> Vec<f32> {
        let n_tokens = x.len() / d_model;
        let mut out = vec![0.0f32; x.len()];
        for t in 0..n_tokens {
            let offset = t * d_model;
            let slice = &x[offset..offset + d_model];
            let mean: f32 = slice.iter().sum::<f32>() / d_model as f32;
            let var: f32 = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / d_model as f32;
            let rstd = 1.0f32 / (var + 1e-5).sqrt();
            for i in 0..d_model {
                out[offset + i] = (slice[i] - mean) * rstd * gamma[i] + beta[i];
            }
        }
        out
    }

    fn cpu_linear(x: &[f32], w: &[f32], b: Option<&[f32]>, in_f: usize, out_f: usize) -> Vec<f32> {
        let n_tokens = x.len() / in_f;
        let mut out = vec![0.0f32; n_tokens * out_f];
        for t in 0..n_tokens {
            let x_off = t * in_f;
            let out_off = t * out_f;
            for o in 0..out_f {
                let mut acc = b.map(|bias| bias[o]).unwrap_or(0.0f32);
                let w_off = o * in_f;
                for i in 0..in_f {
                    acc += x[x_off + i] * w[w_off + i];
                }
                out[out_off + o] = acc;
            }
        }
        out
    }

    pub fn forward(&mut self, input_tokens: &[usize]) -> Result<Vec<f32>> {
        let t = input_tokens.len();
        if t == 0 || t > self.seq_len {
            return Err(anyhow!("forward: token length {} exceeds seq_len {}", t, self.seq_len));
        }
        let dm = self.d_model;
        let nh = self.n_heads;
        let dh = dm / nh;
        let scale = 1.0f32 / (dh as f32).sqrt();

        // 1. Embeddings: x = tok_emb[tok] + pos_emb[pos]
        let mut x = vec![0.0f32; t * dm];
        for i in 0..t {
            let tok = input_tokens[i];
            if tok >= self.vocab_size {
                return Err(anyhow!("Token {} out of vocab bounds {}", tok, self.vocab_size));
            }
            let tok_slice = &self.tok_emb[tok * dm..(tok + 1) * dm];
            let pos_slice = &self.pos_emb[i * dm..(i + 1) * dm];
            for j in 0..dm {
                x[i * dm + j] = tok_slice[j] + pos_slice[j];
            }
        }

        // 2. Transformer layers
        for l in 0..self.n_layers {
            let ln1 = Self::cpu_layernorm(&x, &self.layer_ln1_g[l], &self.layer_ln1_b[l], dm);
            let q = Self::cpu_linear(&ln1, &self.layer_wq[l], Some(&self.layer_bq[l]), dm, dm);
            let k = Self::cpu_linear(&ln1, &self.layer_wk[l], Some(&self.layer_bk[l]), dm, dm);
            let v = Self::cpu_linear(&ln1, &self.layer_wv[l], Some(&self.layer_bv[l]), dm, dm);

            // Causal multi-head self-attention
            let mut attn_out = vec![0.0f32; t * dm];
            for h in 0..nh {
                for i in 0..t {
                    // compute attention scores with causal mask (j <= i)
                    let mut scores = vec![f32::NEG_INFINITY; t];
                    for j in 0..=i {
                        let mut dot = 0.0f32;
                        for d in 0..dh {
                            let q_val = q[i * dm + h * dh + d];
                            let k_val = k[j * dm + h * dh + d];
                            dot += q_val * k_val;
                        }
                        scores[j] = dot * scale;
                    }
                    let max_score = scores[0..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    let mut exp_scores = vec![0.0f32; t];
                    for j in 0..=i {
                        exp_scores[j] = (scores[j] - max_score).exp();
                        exp_sum += exp_scores[j];
                    }
                    let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                    for d in 0..dh {
                        let mut val = 0.0f32;
                        for j in 0..=i {
                            let weight = exp_scores[j] * inv_sum;
                            val += weight * v[j * dm + h * dh + d];
                        }
                        attn_out[i * dm + h * dh + d] = val;
                    }
                }
            }

            let proj = Self::cpu_linear(&attn_out, &self.layer_wo[l], Some(&self.layer_bo[l]), dm, dm);
            // Residual add: x = x + proj
            for idx in 0..x.len() {
                x[idx] += proj[idx];
            }

            // LayerNorm 2
            let ln2 = Self::cpu_layernorm(&x, &self.layer_ln2_g[l], &self.layer_ln2_b[l], dm);

            // MLP: w1 -> GELU -> w2
            let mut h1 = Self::cpu_linear(&ln2, &self.layer_w1[l], Some(&self.layer_b1[l]), dm, self.d_ff);
            for v_elem in &mut h1 {
                let x_val = *v_elem;
                // GELU approximation
                *v_elem = 0.5 * x_val * (1.0 + (0.79788456 * (x_val + 0.044715 * x_val.powi(3))).tanh());
            }
            let mlp_out = Self::cpu_linear(&h1, &self.layer_w2[l], Some(&self.layer_b2[l]), self.d_ff, dm);

            // Residual add: x = x + mlp_out
            for idx in 0..x.len() {
                x[idx] += mlp_out[idx];
            }
        }

        // 3. Final LayerNorm & LM Head projection
        let ln_f = Self::cpu_layernorm(&x, &self.ln_f_gamma, &self.ln_f_beta, dm);
        let logits = Self::cpu_linear(&ln_f, &self.lm_head, None, dm, self.vocab_size);
        Ok(logits)
    }

    pub fn forward_last_logits(&mut self, input_tokens: &[usize]) -> Result<Vec<f32>> {
        let logits = self.forward(input_tokens)?;
        let t = input_tokens.len();
        let vs = self.vocab_size;
        let last_slice = &logits[(t - 1) * vs..t * vs];
        Ok(last_slice.to_vec())
    }

    pub fn predict(&mut self, input_tokens: &[usize]) -> Result<usize> {
        let last_logits = self.forward_last_logits(input_tokens)?;
        Ok(crate::engine::generate::argmax(&last_logits))
    }

    pub fn eval_step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        if input_tokens.len() != target_tokens.len() {
            return Err(anyhow!("input and target length mismatch"));
        }
        let logits = self.forward(input_tokens)?;
        let t = input_tokens.len();
        let vs = self.vocab_size;
        let mut total_loss = 0.0f32;

        for i in 0..t {
            let tgt = target_tokens[i];
            let offset = i * vs;
            let slice = &logits[offset..offset + vs];
            let max_logit = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = max_logit + slice.iter().map(|&v| (v - max_logit).exp()).sum::<f32>().ln();
            let loss_i = log_sum_exp - slice[tgt];
            total_loss += loss_i;
        }

        Ok(total_loss / t as f32)
    }

    pub fn step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        self.step_count += 1;
        self.eval_step(input_tokens, target_tokens)
    }

    pub fn step_graph(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<f32> {
        self.step(input_tokens, target_tokens)
    }

    pub fn generate_fast(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &crate::engine::SamplerConfig,
    ) -> Result<Vec<usize>> {
        let report = self.generate_stream(prompt_tokens, max_new_tokens, cfg, |_| true)?;
        Ok(report.tokens)
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &crate::engine::SamplerConfig,
        mut on_token: F,
    ) -> Result<crate::engine::generate::GenerationReport>
    where
        F: FnMut(usize) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err(anyhow!("prompt_tokens cannot be empty"));
        }
        let t0 = std::time::Instant::now();
        let mut current_tokens = prompt_tokens.to_vec();
        let mut generated_count = 0;
        let mut stop_reason = crate::engine::generate::StopReason::MaxTokens;

        let t_decode = std::time::Instant::now();
        for _ in 0..max_new_tokens {
            let window_len = current_tokens.len().min(self.seq_len);
            let ctx = &current_tokens[current_tokens.len() - window_len..];
            let last_logits = self.forward_last_logits(ctx)?;
            let next_tok = crate::engine::generate::sample_next_token_slice_with_context(
                &last_logits,
                &current_tokens,
                cfg,
            );
            current_tokens.push(next_tok);
            generated_count += 1;

            if !on_token(next_tok) {
                stop_reason = crate::engine::generate::StopReason::Cancelled;
                break;
            }

            if generated_count >= cfg.min_tokens && cfg.stop_tokens.contains(&next_tok) {
                stop_reason = crate::engine::generate::StopReason::StopToken(next_tok);
                break;
            }
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let tok_s = if decode_ms > 0.0 {
            (generated_count as f64 / decode_ms) * 1000.0
        } else {
            0.0
        };

        Ok(crate::engine::generate::GenerationReport {
            tokens: current_tokens,
            prompt_len: prompt_tokens.len(),
            new_tokens: generated_count,
            prefill_time_ms: 0.0,
            decode_time_ms: decode_ms,
            total_time_ms: total_ms,
            tokens_per_sec: tok_s,
            ms_per_token: if generated_count > 0 { decode_ms / generated_count as f64 } else { 0.0 },
            stop_reason,
        })
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        cfg: &crate::engine::SamplerConfig,
    ) -> Result<Vec<usize>> {
        self.generate_fast(prompt_tokens, max_new_tokens, cfg)
    }

    pub fn decode_step(&mut self, _token: usize, _pos: usize) -> Result<()> {
        Ok(())
    }

    pub fn profile_step(&mut self, input_tokens: &[usize], target_tokens: &[usize]) -> Result<()> {
        let _ = self.step(input_tokens, target_tokens)?;
        Ok(())
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
    pub fn lr(&self) -> f32 {
        self.lr
    }
    pub fn set_max_grad_norm(&mut self, max_norm: Option<f32>) {
        self.max_grad_norm = max_norm;
    }
    pub fn max_grad_norm(&self) -> Option<f32> {
        self.max_grad_norm
    }
    pub fn step_count(&self) -> usize {
        self.step_count
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    pub fn d_ff(&self) -> usize {
        self.d_ff
    }
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
    pub fn reset_moments(&mut self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // WEIGHT GETTERS & SETTERS (SafeTensors & Interoperability)
    // -----------------------------------------------------------------------
    pub fn get_tok_emb(&self) -> Result<Vec<f32>> { Ok(self.tok_emb.clone()) }
    pub fn get_pos_emb(&self) -> Result<Vec<f32>> { Ok(self.pos_emb.clone()) }
    pub fn get_ln_f_gamma(&self) -> Result<Vec<f32>> { Ok(self.ln_f_gamma.clone()) }
    pub fn get_ln_f_beta(&self) -> Result<Vec<f32>> { Ok(self.ln_f_beta.clone()) }
    pub fn get_lm_head(&self) -> Result<Vec<f32>> { Ok(self.lm_head.clone()) }

    pub fn get_layer_ln1_gamma(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_ln1_g.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_ln1_beta(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_ln1_b.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_wq(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_wq.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_wk(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_wk.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_wv(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_wv.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_wo(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_wo.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_bq(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_bq.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_bk(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_bk.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_bv(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_bv.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_bo(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_bo.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_ln2_gamma(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_ln2_g.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_ln2_beta(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_ln2_b.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_w1(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_w1.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_b1(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_b1.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_w2(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_w2.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }
    pub fn get_layer_b2(&self, l: usize) -> Result<Vec<f32>> {
        self.layer_b2.get(l).cloned().ok_or_else(|| anyhow!("Layer index {l} out of bounds"))
    }

    pub fn set_tok_emb(&mut self, data: &[f32]) -> Result<()> {
        self.tok_emb.clear(); self.tok_emb.extend_from_slice(data); Ok(())
    }
    pub fn set_pos_emb(&mut self, data: &[f32]) -> Result<()> {
        self.pos_emb.clear(); self.pos_emb.extend_from_slice(data); Ok(())
    }
    pub fn set_ln_f_gamma(&mut self, data: &[f32]) -> Result<()> {
        self.ln_f_gamma.clear(); self.ln_f_gamma.extend_from_slice(data); Ok(())
    }
    pub fn set_ln_f_beta(&mut self, data: &[f32]) -> Result<()> {
        self.ln_f_beta.clear(); self.ln_f_beta.extend_from_slice(data); Ok(())
    }
    pub fn set_lm_head(&mut self, data: &[f32]) -> Result<()> {
        self.lm_head.clear(); self.lm_head.extend_from_slice(data); Ok(())
    }

    pub fn set_layer_ln1_gamma(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_ln1_g.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_ln1_beta(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_ln1_b.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_wq(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_wq.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_wk(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_wk.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_wv(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_wv.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_wo(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_wo.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_bq(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_bq.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_bk(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_bk.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_bv(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_bv.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_bo(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_bo.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_ln2_gamma(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_ln2_g.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_ln2_beta(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_ln2_b.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_w1(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_w1.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_b1(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_b1.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_w2(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_w2.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }
    pub fn set_layer_b2(&mut self, l: usize, data: &[f32]) -> Result<()> {
        let target = self.layer_b2.get_mut(l).ok_or_else(|| anyhow!("Layer index {l} out of bounds"))?;
        target.clear(); target.extend_from_slice(data); Ok(())
    }

    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        use std::io::{BufWriter, Write};
        let file = std::fs::File::create(path)
            .map_err(|e| anyhow!("save_checkpoint: cannot create {:?}: {e}", path))?;
        let mut w = BufWriter::new(file);

        let mut write_vec = |data: &[f32]| -> Result<()> {
            w.write_all(&(data.len() as u64).to_le_bytes())?;
            for &val in data {
                w.write_all(&val.to_le_bytes())?;
            }
            Ok(())
        };

        write_vec(&self.tok_emb)?;
        write_vec(&self.pos_emb)?;
        write_vec(&self.ln_f_gamma)?;
        write_vec(&self.ln_f_beta)?;
        write_vec(&self.lm_head)?;

        for l in 0..self.n_layers {
            write_vec(&self.layer_ln1_g[l])?;
            write_vec(&self.layer_ln1_b[l])?;
            write_vec(&self.layer_wq[l])?;
            write_vec(&self.layer_bq[l])?;
            write_vec(&self.layer_wk[l])?;
            write_vec(&self.layer_bk[l])?;
            write_vec(&self.layer_wv[l])?;
            write_vec(&self.layer_bv[l])?;
            write_vec(&self.layer_wo[l])?;
            write_vec(&self.layer_bo[l])?;
            write_vec(&self.layer_ln2_g[l])?;
            write_vec(&self.layer_ln2_b[l])?;
            write_vec(&self.layer_w1[l])?;
            write_vec(&self.layer_b1[l])?;
            write_vec(&self.layer_w2[l])?;
            write_vec(&self.layer_b2[l])?;
        }

        w.write_all(&(self.step_count as u64).to_le_bytes())?;
        w.write_all(&self.lr.to_le_bytes())?;
        w.flush()?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
        use std::io::{BufReader, Read};
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow!("load_checkpoint: cannot open {:?}: {e}", path))?;
        let mut r = BufReader::new(file);

        let mut read_vec = |expected_len: usize| -> Result<Vec<f32>> {
            let mut len_b = [0u8; 8];
            r.read_exact(&mut len_b)?;
            let n = u64::from_le_bytes(len_b) as usize;
            if n != expected_len {
                return Err(anyhow!("checkpoint length mismatch: file={}, model={}", n, expected_len));
            }
            let mut data = vec![0.0f32; n];
            for v in &mut data {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                *v = f32::from_le_bytes(b);
            }
            Ok(data)
        };

        self.tok_emb = read_vec(self.vocab_size * self.d_model)?;
        self.pos_emb = read_vec(self.seq_len * self.d_model)?;
        self.ln_f_gamma = read_vec(self.d_model)?;
        self.ln_f_beta = read_vec(self.d_model)?;
        self.lm_head = read_vec(self.vocab_size * self.d_model)?;

        for l in 0..self.n_layers {
            self.layer_ln1_g[l] = read_vec(self.d_model)?;
            self.layer_ln1_b[l] = read_vec(self.d_model)?;
            self.layer_wq[l] = read_vec(self.d_model * self.d_model)?;
            self.layer_bq[l] = read_vec(self.d_model)?;
            self.layer_wk[l] = read_vec(self.d_model * self.d_model)?;
            self.layer_bk[l] = read_vec(self.d_model)?;
            self.layer_wv[l] = read_vec(self.d_model * self.d_model)?;
            self.layer_bv[l] = read_vec(self.d_model)?;
            self.layer_wo[l] = read_vec(self.d_model * self.d_model)?;
            self.layer_bo[l] = read_vec(self.d_model)?;
            self.layer_ln2_g[l] = read_vec(self.d_model)?;
            self.layer_ln2_b[l] = read_vec(self.d_model)?;
            self.layer_w1[l] = read_vec(self.d_ff * self.d_model)?;
            self.layer_b1[l] = read_vec(self.d_ff)?;
            self.layer_w2[l] = read_vec(self.d_model * self.d_ff)?;
            self.layer_b2[l] = read_vec(self.d_model)?;
        }

        let mut step_b = [0u8; 8];
        if r.read_exact(&mut step_b).is_ok() {
            self.step_count = u64::from_le_bytes(step_b) as usize;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_gpt2_metadata_and_setters() {
        let mut model = StaticGPT2::new(100, 64, 2, 2, 256, 32, 1e-3).unwrap();
        assert_eq!(model.vocab_size(), 100);
        assert_eq!(model.d_model(), 64);
        assert_eq!(model.n_heads(), 2);
        assert_eq!(model.n_layers(), 2);
        assert_eq!(model.d_ff(), 256);
        assert_eq!(model.seq_len(), 32);
        assert_eq!(model.lr(), 1e-3);
        assert_eq!(model.step_count(), 0);
        assert_eq!(model.max_grad_norm(), None);

        model.set_lr(5e-4);
        assert_eq!(model.lr(), 5e-4);

        model.set_max_grad_norm(Some(1.0));
        assert_eq!(model.max_grad_norm(), Some(1.0));
    }
}
