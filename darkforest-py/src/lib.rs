//! Python bindings for the Dark Forest runtime.
//!
//! The product goal is not to clone every PyTorch feature blindly, but to ship
//! a smaller, faster, more opinionated ML engine for the workloads we actually
//! need: tight CUDA control, explicit device residency, deterministic training,
//! and a predictable Rust core.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::collections::HashMap;

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

#[cfg(feature = "python")]
#[pyfunction]
fn product_name() -> PyResult<String> {
    Ok("Dark Forest".to_string())
}

#[cfg(feature = "python")]
#[pyfunction]
fn status() -> PyResult<String> {
    Ok(
        "alpha-ready: Rust autograd core, CUDA fallback path, Python bindings scaffolded"
            .to_string(),
    )
}

#[cfg(feature = "python")]
#[pyfunction]
fn smoke_test() -> PyResult<String> {
    use darkforest_core::autograd::Value;
    use darkforest_core::tensor::Tensor;

    let a = Value::leaf(
        Tensor::from_vec(vec![1.0, -1.0, 2.0], vec![3]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("tensor init failed: {e}"))
        })?,
    );
    let b = Value::leaf(
        Tensor::from_vec(vec![0.5, 1.5, -0.5], vec![3]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("tensor init failed: {e}"))
        })?,
    );

    let loss = a
        .add(&b)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("op failed: {e}")))?
        .sum()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("sum failed: {e}"))
        })?;
    loss.backward();

    let grad_a = a.grad();
    let grad_b = b.grad();
    let loss_val = loss.tensor().get(0);

    Ok(format!(
        "loss={loss_val:.6}, grad_a={grad_a:?}, grad_b={grad_b:?}"
    ))
}

#[cfg(feature = "python")]
#[pyfunction]
fn benchmark_matrix_step(steps: usize, rows: usize, cols: usize) -> PyResult<HashMap<String, f64>> {
    use darkforest_core::autograd::Value;
    use darkforest_core::tensor::Tensor;
    use std::time::Instant;

    if rows == 0 || cols == 0 || steps == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "steps, rows, and cols must all be > 0",
        ));
    }

    let device = {
        #[cfg(feature = "cuda")]
        {
            darkforest_core::tensor::Device::Cuda(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            darkforest_core::tensor::Device::Cpu
        }
    };

    let a_vec: Vec<f32> = (0..rows * cols)
        .map(|idx| ((idx as f32) + 1.0) % 7.0 - 3.0)
        .collect();
    let b_vec: Vec<f32> = (0..cols * rows)
        .map(|idx| (((idx as f32) + 2.0) % 9.0) - 4.0)
        .collect();

    let a_tensor =
        Tensor::from_vec_device(a_vec, vec![rows, cols], device.clone()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("a tensor init failed: {e}"))
        })?;
    let b_tensor =
        Tensor::from_vec_device(b_vec, vec![cols, rows], device.clone()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("b tensor init failed: {e}"))
        })?;

    // Warmup (5 iterations)
    for _ in 0..5 {
        let a = Value::leaf(a_tensor.clone());
        let b = Value::leaf(b_tensor.clone());
        let out = a.matmul(&b).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("matmul failed: {e}"))
        })?;
        let loss = out.sum().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("sum failed: {e}"))
        })?;
        loss.backward();
        let _ = darkforest_core::cuda_sync();
    }

    #[cfg(feature = "cuda")]
    let timer = darkforest_core::CudaEventTimer::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "CudaEventTimer init failed: {e}"
        ))
    })?;

    let mut samples = Vec::with_capacity(steps);
    for _ in 0..steps {
        let a = Value::leaf(a_tensor.clone());
        let b = Value::leaf(b_tensor.clone());

        #[cfg(feature = "cuda")]
        timer.start().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("timer start failed: {e}"))
        })?;
        #[cfg(not(feature = "cuda"))]
        let start = Instant::now();

        let out = a.matmul(&b).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("matmul failed: {e}"))
        })?;
        let loss = out.sum().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("sum failed: {e}"))
        })?;
        loss.backward();

        #[cfg(feature = "cuda")]
        let elapsed_ms = timer.stop_and_elapsed_ms().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("timer stop failed: {e}"))
        })?;
        #[cfg(not(feature = "cuda"))]
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        samples.push(elapsed_ms);
    }

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    let mut result = HashMap::new();
    result.insert("steps".to_string(), steps as f64);
    result.insert("rows".to_string(), rows as f64);
    result.insert("cols".to_string(), cols as f64);
    result.insert("mean_ms".to_string(), mean);
    result.insert("median_ms".to_string(), median);
    result.insert(
        "min_ms".to_string(),
        samples.iter().copied().fold(f64::INFINITY, f64::min),
    );
    result.insert(
        "max_ms".to_string(),
        samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    Ok(result)
}

#[cfg(feature = "python")]
#[pyfunction]
fn cuda_sync() -> PyResult<()> {
    darkforest_core::cuda_sync()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("cuda_sync failed: {e}")))
}

#[cfg(feature = "python")]
#[pyclass(name = "StaticGPT2")]
pub struct PyStaticGPT2 {
    inner: darkforest_core::engine::StaticGPT2,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStaticGPT2 {
    #[new]
    #[pyo3(signature = (vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr=3e-4))]
    fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        seq_len: usize,
        lr: f32,
    ) -> PyResult<Self> {
        let inner = darkforest_core::engine::StaticGPT2::new(
            vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize StaticGPT2: {e}"
            ))
        })?;
        Ok(PyStaticGPT2 { inner })
    }

    /// Run one complete training step (forward -> loss -> backward -> AdamW update).
    /// Returns the scalar cross-entropy loss value.
    fn step(&mut self, input_tokens: Vec<usize>, target_tokens: Vec<usize>) -> PyResult<f32> {
        self.inner
            .step(&input_tokens, &target_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("step failed: {e}")))
    }

    /// Run one training step captured or replayed via CUDA Graphs for minimum launch overhead.
    fn step_graph(&mut self, input_tokens: Vec<usize>, target_tokens: Vec<usize>) -> PyResult<f32> {
        self.inner
            .step_graph(&input_tokens, &target_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("step_graph failed: {e}")))
    }

    /// Run one evaluation / validation step: forward pass and cross-entropy loss calculation without backward or optimizer update.
    fn eval_step(&mut self, input_tokens: Vec<usize>, target_tokens: Vec<usize>) -> PyResult<f32> {
        self.inner
            .eval_step(&input_tokens, &target_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("eval_step failed: {e}")))
    }

    /// Run forward inference pass returning full unnormalized logits matrix `[seq_len, vocab_size]` as a 1D list.
    fn forward(&mut self, input_tokens: Vec<usize>) -> PyResult<Vec<f32>> {
        self.inner
            .forward(&input_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("forward failed: {e}")))
    }

    /// Run forward inference pass returning only the last token position logits `[vocab_size]`.
    fn forward_last_logits(&mut self, input_tokens: Vec<usize>) -> PyResult<Vec<f32>> {
        self.inner
            .forward_last_logits(&input_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("forward_last_logits failed: {e}")))
    }

    /// Predict the greedy next-token index (argmax of last token logits).
    fn predict(&mut self, input_tokens: Vec<usize>) -> PyResult<usize> {
        self.inner
            .predict(&input_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("predict failed: {e}")))
    }

    /// Autoregressive text generation using the pre-allocated static execution graph.
    #[pyo3(signature = (prompt_tokens, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9))]
    fn generate(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> PyResult<Vec<usize>> {
        let cfg = darkforest_core::engine::SamplerConfig {
            temperature,
            top_k,
            top_p,
        };
        self.inner
            .generate(&prompt_tokens, max_new_tokens, &cfg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("generate failed: {e}")))
    }

    /// Fast autoregressive generation using KV cache and zero-allocation decode loop.
    #[pyo3(signature = (prompt_tokens, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9))]
    fn generate_fast(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> PyResult<Vec<usize>> {
        let cfg = darkforest_core::engine::SamplerConfig {
            temperature,
            top_k,
            top_p,
        };
        self.inner
            .generate_fast(&prompt_tokens, max_new_tokens, &cfg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("generate_fast failed: {e}")))
    }


    /// Save all model weights, moment buffers, and optimizer state to a flat binary checkpoint file.
    fn save_checkpoint(&self, path: String) -> PyResult<()> {
        self.inner
            .save_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("save_checkpoint failed: {e}")))
    }

    /// Restore all model weights, moment buffers, and optimizer state from a checkpoint file.
    fn load_checkpoint(&mut self, path: String) -> PyResult<()> {
        self.inner
            .load_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("load_checkpoint failed: {e}")))
    }

    /// Reset AdamW moment accumulators (m and v) to zero across all weights.
    fn reset_moments(&mut self) -> PyResult<()> {
        self.inner
            .reset_moments()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("reset_moments failed: {e}")))
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    #[getter]
    fn max_grad_norm(&self) -> Option<f32> {
        self.inner.max_grad_norm()
    }

    #[setter]
    fn set_max_grad_norm(&mut self, max_norm: Option<f32>) {
        self.inner.set_max_grad_norm(max_norm);
    }

    #[getter]
    fn step_count(&self) -> usize {
        self.inner.step_count()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    #[getter]
    fn d_model(&self) -> usize {
        self.inner.d_model()
    }

    #[getter]
    fn n_heads(&self) -> usize {
        self.inner.n_heads()
    }

    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }

    #[getter]
    fn d_ff(&self) -> usize {
        self.inner.d_ff()
    }

    #[getter]
    fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticGPT2(vocab_size={}, d_model={}, n_heads={}, n_layers={}, d_ff={}, seq_len={}, lr={}, step_count={})",
            self.inner.vocab_size(),
            self.inner.d_model(),
            self.inner.n_heads(),
            self.inner.n_layers(),
            self.inner.d_ff(),
            self.inner.seq_len(),
            self.inner.lr(),
            self.inner.step_count(),
        )
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StaticLoRAConfig")]
#[derive(Clone)]
pub struct PyStaticLoRAConfig {
    pub inner: darkforest_core::engine::StaticLoRAConfig,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStaticLoRAConfig {
    #[new]
    #[pyo3(signature = (rank=8, alpha=16.0, target_modules=None, qlora_nf4=false, block_size=64))]
    fn new(
        rank: usize,
        alpha: f32,
        target_modules: Option<Vec<String>>,
        qlora_nf4: bool,
        block_size: usize,
    ) -> Self {
        let targets = target_modules.unwrap_or_else(|| vec!["wq".to_string(), "wv".to_string()]);
        Self {
            inner: darkforest_core::engine::StaticLoRAConfig {
                rank,
                alpha,
                target_modules: targets,
                qlora_nf4,
                block_size,
            },
        }
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    #[getter]
    fn scaling(&self) -> f32 {
        self.inner.scaling()
    }

    #[getter]
    fn qlora_nf4(&self) -> bool {
        self.inner.qlora_nf4
    }

    #[getter]
    fn target_modules(&self) -> Vec<String> {
        self.inner.target_modules.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticLoRAConfig(rank={}, alpha={}, scaling={}, qlora_nf4={}, targets={:?})",
            self.inner.rank,
            self.inner.alpha,
            self.inner.scaling(),
            self.inner.qlora_nf4,
            self.inner.target_modules
        )
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "StaticLoRAGPT2")]
pub struct PyStaticLoRAGPT2 {
    inner: darkforest_core::engine::StaticLoRAGPT2,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStaticLoRAGPT2 {
    #[new]
    #[pyo3(signature = (vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr=3e-4, rank=8, alpha=16.0, qlora=false))]
    fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        seq_len: usize,
        lr: f32,
        rank: usize,
        alpha: f32,
        qlora: bool,
    ) -> PyResult<Self> {
        let base = darkforest_core::engine::StaticGPT2::new(
            vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize base StaticGPT2 for LoRA: {e}"
            ))
        })?;
        let config = darkforest_core::engine::StaticLoRAConfig {
            rank,
            alpha,
            target_modules: vec!["wq".to_string(), "wv".to_string()],
            qlora_nf4: qlora,
            block_size: 64,
        };
        let inner = darkforest_core::engine::StaticLoRAGPT2::new(base, config).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize StaticLoRAGPT2: {e}"
            ))
        })?;
        Ok(PyStaticLoRAGPT2 { inner })
    }

    /// Run one parameter-efficient fine-tuning step.
    fn step(&mut self, input_tokens: Vec<usize>, target_tokens: Vec<usize>) -> PyResult<f32> {
        self.inner
            .step(&input_tokens, &target_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LoRA step failed: {e}")))
    }

    /// Run one validation / evaluation step.
    fn eval_step(&mut self, input_tokens: Vec<usize>, target_tokens: Vec<usize>) -> PyResult<f32> {
        self.inner
            .eval_step(&input_tokens, &target_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LoRA eval_step failed: {e}")))
    }

    /// Autoregressive generation using the fine-tuned LoRA model.
    #[pyo3(signature = (prompt_tokens, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9))]
    fn generate(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> PyResult<Vec<usize>> {
        let cfg = darkforest_core::engine::SamplerConfig {
            temperature,
            top_k,
            top_p,
        };
        self.inner
            .generate(&prompt_tokens, max_new_tokens, &cfg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LoRA generate failed: {e}")))
    }

    /// Fast autoregressive generation using KV cache and zero-allocation decode loop.
    #[pyo3(signature = (prompt_tokens, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9))]
    fn generate_fast(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> PyResult<Vec<usize>> {
        let cfg = darkforest_core::engine::SamplerConfig {
            temperature,
            top_k,
            top_p,
        };
        self.inner
            .generate_fast(&prompt_tokens, max_new_tokens, &cfg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LoRA generate_fast failed: {e}")))
    }


    fn merge_weights(&mut self) -> PyResult<()> {
        self.inner
            .merge_weights()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("merge_weights failed: {e}")))
    }

    #[getter]
    fn is_merged(&self) -> bool {
        self.inner.is_merged()
    }

    #[getter]
    fn total_base_params(&self) -> usize {
        self.inner.total_base_params()
    }

    #[getter]
    fn trainable_lora_params(&self) -> usize {
        self.inner.trainable_lora_params()
    }

    #[getter]
    fn trainable_fraction(&self) -> f32 {
        self.inner.trainable_fraction()
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    #[getter]
    fn step_count(&self) -> usize {
        self.inner.step_count()
    }

    fn save_lora_checkpoint(&self, path: String) -> PyResult<()> {
        self.inner
            .save_lora_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("save_lora_checkpoint failed: {e}")))
    }

    fn load_lora_checkpoint(&mut self, path: String) -> PyResult<()> {
        self.inner
            .load_lora_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("load_lora_checkpoint failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticLoRAGPT2(rank={}, alpha={}, base_params={}, trainable_lora_params={}, trainable_fraction={:.3}%)",
            self.inner.config.rank,
            self.inner.config.alpha,
            self.inner.total_base_params(),
            self.inner.trainable_lora_params(),
            self.inner.trainable_fraction(),
        )
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "EngineConfig")]
#[derive(Clone)]
pub struct PyEngineConfig {
    pub inner: darkforest_core::engine::EngineConfig,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEngineConfig {
    #[staticmethod]
    fn gpt2_124m() -> Self {
        Self { inner: darkforest_core::engine::EngineConfig::gpt2_124m() }
    }
    #[staticmethod]
    fn small() -> Self {
        Self { inner: darkforest_core::engine::EngineConfig::small() }
    }
    #[staticmethod]
    fn micro(vocab_size: usize, seq_len: usize) -> Self {
        Self { inner: darkforest_core::engine::EngineConfig::micro(vocab_size, seq_len) }
    }
    #[new]
    #[pyo3(signature = (vocab_size=50257, d_model=768, n_heads=12, n_layers=12, d_ff=3072, seq_len=128, lr=3e-4, max_grad_norm=None, lora_rank=None, lora_alpha=16.0, qlora=false))]
    fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        seq_len: usize,
        lr: f32,
        max_grad_norm: Option<f32>,
        lora_rank: Option<usize>,
        lora_alpha: f32,
        qlora: bool,
    ) -> Self {
        let lora = lora_rank.map(|r| {
            darkforest_core::engine::StaticLoRAConfig::new(r, lora_alpha).with_qlora(qlora)
        });
        Self {
            inner: darkforest_core::engine::EngineConfig {
                vocab_size,
                d_model,
                n_heads,
                n_layers,
                d_ff,
                seq_len,
                lr,
                max_grad_norm,
                weight_decay: 0.01,
                lora,
            },
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Engine")]
pub struct PyEngine {
    inner: darkforest_core::engine::Engine,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEngine {
    #[new]
    fn new(config: PyEngineConfig) -> PyResult<Self> {
        let inner = darkforest_core::engine::Engine::new(config.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Engine init failed: {e}")))?;
        Ok(PyEngine { inner })
    }

    #[staticmethod]
    fn from_checkpoint(config: PyEngineConfig, path: String) -> PyResult<Self> {
        let inner = darkforest_core::engine::Engine::from_checkpoint(config.inner, std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load engine from checkpoint: {e}")))?;
        Ok(PyEngine { inner })
    }

    fn step(&mut self, input_tokens: Vec<usize>, target_tokens: Vec<usize>) -> PyResult<f32> {
        self.inner.step(&input_tokens, &target_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("step failed: {e}")))
    }

    fn eval_step(&mut self, input_tokens: Vec<usize>, target_tokens: Vec<usize>) -> PyResult<f32> {
        self.inner.eval_step(&input_tokens, &target_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("eval_step failed: {e}")))
    }

    fn forward(&mut self, input_tokens: Vec<usize>) -> PyResult<Vec<f32>> {
        self.inner.forward(&input_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("forward failed: {e}")))
    }

    fn predict(&mut self, input_tokens: Vec<usize>) -> PyResult<usize> {
        self.inner.predict(&input_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("predict failed: {e}")))
    }

    #[pyo3(signature = (prompt_tokens, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0, stop_tokens=None))]
    fn generate(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        stop_tokens: Option<Vec<usize>>,
    ) -> PyResult<Vec<usize>> {
        let cfg = darkforest_core::engine::SamplerConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            stop_tokens: stop_tokens.unwrap_or_default(),
            min_tokens: 0,
        };
        self.inner.generate(&prompt_tokens, max_new_tokens, &cfg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("generate failed: {e}")))
    }

    #[pyo3(signature = (prompt_tokens, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0, stop_tokens=None))]
    fn generate_fast(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        stop_tokens: Option<Vec<usize>>,
    ) -> PyResult<Vec<usize>> {
        let cfg = darkforest_core::engine::SamplerConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            stop_tokens: stop_tokens.unwrap_or_default(),
            min_tokens: 0,
        };
        self.inner.generate_fast(&prompt_tokens, max_new_tokens, &cfg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("generate_fast failed: {e}")))
    }

    fn save_checkpoint(&self, path: String) -> PyResult<()> {
        self.inner.save_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("save_checkpoint failed: {e}")))
    }

    fn load_checkpoint(&mut self, path: String) -> PyResult<()> {
        self.inner.load_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("load_checkpoint failed: {e}")))
    }

    fn load_safetensors(&mut self, path: String) -> PyResult<()> {
        self.inner.load_safetensors(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("load_safetensors failed: {e}")))
    }

    fn save_safetensors(&self, path: String) -> PyResult<()> {
        self.inner.save_safetensors(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("save_safetensors failed: {e}")))
    }

    fn save_lora_checkpoint(&self, path: String) -> PyResult<()> {
        self.inner.save_lora_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("save_lora_checkpoint failed: {e}")))
    }

    fn load_lora_checkpoint(&mut self, path: String) -> PyResult<()> {
        self.inner.load_lora_checkpoint(std::path::Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("load_lora_checkpoint failed: {e}")))
    }

    fn merge_lora(&mut self) -> PyResult<()> {
        self.inner.merge_lora()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("merge_lora failed: {e}")))
    }

    #[getter]
    fn is_lora(&self) -> bool {
        self.inner.is_lora()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    #[getter]
    fn d_model(&self) -> usize {
        self.inner.d_model()
    }

    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }

    #[getter]
    fn n_heads(&self) -> usize {
        self.inner.n_heads()
    }

    #[getter]
    fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    #[getter]
    fn step_count(&self) -> usize {
        self.inner.step_count()
    }

    fn summary(&self) -> String {
        self.inner.summary()
    }

    fn __repr__(&self) -> String {
        self.inner.summary()
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn _darkforest_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(product_name, m)?)?;
    m.add_function(wrap_pyfunction!(status, m)?)?;
    m.add_function(wrap_pyfunction!(smoke_test, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_matrix_step, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_sync, m)?)?;
    m.add_class::<PyStaticGPT2>()?;
    m.add_class::<PyStaticLoRAConfig>()?;
    m.add_class::<PyStaticLoRAGPT2>()?;
    m.add_class::<PyEngineConfig>()?;
    m.add_class::<PyEngine>()?;
    Ok(())
}

