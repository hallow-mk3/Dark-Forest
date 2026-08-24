//! `Value` — a tracked tensor node in the autograd graph.
//!
//! A `Value` wraps a `Tensor` and carries:
//!   - a `GradFn` for the backward pass
//!   - references to its input `Value`s
//!
//! The `backward()` method on the final scalar Loss `Value` triggers
//! topological traversal of the graph and accumulates gradients.

use crate::autograd::grad_fn::GradFnBox;
use crate::autograd::tape::is_grad_enabled;
use crate::ops::{
    activation::gelu as op_gelu,
    add::{add as op_add, add_bias as op_add_bias, mul as op_mul, scale as op_scale},
    layernorm::layernorm as op_layernorm,
    matmul::matmul as op_matmul,
    softmax::softmax as op_softmax,
};
use crate::tensor::{Device, Tensor};
use anyhow::Result;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// ValueInner — the shared node in the graph
// ---------------------------------------------------------------------------
pub(crate) struct ValueInner {
    pub(crate) tensor: Tensor,
    pub(crate) grad: Option<Tensor>, // accumulated gradient Tensor for this node
    pub(crate) grad_fn: Option<GradFnBox>, // None for leaf values
    pub(crate) inputs: Vec<Value>,   // input nodes (shared ownership)
}

// ---------------------------------------------------------------------------
// Value — cheap Arc-clone handle
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Value(pub(crate) Arc<Mutex<ValueInner>>);

impl Value {
    pub fn update_tensor_data(&self, new_data: Vec<f32>) {
        let inner = self.0.lock().unwrap();
        let shape = inner.tensor.shape.clone();
        let device = inner.tensor.device.clone();
        let mut storage = inner.tensor.storage.write().unwrap();
        *storage = crate::tensor::TensorData::from_vec(new_data, &shape, &device);
    }

    pub fn to_device(&self, device: Device) -> Result<Value> {
        let inner = self.0.lock().unwrap();
        let t = inner.tensor.to_device(device)?;
        Ok(Value::leaf(t))
    }

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a leaf `Value` (no grad_fn).
    pub fn leaf(tensor: Tensor) -> Self {
        let dev = tensor.device.clone();
        let shape = tensor.shape.clone();
        Value(Arc::new(Mutex::new(ValueInner {
            grad: Some(Tensor::zeros_device(shape, dev)),
            tensor,
            grad_fn: None,
            inputs: vec![],
        })))
    }

    /// Create from op result with a GradFn.
    pub(crate) fn from_op(tensor: Tensor, grad_fn: GradFnBox, inputs: Vec<Value>) -> Self {
        let dev = tensor.device.clone();
        let shape = tensor.shape.clone();
        Value(Arc::new(Mutex::new(ValueInner {
            grad: Some(Tensor::zeros_device(shape, dev)),
            tensor,
            grad_fn: Some(grad_fn),
            inputs,
        })))
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    pub fn tensor(&self) -> Tensor {
        self.0.lock().unwrap().tensor.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.lock().unwrap().tensor.shape.clone()
    }

    pub fn numel(&self) -> usize {
        self.0.lock().unwrap().tensor.numel()
    }

    pub fn grad(&self) -> Vec<f32> {
        self.0
            .lock()
            .unwrap()
            .grad
            .as_ref()
            .map(|g| g.to_vec())
            .unwrap_or_default()
    }

    pub fn grad_tensor(&self) -> Option<Tensor> {
        self.0.lock().unwrap().grad.clone()
    }

    pub fn zero_grad(&self) {
        let mut inner = self.0.lock().unwrap();
        if let Some(ref g) = inner.grad {
            // Zero the storage of the grad-accumulation Tensor directly
            let mut guard = g.data_mut();
            use crate::tensor::Storage;
            match &mut guard.storage {
                Storage::Cpu(v) => v.fill(0.0f32),
                #[cfg(feature = "cuda")]
                Storage::Cuda(d) => d.zero_().expect("CUDA zero_grad failed"),
            }
        } else {
            inner.grad = Some(Tensor::zeros_device(
                inner.tensor.shape.clone(),
                inner.tensor.device.clone(),
            ));
        }
    }

    // -----------------------------------------------------------------------
    // backward()
    // -----------------------------------------------------------------------

    /// Backpropagate gradients from this value (should be scalar loss).
    /// Seeds this node's gradient with 1.0 and traverses in reverse topo order.
    pub fn backward(&self) {
        let root_grad = {
            let mut inner = self.0.lock().unwrap();
            let n = inner.tensor.numel();
            let norm = 1.0f32 / n as f32;
            let shape = inner.tensor.shape.clone();
            let device = inner.tensor.device.clone();
            let seed_tensor = if n == 1 {
                Tensor::from_vec_device(vec![1.0], shape, device).unwrap()
            } else {
                Tensor::from_vec_device(vec![norm; n], shape, device).unwrap()
            };
            inner.grad = Some(seed_tensor.clone());
            seed_tensor
        };
        self._backward_recursive(&root_grad);
    }

    pub(crate) fn _backward_recursive(&self, upstream_grad: &Tensor) {
        let (grad_fn_opt, inputs) = {
            let inner = self.0.lock().unwrap();
            if inner.grad_fn.is_none() {
                return;
            }
            (true, inner.inputs.clone())
        };

        if !grad_fn_opt {
            return;
        }

        let (grad_inputs, req) = {
            let inner = self.0.lock().unwrap();
            if let Some(ref gfn) = inner.grad_fn {
                (gfn.backward(upstream_grad), gfn.required_inputs())
            } else {
                return;
            }
        };

        for (grad_idx, &input_idx) in req.iter().enumerate() {
            if let Some(input) = inputs.get(input_idx) {
                if let Some(grad_t) = grad_inputs.get(grad_idx) {
                    {
                        let mut inp_inner = input.0.lock().unwrap();
                        if let Some(ref inp_g) = inp_inner.grad {
                            // inp_g is the Tensor holding accumulated grad in its storage
                            inp_g.add_storage_inplace(grad_t);
                        } else {
                            // Initialize a zero grad tensor on matching device then add
                            let new_g = Tensor::zeros_device(
                                inp_inner.tensor.shape.clone(),
                                inp_inner.tensor.device.clone(),
                            );
                            new_g.add_storage_inplace(grad_t);
                            inp_inner.grad = Some(new_g);
                        }
                    }
                    input._backward_recursive(grad_t);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Differentiable ops on Value
    // -----------------------------------------------------------------------

    pub fn add(&self, other: &Value) -> Result<Value> {
        let t_self = self.tensor();
        let t_other = other.tensor();
        if t_self.device != t_other.device {
            return Err(anyhow::anyhow!("add requires tensors on the same device"));
        }

        let out = if t_self.is_cuda() && t_other.is_cuda() {
            // The current custom CUDA add path is not stable in the full model stack.
            // Use the proven CPU kernel and re-upload the result to the original device
            // so the graph stays device-consistent without corrupting CUDA state.
            let cpu_self = t_self.to_device(Device::Cpu)?;
            let cpu_other = t_other.to_device(Device::Cpu)?;
            let cpu_out = op_add(&cpu_self, &cpu_other)?;
            Tensor::from_vec_device(cpu_out.to_vec(), cpu_out.shape.clone(), t_self.device.clone())?
        } else {
            op_add(&t_self, &t_other)?
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::AddGradFn { n_inputs: 2 });
        Ok(Value::from_op(
            out,
            grad_fn,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn scale(&self, alpha: f32) -> Result<Value> {
        let t = self.tensor();
        let out = if t.is_cuda() {
            let cpu_t = t.to_device(Device::Cpu)?;
            op_scale(&cpu_t, alpha)?
        } else {
            op_scale(&t, alpha)?
        };

        if t.is_cuda() {
            let out = Tensor::from_vec_device(out.to_vec(), out.shape.clone(), t.device.clone())?;
            if !is_grad_enabled() {
                return Ok(Value::leaf(out));
            }
            let grad_fn = Box::new(crate::autograd::grad_fn::ScaleGradFn { alpha });
            return Ok(Value::from_op(out, grad_fn, vec![self.clone()]));
        }

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::ScaleGradFn { alpha });
        Ok(Value::from_op(out, grad_fn, vec![self.clone()]))
    }

    pub fn mul(&self, other: &Value) -> Result<Value> {
        let t_self = self.tensor();
        let t_other = other.tensor();
        if t_self.device != t_other.device {
            return Err(anyhow::anyhow!("mul requires tensors on the same device"));
        }
        let out = if t_self.is_cuda() && t_other.is_cuda() {
            let cpu_self = t_self.to_device(Device::Cpu)?;
            let cpu_other = t_other.to_device(Device::Cpu)?;
            let cpu_out = op_mul(&cpu_self, &cpu_other)?;
            Tensor::from_vec_device(cpu_out.to_vec(), cpu_out.shape.clone(), t_self.device.clone())?
        } else {
            op_mul(&t_self, &t_other)?
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::MulGradFn {
            a: t_self,
            b: t_other,
        });
        Ok(Value::from_op(
            out,
            grad_fn,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn matmul(&self, other: &Value) -> Result<Value> {
        let t_self = self.tensor();
        let t_other = other.tensor();
        if t_self.device != t_other.device {
            return Err(anyhow::anyhow!(
                "matmul requires tensors on the same device"
            ));
        }
        let ndim = t_self.ndim();
        let m = t_self.shape[ndim - 2];
        let k = t_self.shape[ndim - 1];
        let n = t_other.shape[t_other.ndim() - 1];
        let batch: usize = t_self.shape[..ndim - 2].iter().product::<usize>().max(1);

        let out = if t_self.is_cuda() && t_other.is_cuda() && ndim == 2 && t_other.ndim() == 2 {
            // Keep the matmul result stable at the Python/Rust graph boundary while the
            // custom CUDA GEMM path is still being corrected. This keeps the full model
            // forward pass numerically correct and avoids the device state corruption that
            // causes the later cudaMalloc/cudaMemcpy 700 failures.
            let cpu_self = t_self.to_device(Device::Cpu)?;
            let cpu_other = t_other.to_device(Device::Cpu)?;
            let cpu_out = op_matmul(&cpu_self, &cpu_other)?;
            Tensor::from_vec_device(cpu_out.to_vec(), cpu_out.shape.clone(), t_self.device.clone())?
        } else {
            op_matmul(&t_self, &t_other)?
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::MatMulGradFn {
            a: t_self,
            b: t_other,
            batch,
            m,
            k,
            n,
        });
        Ok(Value::from_op(
            out,
            grad_fn,
            vec![self.clone(), other.clone()],
        ))
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_attention(
        &self,
        key: &Value,
        value: &Value,
        context: std::sync::Arc<darkforest_cuda::AttentionContext>,
        _seq_len: usize,
        _d_head: usize,
        scale: f32,
        causal: bool,
    ) -> Result<Value> {
        let q_tensor = self.tensor().to_device(Device::Cuda(0))?;
        let k_tensor = key.tensor().to_device(Device::Cuda(0))?;
        let v_tensor = value.tensor().to_device(Device::Cuda(0))?;
        let q_guard = q_tensor.data();
        let k_guard = k_tensor.data();
        let v_guard = v_tensor.data();
        let dev_out = context.forward_device(
            q_guard.device_tensor(),
            k_guard.device_tensor(),
            v_guard.device_tensor(),
            scale,
            causal,
        )?;
        drop(q_guard);
        drop(k_guard);
        drop(v_guard);
        let tensor = Tensor::from_cuda(dev_out);

        if !is_grad_enabled() {
            return Ok(Value::leaf(tensor));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::CudaAttentionGradFn {
            context,
            q: q_tensor,
            k: k_tensor,
            v: v_tensor,
            scale,
            causal,
        });
        Ok(Value::from_op(
            tensor,
            grad_fn,
            vec![self.clone(), key.clone(), value.clone()],
        ))
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_linear(
        &self,
        weight: &Value,
        bias: Option<&Value>,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Value> {
        let t_self = self.tensor().to_device(Device::Cuda(0))?;
        let t_weight = weight.tensor().to_device(Device::Cuda(0))?;
        let t_bias = match bias {
            Some(b) => Some(b.tensor().to_device(Device::Cuda(0))?),
            None => None,
        };
        let dev_x = t_self.data();
        let dev_w = t_weight.data();
        let dev_b_guard = t_bias.as_ref().map(|b| b.data());
        let dev_b = dev_b_guard.as_ref().map(|g| g.device_tensor());
        let dev_out = dev_x.device_tensor().linear(dev_w.device_tensor(), dev_b)?;
        drop(dev_x);
        drop(dev_w);
        drop(dev_b_guard);
        let out = Tensor::from_cuda(dev_out);

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let has_bias = bias.is_some();
        let grad_fn = Box::new(crate::autograd::grad_fn::CudaLinearGradFn {
            x: t_self,
            weight: t_weight,
            has_bias,
        });
        let mut inputs = vec![self.clone(), weight.clone()];
        if let Some(bias) = bias {
            inputs.push(bias.clone());
        }
        Ok(Value::from_op(out, grad_fn, inputs))
    }

    pub fn add_bias(&self, bias: &Value) -> Result<Value> {
        let t_self = self.tensor();
        let t_bias = bias.tensor();
        let features = *t_self.shape.last().unwrap();
        let batch_size = t_self.numel() / features;
        let in_shape = t_self.shape.clone();
        let out = if t_self.is_cuda() && t_bias.is_cuda() {
            let cpu_self = t_self.to_device(Device::Cpu)?;
            let cpu_bias = t_bias.to_device(Device::Cpu)?;
            let cpu_out = op_add_bias(&cpu_self, &cpu_bias)?;
            Tensor::from_vec_device(cpu_out.to_vec(), cpu_out.shape.clone(), t_self.device.clone())?
        } else {
            op_add_bias(&t_self, &t_bias)?
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::AddBiasGradFn {
            batch_size,
            features,
            in_shape,
        });
        Ok(Value::from_op(
            out,
            grad_fn,
            vec![self.clone(), bias.clone()],
        ))
    }

    pub fn sum(&self) -> Result<Value> {
        let t = self.tensor();
        let shape = t.shape.clone();
        let dev = t.device.clone();

        let out = if t.is_cuda() {
            let cpu_t = t.to_device(Device::Cpu)?;
            let s: f32 = cpu_t.to_vec().iter().sum();
            Tensor::from_vec_device(vec![s], vec![1], dev.clone())?
        } else {
            let s: f32 = t.to_vec().iter().sum();
            Tensor::from_vec_device(vec![s], vec![1], dev.clone())?
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        struct SumGradFn {
            shape: Vec<usize>,
            device: Device,
        }
        impl crate::autograd::grad_fn::GradFn for SumGradFn {
            fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
                if self.device != Device::Cpu {
                    #[cfg(feature = "cuda")]
                    {
                        return vec![Tensor::ones_device(self.shape.clone(), self.device.clone())];
                    }
                }
                let g = grad_output.get(0);
                let n = crate::tensor::numel(&self.shape);
                let data = vec![g; n];
                vec![
                    Tensor::from_vec_device(data, self.shape.clone(), self.device.clone()).unwrap(),
                ]
            }
            fn required_inputs(&self) -> Vec<usize> {
                vec![0]
            }
        }

        Ok(Value::from_op(
            out,
            Box::new(SumGradFn { shape, device: dev }),
            vec![self.clone()],
        ))
    }

    pub fn softmax(&self) -> Result<Value> {
        let t = self.tensor();
        let vocab = *t.shape.last().unwrap();

        // The custom CUDA softmax kernel still corrupts device memory in the current
        // backend, which leads to cudaMalloc/cudaMemcpy returning code 700 after the
        // first kernel invocation. Keep the math correct by falling back to the stable
        // CPU implementation until the kernel is rewritten and validated.
        let out = op_softmax(&t)?;

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::SoftmaxGradFn {
            saved_output: out.clone(),
            vocab,
        });
        Ok(Value::from_op(out, grad_fn, vec![self.clone()]))
    }

    pub fn layernorm(&self, gamma: &Value, beta: &Value) -> Result<Value> {
        let t_self = self.tensor();
        let t_gamma = gamma.tensor();
        let t_beta = beta.tensor();

        if t_self.is_cuda() {
            let cpu_self = t_self.to_device(Device::Cpu)?;
            let cpu_gamma = t_gamma.to_device(Device::Cpu)?;
            let cpu_beta = t_beta.to_device(Device::Cpu)?;
            let (out, means, rstds) = op_layernorm(&cpu_self, &cpu_gamma, &cpu_beta)?;
            let out_t = Tensor::from_vec_device(out.to_vec(), out.shape.clone(), t_self.device.clone())?;
            if !is_grad_enabled() {
                return Ok(Value::leaf(out_t));
            }
            let features = *t_self.shape.last().unwrap();
            let batch = t_self.numel() / features;
            let grad_fn = Box::new(crate::autograd::grad_fn::LayerNormGradFn {
                x: t_self,
                gamma: t_gamma.clone(),
                means: Tensor::from_vec(means, vec![batch])?,
                rstds: Tensor::from_vec(rstds, vec![batch])?,
                has_gamma: true,
                has_beta: true,
                batch,
                features,
            });
            return Ok(Value::from_op(
                out_t,
                grad_fn,
                vec![self.clone(), gamma.clone(), beta.clone()],
            ));
        }

        let (out, means, rstds) = op_layernorm(&t_self, &t_gamma, &t_beta)?;

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let features = *t_self.shape.last().unwrap();
        let batch = t_self.numel() / features;
        let grad_fn = Box::new(crate::autograd::grad_fn::LayerNormGradFn {
            x: t_self,
            gamma: t_gamma.clone(),
            means: Tensor::from_vec(means, vec![batch])?,
            rstds: Tensor::from_vec(rstds, vec![batch])?,
            has_gamma: true,
            has_beta: true,
            batch,
            features,
        });
        Ok(Value::from_op(
            out,
            grad_fn,
            vec![self.clone(), gamma.clone(), beta.clone()],
        ))
    }

    pub fn gelu(&self) -> Result<Value> {
        let t = self.tensor();
        let out = if t.is_cuda() {
            let cpu_t = t.to_device(Device::Cpu)?;
            let cpu_out = op_gelu(&cpu_t)?;
            Tensor::from_vec_device(cpu_out.to_vec(), cpu_out.shape.clone(), t.device.clone())?
        } else {
            op_gelu(&t)?
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::GeluGradFn { x: t });
        Ok(Value::from_op(out, grad_fn, vec![self.clone()]))
    }

    /// Cross-entropy loss from logits (shape [batch, vocab]) and target indices.
    ///
    /// Returns scalar `Value` (shape [1]).
    pub fn cross_entropy_loss(&self, targets: &[usize]) -> Result<Value> {
        let logits = self.tensor();
        let ndim = logits.ndim();
        let vocab = logits.shape[ndim - 1];
        let batch = logits.numel() / vocab;

        if targets.len() != batch || targets.iter().any(|&target| target >= vocab) {
            return Err(anyhow::anyhow!("cross entropy targets do not match logits"));
        }

        // Softmax over logits
        let (loss_t, probs_t) = if logits.is_cuda() {
            let cpu_logits = logits.to_device(Device::Cpu)?;
            let probs = op_softmax(&cpu_logits)?;
            let probs_data = probs.to_vec();
            let mut loss = 0.0f32;
            for (b, &target) in targets.iter().enumerate() {
                loss -= probs_data[b * vocab + target].max(1e-12).ln();
            }
            loss /= batch as f32;
            (
                Tensor::from_vec_device(vec![loss], vec![1], logits.device.clone())?,
                Tensor::from_vec_device(probs.to_vec(), probs.shape.clone(), logits.device.clone())?,
            )
        } else {
            let probs = op_softmax(&logits)?;
            let probs_data = probs.to_vec();
            let mut loss = 0.0f32;
            for (b, &target) in targets.iter().enumerate() {
                loss -= probs_data[b * vocab + target].max(1e-12).ln();
            }
            loss /= batch as f32;
            (Tensor::from_vec(vec![loss], vec![1])?, probs)
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(loss_t));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::CrossEntropyGradFn {
            probs: probs_t,
            targets: targets.to_vec(),
            batch,
            vocab,
        });
        Ok(Value::from_op(loss_t, grad_fn, vec![self.clone()]))
    }

    /// Embedding lookup: indices → [batch, embed_dim]
    pub fn embedding_lookup(weight: &Value, indices: &[usize]) -> Result<Value> {
        let w = weight.tensor();
        let vocab_size = w.shape[0];
        let embed_dim = w.shape[1];
        let seq_len = indices.len();

        let out = if w.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let dev_out = darkforest_cuda::DeviceTensor::embedding_lookup(
                    indices,
                    w.data().device_tensor(),
                )?;
                Tensor::from_cuda(dev_out)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let w_data = w.to_vec();
                let mut out_data = vec![0.0f32; seq_len * embed_dim];
                for (pos, &idx) in indices.iter().enumerate() {
                    let src = idx * embed_dim;
                    let dst = pos * embed_dim;
                    out_data[dst..dst + embed_dim].copy_from_slice(&w_data[src..src + embed_dim]);
                }
                Tensor::from_vec(out_data, vec![seq_len, embed_dim])?
            }
        } else {
            let w_data = w.to_vec();
            let mut out_data = vec![0.0f32; seq_len * embed_dim];
            for (pos, &idx) in indices.iter().enumerate() {
                let src = idx * embed_dim;
                let dst = pos * embed_dim;
                out_data[dst..dst + embed_dim].copy_from_slice(&w_data[src..src + embed_dim]);
            }
            Tensor::from_vec(out_data, vec![seq_len, embed_dim])?
        };

        if !is_grad_enabled() {
            return Ok(Value::leaf(out));
        }

        let grad_fn = Box::new(crate::autograd::grad_fn::EmbeddingGradFn {
            indices: indices.to_vec(),
            vocab_size,
            embed_dim,
        });
        Ok(Value::from_op(out, grad_fn, vec![weight.clone()]))
    }
}
