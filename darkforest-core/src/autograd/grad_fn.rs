//! GradFn trait and all concrete backward functions.
//!
//! Each op's backward creates a closure (or struct) implementing `GradFn`.
//! The engine calls `backward(grad_output) → Vec<Tensor>` during backprop.

use crate::tensor::Tensor;

/// A boxed backward function.  
/// Takes the upstream gradient (`&Tensor`) and returns per-input gradient Tensors.
pub type GradFnBox = Box<dyn GradFn + Send + Sync>;

pub trait GradFn {
    /// Compute gradients for this op's inputs.
    ///
    /// `grad_output`: upstream gradient Tensor.
    /// Returns one gradient Tensor per input that requires grad.
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;

    /// Which inputs require gradients (indices into the Value's inputs list).
    fn required_inputs(&self) -> Vec<usize>;
}

#[cfg(feature = "cuda")]
pub struct CudaAttentionGradFn {
    pub context: std::sync::Arc<darkforest_cuda::AttentionContext>,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub scale: f32,
    pub causal: bool,
}

#[cfg(feature = "cuda")]
impl GradFn for CudaAttentionGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (dq, dk, dv) = self
            .context
            .backward_device(
                self.q.data().device_tensor(),
                self.k.data().device_tensor(),
                self.v.data().device_tensor(),
                grad_output.data().device_tensor(),
                self.scale,
                self.causal,
            )
            .expect("CUDA attention backward failed");
        vec![
            Tensor::from_cuda(dq),
            Tensor::from_cuda(dk),
            Tensor::from_cuda(dv),
        ]
    }

    fn required_inputs(&self) -> Vec<usize> {
        vec![0, 1, 2]
    }
}

#[cfg(feature = "cuda")]
pub struct CudaLinearGradFn {
    pub x: Tensor,
    pub weight: Tensor,
    pub has_bias: bool,
}

#[cfg(feature = "cuda")]
impl GradFn for CudaLinearGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (grad_x, grad_weight, grad_bias) = darkforest_cuda::DeviceTensor::linear_backward(
            self.x.data().device_tensor(),
            self.weight.data().device_tensor(),
            grad_output.data().device_tensor(),
            self.has_bias,
        )
        .expect("CUDA linear backward failed");

        let mut gradients = vec![Tensor::from_cuda(grad_x), Tensor::from_cuda(grad_weight)];
        if let Some(grad_bias) = grad_bias {
            gradients.push(Tensor::from_cuda(grad_bias));
        }
        gradients
    }

    fn required_inputs(&self) -> Vec<usize> {
        if self.has_bias {
            vec![0, 1, 2]
        } else {
            vec![0, 1]
        }
    }
}

// ---------------------------------------------------------------------------
// AddGradFn
// ---------------------------------------------------------------------------
pub struct AddGradFn {
    pub n_inputs: usize, // 2
}

impl GradFn for AddGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.clone(), grad_output.clone()]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0, 1]
    }
}

// ---------------------------------------------------------------------------
// ScaleGradFn
// ---------------------------------------------------------------------------
pub struct ScaleGradFn {
    pub alpha: f32,
}

impl GradFn for ScaleGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda() && self.alpha == self.alpha {
            #[cfg(feature = "cuda")]
            {
                let dev = grad_output
                    .data()
                    .device_tensor()
                    .scale(self.alpha)
                    .unwrap();
                return vec![Tensor::from_cuda(dev)];
            }
        }
        let data: Vec<f32> = grad_output
            .to_vec()
            .iter()
            .map(|&g| g * self.alpha)
            .collect();
        vec![Tensor::from_vec(data, grad_output.shape.clone()).unwrap()]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

// ---------------------------------------------------------------------------
// MulGradFn
// ---------------------------------------------------------------------------
pub struct MulGradFn {
    pub a: Tensor,
    pub b: Tensor,
}

impl GradFn for MulGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda() && self.a.is_cuda() && self.b.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let (ga, gb) = self
                    .a
                    .data()
                    .device_tensor()
                    .mul_backward(
                        self.b.data().device_tensor(),
                        grad_output.data().device_tensor(),
                    )
                    .unwrap();
                return vec![Tensor::from_cuda(ga), Tensor::from_cuda(gb)];
            }
        }
        let a_vec = self.a.to_vec();
        let b_vec = self.b.to_vec();
        let (ga, gb) = crate::ops::add::mul_backward(&grad_output.to_vec(), &a_vec, &b_vec);
        vec![
            Tensor::from_vec(ga, self.a.shape.clone()).unwrap(),
            Tensor::from_vec(gb, self.b.shape.clone()).unwrap(),
        ]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0, 1]
    }
}

// ---------------------------------------------------------------------------
// MatMulGradFn
// ---------------------------------------------------------------------------
pub struct MatMulGradFn {
    pub a: Tensor,
    pub b: Tensor,
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl GradFn for MatMulGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda() && self.a.is_cuda() && self.b.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let (ga, gb) = darkforest_cuda::DeviceTensor::matmul_backward(
                    self.a.data().device_tensor(),
                    self.b.data().device_tensor(),
                    grad_output.data().device_tensor(),
                )
                .expect("CUDA matmul backward failed");
                return vec![Tensor::from_cuda(ga), Tensor::from_cuda(gb)];
            }
        }
        let (ga, gb) = crate::ops::matmul::matmul_backward(
            &grad_output.to_vec(),
            &self.a.to_vec(),
            &self.b.to_vec(),
            self.batch,
            self.m,
            self.k,
            self.n,
        );
        vec![
            Tensor::from_vec_device(ga, self.a.shape.clone(), grad_output.device.clone()).unwrap(),
            Tensor::from_vec_device(gb, self.b.shape.clone(), grad_output.device.clone()).unwrap(),
        ]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0, 1]
    }
}

// ---------------------------------------------------------------------------
// AddBiasGradFn
// ---------------------------------------------------------------------------
pub struct AddBiasGradFn {
    pub batch_size: usize,
    pub features: usize,
    pub in_shape: Vec<usize>,
}

impl GradFn for AddBiasGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (gi, gb) = crate::ops::add::add_bias_backward(
            &grad_output.to_vec(),
            self.batch_size,
            self.features,
        );
        vec![
            Tensor::from_vec_device(gi, self.in_shape.clone(), grad_output.device.clone()).unwrap(),
            Tensor::from_vec_device(gb, vec![self.features], grad_output.device.clone()).unwrap(),
        ]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0, 1]
    }
}

// ---------------------------------------------------------------------------
// SoftmaxGradFn
// ---------------------------------------------------------------------------
pub struct SoftmaxGradFn {
    pub saved_output: Tensor,
    pub vocab: usize,
}

impl GradFn for SoftmaxGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda() && self.saved_output.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let gx = darkforest_cuda::DeviceTensor::softmax_backward(
                    self.saved_output.data().device_tensor(),
                    grad_output.data().device_tensor(),
                )
                .expect("CUDA softmax backward failed");
                return vec![Tensor::from_cuda(gx)];
            }
        }
        let res = crate::ops::softmax::softmax_backward(
            &self.saved_output.to_vec(),
            &grad_output.to_vec(),
            self.vocab,
        );
        vec![Tensor::from_vec_device(
            res,
            self.saved_output.shape.clone(),
            grad_output.device.clone(),
        )
        .unwrap()]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

// ---------------------------------------------------------------------------
// LayerNormGradFn
// ---------------------------------------------------------------------------
pub struct LayerNormGradFn {
    pub x: Tensor,
    pub gamma: Tensor,
    pub means: Tensor,
    pub rstds: Tensor,
    pub has_gamma: bool,
    pub has_beta: bool,
    pub batch: usize,
    pub features: usize,
}

impl GradFn for LayerNormGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda()
            && self.x.is_cuda()
            && (!self.has_gamma || self.gamma.is_cuda())
            && self.means.is_cuda()
            && self.rstds.is_cuda()
        {
            #[cfg(feature = "cuda")]
            {
                let g_out_guard = grad_output.data();
                let x_guard = self.x.data();
                let gamma_guard = if self.has_gamma {
                    Some(self.gamma.data())
                } else {
                    None
                };
                let means_guard = self.means.data();
                let rstds_guard = self.rstds.data();

                let (gx, gg, gb) = darkforest_cuda::DeviceTensor::layernorm_backward(
                    g_out_guard.device_tensor(),
                    x_guard.device_tensor(),
                    gamma_guard.as_ref().map(|g| g.device_tensor()),
                    means_guard.device_tensor(),
                    rstds_guard.device_tensor(),
                )
                .unwrap();
                let mut res = vec![Tensor::from_cuda(gx)];
                if let Some(g) = gg {
                    res.push(Tensor::from_cuda(g));
                }
                if let Some(b) = gb {
                    res.push(Tensor::from_cuda(b));
                }
                return res;
            }
        }
        let (gx, gg, gb) = crate::ops::layernorm::layernorm_backward(
            &grad_output.to_vec(),
            &self.x.to_vec(),
            &self.gamma.to_vec(),
            &self.means.to_vec(),
            &self.rstds.to_vec(),
            self.batch,
            self.features,
        );
        vec![
            Tensor::from_vec(gx, self.x.shape.clone()).unwrap(),
            Tensor::from_vec(gg, self.gamma.shape.clone()).unwrap(),
            Tensor::from_vec(gb, self.gamma.shape.clone()).unwrap(),
        ]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0, 1, 2]
    }
}

// ---------------------------------------------------------------------------
// GeluGradFn
// ---------------------------------------------------------------------------
pub struct GeluGradFn {
    pub x: Tensor,
}

impl GradFn for GeluGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda() && self.x.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let gx = self
                    .x
                    .data()
                    .device_tensor()
                    .gelu_backward(grad_output.data().device_tensor())
                    .unwrap();
                return vec![Tensor::from_cuda(gx)];
            }
        }
        let gx = crate::ops::activation::gelu_backward(&grad_output.to_vec(), &self.x.to_vec());
        vec![Tensor::from_vec(gx, self.x.shape.clone()).unwrap()]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

// ---------------------------------------------------------------------------
// CrossEntropyGradFn
// ---------------------------------------------------------------------------
pub struct CrossEntropyGradFn {
    pub probs: Tensor,       // softmax probabilities
    pub targets: Vec<usize>, // token indices
    pub batch: usize,
    pub vocab: usize,
}

impl GradFn for CrossEntropyGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda() && self.probs.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let gx = darkforest_cuda::DeviceTensor::cross_entropy_backward(
                    self.probs.data().device_tensor(),
                    &self.targets,
                    grad_output.data().device_tensor(),
                )
                .expect("CUDA cross-entropy backward failed");
                return vec![Tensor::from_cuda(gx)];
            }
        }
        let scale = grad_output.get(0) / self.batch as f32;
        let mut grad = self.probs.to_vec();
        for (b, &t) in self.targets.iter().enumerate() {
            grad[b * self.vocab + t] -= 1.0;
        }
        for g in grad.iter_mut() {
            *g *= scale;
        }
        vec![Tensor::from_vec_device(
            grad,
            vec![self.batch, self.vocab],
            grad_output.device.clone(),
        )
        .unwrap()]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

// ---------------------------------------------------------------------------
// EmbeddingGradFn
// ---------------------------------------------------------------------------
pub struct EmbeddingGradFn {
    pub indices: Vec<usize>,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl GradFn for EmbeddingGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        if grad_output.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let gw = darkforest_cuda::DeviceTensor::embedding_backward(
                    &self.indices,
                    grad_output.data().device_tensor(),
                    self.vocab_size,
                    self.embed_dim,
                )
                .unwrap();
                return vec![Tensor::from_cuda(gw)];
            }
        }
        let mut grad_w = vec![0.0f32; self.vocab_size * self.embed_dim];
        let g_out = grad_output.to_vec();
        for (pos, &idx) in self.indices.iter().enumerate() {
            let src_off = pos * self.embed_dim;
            let dst_off = idx * self.embed_dim;
            for d in 0..self.embed_dim {
                grad_w[dst_off + d] += g_out[src_off + d];
            }
        }
        vec![Tensor::from_vec(grad_w, vec![self.vocab_size, self.embed_dim]).unwrap()]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}
