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
// ReluGradFn
// ---------------------------------------------------------------------------
pub struct ReluGradFn {
    pub x: Tensor,
}

impl GradFn for ReluGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let x_vec = self.x.to_vec();
        let g_vec = grad_output.to_vec();
        let gx: Vec<f32> = x_vec
            .iter()
            .zip(g_vec.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect();
        vec![Tensor::from_vec(gx, self.x.shape.clone()).unwrap()]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

// ---------------------------------------------------------------------------
// ReshapeGradFn
// ---------------------------------------------------------------------------
pub struct ReshapeGradFn {
    pub orig_shape: Vec<usize>,
}

impl GradFn for ReshapeGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.reshape(self.orig_shape.clone()).unwrap()]
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


// ---------------------------------------------------------------------------
// TransposeLastTwoGradFn
// ---------------------------------------------------------------------------
pub struct TransposeLastTwoGradFn;

impl GradFn for TransposeLastTwoGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let gt = grad_output.transpose_last_two().expect("transpose_last_two backward failed");
        vec![gt]
    }
    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

// ---------------------------------------------------------------------------
// Conv2dGradFn
// ---------------------------------------------------------------------------
pub struct Conv2dGradFn {
    pub x: Tensor,
    pub weight: Tensor,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub has_bias: bool,
}

impl GradFn for Conv2dGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let xt = &self.x;
        let wt = &self.weight;
        let (batch, in_c, h, w) = (xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]);
        let (out_c, _, kh, kw) = (wt.shape[0], wt.shape[1], wt.shape[2], wt.shape[3]);
        let (out_h, out_w) = (grad_output.shape[2], grad_output.shape[3]);
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let g_data = grad_output.to_vec();
        let x_data = xt.to_vec();
        let w_data = wt.to_vec();

        let mut grad_x = vec![0.0f32; xt.numel()];
        let mut grad_w = vec![0.0f32; wt.numel()];
        let mut grad_b = if self.has_bias { vec![0.0f32; out_c] } else { Vec::new() };

        for b in 0..batch {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let g_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                        let g = g_data[g_idx];
                        if self.has_bias {
                            grad_b[oc] += g;
                        }
                        let ih_start = (oh * sh) as isize - ph as isize;
                        let iw_start = (ow * sw) as isize - pw as isize;

                        for ic in 0..in_c {
                            for f_h in 0..kh {
                                for f_w in 0..kw {
                                    let cur_h = ih_start + f_h as isize;
                                    let cur_w = iw_start + f_w as isize;
                                    if cur_h >= 0 && cur_h < h as isize && cur_w >= 0 && cur_w < w as isize {
                                        let in_idx = b * (in_c * h * w) + ic * (h * w) + (cur_h as usize) * w + (cur_w as usize);
                                        let w_idx = oc * (in_c * kh * kw) + ic * (kh * kw) + f_h * kw + f_w;
                                        grad_w[w_idx] += g * x_data[in_idx];
                                        grad_x[in_idx] += g * w_data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let gx = Tensor::from_vec_device(grad_x, xt.shape.clone(), xt.device.clone()).unwrap();
        let gw = Tensor::from_vec_device(grad_w, wt.shape.clone(), wt.device.clone()).unwrap();
        if self.has_bias {
            let gb = Tensor::from_vec_device(grad_b, vec![out_c], xt.device.clone()).unwrap();
            vec![gx, gw, gb]
        } else {
            vec![gx, gw]
        }
    }

    fn required_inputs(&self) -> Vec<usize> {
        if self.has_bias { vec![0, 1, 2] } else { vec![0, 1] }
    }
}

// ---------------------------------------------------------------------------
// Conv1dGradFn
// ---------------------------------------------------------------------------
pub struct Conv1dGradFn {
    pub x: Tensor,
    pub weight: Tensor,
    pub stride: usize,
    pub padding: usize,
    pub has_bias: bool,
}

impl GradFn for Conv1dGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let xt = &self.x;
        let wt = &self.weight;
        let (batch, in_c, l) = (xt.shape[0], xt.shape[1], xt.shape[2]);
        let (out_c, _, k) = (wt.shape[0], wt.shape[1], wt.shape[2]);
        let out_l = grad_output.shape[2];
        let s = self.stride;
        let p = self.padding;

        let g_data = grad_output.to_vec();
        let x_data = xt.to_vec();
        let w_data = wt.to_vec();

        let mut grad_x = vec![0.0f32; xt.numel()];
        let mut grad_w = vec![0.0f32; wt.numel()];
        let mut grad_b = if self.has_bias { vec![0.0f32; out_c] } else { Vec::new() };

        for b in 0..batch {
            for oc in 0..out_c {
                for ol in 0..out_l {
                    let g_idx = b * (out_c * out_l) + oc * out_l + ol;
                    let g = g_data[g_idx];
                    if self.has_bias {
                        grad_b[oc] += g;
                    }
                    let il_start = (ol * s) as isize - p as isize;
                    for ic in 0..in_c {
                        for f_k in 0..k {
                            let cur_l = il_start + f_k as isize;
                            if cur_l >= 0 && cur_l < l as isize {
                                let in_idx = b * (in_c * l) + ic * l + (cur_l as usize);
                                let w_idx = oc * (in_c * k) + ic * k + f_k;
                                grad_w[w_idx] += g * x_data[in_idx];
                                grad_x[in_idx] += g * w_data[w_idx];
                            }
                        }
                    }
                }
            }
        }

        let gx = Tensor::from_vec_device(grad_x, xt.shape.clone(), xt.device.clone()).unwrap();
        let gw = Tensor::from_vec_device(grad_w, wt.shape.clone(), wt.device.clone()).unwrap();
        if self.has_bias {
            let gb = Tensor::from_vec_device(grad_b, vec![out_c], xt.device.clone()).unwrap();
            vec![gx, gw, gb]
        } else {
            vec![gx, gw]
        }
    }

    fn required_inputs(&self) -> Vec<usize> {
        if self.has_bias { vec![0, 1, 2] } else { vec![0, 1] }
    }
}

// ---------------------------------------------------------------------------
// BatchNormGradFn
// ---------------------------------------------------------------------------
pub struct BatchNormGradFn {
    pub x: Tensor,
    pub gamma: Tensor,
    pub means: Vec<f32>,
    pub rstds: Vec<f32>,
}

impl GradFn for BatchNormGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let xt = &self.x;
        let n = xt.shape[0];
        let c = xt.shape[1];
        let spatial: usize = xt.shape[2..].iter().product::<usize>().max(1);
        let num_per_channel = n * spatial;

        let g_data = grad_output.to_vec();
        let x_data = xt.to_vec();
        let gamma_data = self.gamma.to_vec();

        let mut grad_x = vec![0.0f32; xt.numel()];
        let mut grad_gamma = vec![0.0f32; c];
        let mut grad_beta = vec![0.0f32; c];

        for ch in 0..c {
            let mean = self.means[ch];
            let rstd = self.rstds[ch];
            let g_val = gamma_data[ch];

            let mut sum_dout = 0.0f32;
            let mut sum_dout_xhat = 0.0f32;

            for b in 0..n {
                for s in 0..spatial {
                    let idx = b * (c * spatial) + ch * spatial + s;
                    let dout = g_data[idx];
                    let xhat = (x_data[idx] - mean) * rstd;
                    sum_dout += dout;
                    sum_dout_xhat += dout * xhat;
                    grad_gamma[ch] += dout * xhat;
                    grad_beta[ch] += dout;
                }
            }

            let m_f = num_per_channel as f32;
            let factor = g_val * rstd / m_f;

            for b in 0..n {
                for s in 0..spatial {
                    let idx = b * (c * spatial) + ch * spatial + s;
                    let dout = g_data[idx];
                    let xhat = (x_data[idx] - mean) * rstd;
                    grad_x[idx] = factor * (m_f * dout - sum_dout - xhat * sum_dout_xhat);
                }
            }
        }

        let gx = Tensor::from_vec_device(grad_x, xt.shape.clone(), xt.device.clone()).unwrap();
        let gg = Tensor::from_vec_device(grad_gamma, vec![c], xt.device.clone()).unwrap();
        let gb = Tensor::from_vec_device(grad_beta, vec![c], xt.device.clone()).unwrap();
        vec![gx, gg, gb]
    }

    fn required_inputs(&self) -> Vec<usize> {
        vec![0, 1, 2]
    }
}

// ---------------------------------------------------------------------------
// MaxPool2dGradFn
// ---------------------------------------------------------------------------
pub struct MaxPool2dGradFn {
    pub in_shape: Vec<usize>,
    pub max_indices: Vec<usize>,
}

impl GradFn for MaxPool2dGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let n = crate::tensor::numel(&self.in_shape);
        let mut grad_x = vec![0.0f32; n];
        let g_data = grad_output.to_vec();

        for (out_idx, &in_idx) in self.max_indices.iter().enumerate() {
            if in_idx < n {
                grad_x[in_idx] += g_data[out_idx];
            }
        }

        vec![Tensor::from_vec_device(grad_x, self.in_shape.clone(), grad_output.device.clone()).unwrap()]
    }

    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2dGradFn
// ---------------------------------------------------------------------------
pub struct AdaptiveAvgPool2dGradFn {
    pub in_shape: Vec<usize>,
    pub output_size: (usize, usize),
}

impl GradFn for AdaptiveAvgPool2dGradFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (batch, c, h, w) = (
            self.in_shape[0],
            self.in_shape[1],
            self.in_shape[2],
            self.in_shape[3],
        );
        let (out_h, out_w) = self.output_size;
        let mut grad_x = vec![0.0f32; batch * c * h * w];
        let g_data = grad_output.to_vec();

        for b in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    let start_h = (oh * h) / out_h;
                    let end_h = ((oh + 1) * h + out_h - 1) / out_h;

                    for ow in 0..out_w {
                        let start_w = (ow * w) / out_w;
                        let end_w = ((ow + 1) * w + out_w - 1) / out_w;

                        let count = ((end_h - start_h) * (end_w - start_w)).max(1) as f32;
                        let out_idx = b * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                        let g = g_data[out_idx] / count;

                        for ih in start_h..end_h {
                            for iw in start_w..end_w {
                                let in_idx = b * (c * h * w) + ch * (h * w) + ih * w + iw;
                                grad_x[in_idx] += g;
                            }
                        }
                    }
                }
            }
        }

        vec![Tensor::from_vec_device(grad_x, self.in_shape.clone(), grad_output.device.clone()).unwrap()]
    }

    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}
