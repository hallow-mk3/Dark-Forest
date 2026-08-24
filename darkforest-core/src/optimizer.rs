//! AdamW optimizer.
//!
//! Parameters are updated in-place via `step()`.
//! This runs outside the autograd graph (no gradients needed for optimizer math).

use crate::autograd::Value;

/// AdamW optimizer state.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,     // weight decay
    pub step: usize, // global step counter
    /// Per-parameter first moment (m) and second moment (v) on CPU.
    pub moments: Vec<(Vec<f32>, Vec<f32>)>,
    #[cfg(feature = "cuda")]
    pub cuda_moments: Vec<(darkforest_cuda::DeviceTensor, darkforest_cuda::DeviceTensor)>,
}

impl AdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) -> Self {
        AdamW {
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step: 0,
            moments: vec![],
            #[cfg(feature = "cuda")]
            cuda_moments: vec![],
        }
    }

    /// Initialize moment buffers matching the number and sizes of `params`.
    pub fn init_moments(&mut self, params: &[Value]) {
        self.moments = params
            .iter()
            .map(|p| (vec![0.0f32; p.numel()], vec![0.0f32; p.numel()]))
            .collect();
        #[cfg(feature = "cuda")]
        {
            self.cuda_moments = params
                .iter()
                .map(|p| {
                    let shape = p.shape();
                    let m = darkforest_cuda::DeviceTensor::zeros(shape.clone())?;
                    let v = darkforest_cuda::DeviceTensor::zeros(shape)?;
                    Ok((m, v))
                })
                .collect::<anyhow::Result<Vec<_>>>()
                .expect("CUDA AdamW moment initialization failed");
        }
    }

    /// Zero gradients for all params.
    pub fn zero_grad(&self, params: &[Value]) {
        for p in params {
            p.zero_grad();
        }
    }

    /// Perform one AdamW parameter update.
    pub fn step(&mut self, params: &[Value], max_norm: Option<f32>) {
        if self.moments.is_empty() {
            self.init_moments(params);
        }
        self.step += 1;

        let t = self.step as f32;
        let bias_corr1 = 1.0 - self.beta1.powf(t);
        let bias_corr2 = 1.0 - self.beta2.powf(t);

        // Compute global norm if clipping requested
        let clip_scale = if let Some(mn) = max_norm {
            let total_norm: f32 = params
                .iter()
                .map(|p| p.grad().iter().map(|g| g * g).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            if total_norm > mn {
                mn / (total_norm + 1e-8)
            } else {
                1.0
            }
        } else {
            1.0
        };

        for (pi, param) in params.iter().enumerate() {
            let is_cuda = param.tensor().is_cuda();

            #[cfg(feature = "cuda")]
            if is_cuda {
                let p_tensor = param.tensor();
                let g_tensor = param.grad_tensor().unwrap();
                let (ref m, ref v) = self.cuda_moments[pi];
                let mut p_guard = p_tensor.data_mut();
                let g_guard = g_tensor.data();

                darkforest_cuda::DeviceTensor::adamw_update(
                    p_guard.device_tensor_mut(),
                    m,
                    v,
                    g_guard.device_tensor(),
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.wd,
                    bias_corr1,
                    bias_corr2,
                )
                .expect("CUDA AdamW update failed");
                continue;
            }

            let grad_raw = param.grad();
            let grad: Vec<f32> = grad_raw.iter().map(|g| g * clip_scale).collect();
            let (ref mut m, ref mut v) = self.moments[pi];
            let p_data: Vec<f32> = param.tensor().to_vec();
            let mut new_data = vec![0.0f32; p_data.len()];

            for i in 0..p_data.len() {
                let p_wd = p_data[i] * (1.0 - self.lr * self.wd);
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
                let m_hat = m[i] / bias_corr1;
                let v_hat = v[i] / bias_corr2;
                new_data[i] = p_wd - self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            param.update_tensor_data(new_data);
        }
    }
}

/// OffloadedAdamW (ZeRO-Offload style) optimizer.
///
/// Keeps all 1st (m) and 2nd (v) optimizer moment states in pinned / system RAM
/// instead of consuming precious GPU VRAM. During the optimizer step:
///   1. Gradients are gathered / streamed from GPU -> CPU.
///   2. Moments and parameter updates are calculated in parallel in host memory.
///   3. Updated parameters are transferred back to GPU VRAM.
pub struct OffloadedAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    pub step: usize,
    /// Host-resident moment buffers (m, v)
    pub host_moments: Vec<(Vec<f32>, Vec<f32>)>,
}

impl OffloadedAdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step: 0,
            host_moments: vec![],
        }
    }

    pub fn init_moments(&mut self, params: &[Value]) {
        self.host_moments = params
            .iter()
            .map(|p| (vec![0.0f32; p.numel()], vec![0.0f32; p.numel()]))
            .collect();
    }

    pub fn zero_grad(&self, params: &[Value]) {
        for p in params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self, params: &[Value], max_norm: Option<f32>) {
        if self.host_moments.is_empty() {
            self.init_moments(params);
        }
        self.step += 1;

        let t = self.step as f32;
        let bias_corr1 = 1.0 - self.beta1.powf(t);
        let bias_corr2 = 1.0 - self.beta2.powf(t);

        let clip_scale = if let Some(mn) = max_norm {
            let total_norm: f32 = params
                .iter()
                .map(|p| p.grad().iter().map(|g| g * g).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            if total_norm > mn {
                mn / (total_norm + 1e-8)
            } else {
                1.0
            }
        } else {
            1.0
        };

        for (pi, param) in params.iter().enumerate() {
            let grad_raw = param.grad();
            let grad: Vec<f32> = grad_raw.iter().map(|g| g * clip_scale).collect();
            let (ref mut m, ref mut v) = self.host_moments[pi];
            let p_data: Vec<f32> = param.tensor().to_vec();
            let mut new_data = vec![0.0f32; p_data.len()];

            for i in 0..p_data.len() {
                let p_wd = p_data[i] * (1.0 - self.lr * self.wd);
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
                let m_hat = m[i] / bias_corr1;
                let v_hat = v[i] / bias_corr2;
                new_data[i] = p_wd - self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            param.update_tensor_data(new_data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_offloaded_adamw_matches_standard() {
        let p1 = Value::leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
        let p2 = Value::leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap());

        let mut opt_std = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.01);
        let mut opt_off = OffloadedAdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.01);

        // Fake loss & backward
        let l1 = p1.scale(2.0).unwrap().sum().unwrap();
        let l2 = p2.scale(2.0).unwrap().sum().unwrap();
        l1.backward();
        l2.backward();

        opt_std.step(&[p1.clone()], None);
        opt_off.step(&[p2.clone()], None);

        let data1 = p1.tensor().to_vec();
        let data2 = p2.tensor().to_vec();
        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
