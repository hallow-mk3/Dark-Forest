//! Dark Forest ML Framework — Core Library
//!
//! Rust-based autograd engine with CPU and CUDA backends.
//! Architecture: Tensor → Ops → Autograd Graph → Optimizer

pub mod autograd;
pub mod data;
pub mod engine;
pub mod grad_check;
pub mod nn;
pub mod ops;
pub mod optimizer;
pub mod scheduler;
pub mod tensor;

pub use autograd::no_grad;
pub use data::{DataLoader, Dataset, TensorDataset};
pub use engine::StaticGPT2;
pub use scheduler::{CosineAnnealingLR, ExponentialLR, LRScheduler, StepLR};
pub use tensor::{DType, Device, Shape, Tensor};


pub fn cuda_sync() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    {
        darkforest_cuda::kernels::cuda_sync()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub type CudaEventTimer = darkforest_cuda::CudaEventTimer;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::{autograd::Value, Tensor};

    fn objective(x: &[f32], w: &[f32], b: &[f32]) -> f32 {
        let x = Value::leaf(Tensor::from_vec(x.to_vec(), vec![2, 2]).unwrap());
        let w = Value::leaf(Tensor::from_vec(w.to_vec(), vec![3, 2]).unwrap());
        let b = Value::leaf(Tensor::from_vec(b.to_vec(), vec![3]).unwrap());
        x.cuda_linear(&w, Some(&b), 2, 3)
            .unwrap()
            .sum()
            .unwrap()
            .tensor()
            .get(0)
    }

    #[test]
    fn cuda_linear_gradients_match_finite_difference() {
        let x_data = vec![0.2, -0.4, 0.7, 0.3];
        let w_data = vec![0.5, -0.2, 0.1, 0.6, -0.3, 0.4];
        let b_data = vec![0.1, -0.2, 0.3];
        let x = Value::leaf(Tensor::from_vec(x_data.clone(), vec![2, 2]).unwrap());
        let w = Value::leaf(Tensor::from_vec(w_data.clone(), vec![3, 2]).unwrap());
        let b = Value::leaf(Tensor::from_vec(b_data.clone(), vec![3]).unwrap());
        let loss = x.cuda_linear(&w, Some(&b), 2, 3).unwrap().sum().unwrap();
        loss.backward();

        let delta = 1e-3;
        for (data, gradients) in [
            (&x_data, x.grad()),
            (&w_data, w.grad()),
            (&b_data, b.grad()),
        ] {
            for index in 0..data.len() {
                let mut plus = data.clone();
                let mut minus = data.clone();
                plus[index] += delta;
                minus[index] -= delta;
                let numerical = if std::ptr::eq(data, &x_data) {
                    (objective(&plus, &w_data, &b_data) - objective(&minus, &w_data, &b_data))
                        / (2.0 * delta)
                } else if std::ptr::eq(data, &w_data) {
                    (objective(&x_data, &plus, &b_data) - objective(&x_data, &minus, &b_data))
                        / (2.0 * delta)
                } else {
                    (objective(&x_data, &w_data, &plus) - objective(&x_data, &w_data, &minus))
                        / (2.0 * delta)
                };
                assert!(
                    (gradients[index] - numerical).abs() <= 2e-3,
                    "index={index}, analytical={}, numerical={numerical}",
                    gradients[index]
                );
            }
        }
    }

    #[test]
    fn cuda_matmul_backward_stays_on_device() {
        let device = super::Device::Cuda(0);
        let a = Value::leaf(
            Tensor::from_vec_device(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2, 3],
                device.clone(),
            )
            .unwrap(),
        );
        let b = Value::leaf(
            Tensor::from_vec_device(vec![0.5, -0.2, 0.1, 0.6, -0.3, 0.4], vec![3, 2], device)
                .unwrap(),
        );
        let loss = a.matmul(&b).unwrap().sum().unwrap();
        loss.backward();

        assert_eq!(a.tensor().device, super::Device::Cuda(0));
        assert_eq!(b.tensor().device, super::Device::Cuda(0));
        assert_eq!(a.grad_tensor().unwrap().device, super::Device::Cuda(0));
        assert_eq!(b.grad_tensor().unwrap().device, super::Device::Cuda(0));

        let grad_a = a.grad();
        let grad_b = b.grad();
        for (actual, expected) in grad_a.iter().zip([0.3, 0.7, 0.1, 0.3, 0.7, 0.1]) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "actual={actual}, expected={expected}"
            );
        }
        for (actual, expected) in grad_b.iter().zip([5.0, 5.0, 7.0, 7.0, 9.0, 9.0]) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn mixed_device_binary_ops_are_rejected() {
        let cpu = Value::leaf(Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap());
        let cuda = Value::leaf(
            Tensor::from_vec_device(vec![3.0, 4.0], vec![2], super::Device::Cuda(0)).unwrap(),
        );

        assert!(cpu.add(&cuda).is_err());
        assert!(cpu.mul(&cuda).is_err());
        assert!(cpu.matmul(&cuda).is_err());
    }

    fn cuda_cross_entropy_objective(logits: &[f32], targets: &[usize]) -> f32 {
        let logits = Value::leaf(
            Tensor::from_vec_device(logits.to_vec(), vec![2, 3], super::Device::Cuda(0)).unwrap(),
        );
        logits.cross_entropy_loss(targets).unwrap().tensor().get(0)
    }

    #[test]
    fn cuda_cross_entropy_matches_cpu_and_finite_difference() {
        let logits_data = vec![1.2, -0.3, 0.4, -1.0, 0.7, 1.8];
        let targets = vec![0usize, 2];
        let logits = Value::leaf(
            Tensor::from_vec_device(logits_data.clone(), vec![2, 3], super::Device::Cuda(0))
                .unwrap(),
        );
        let loss = logits.cross_entropy_loss(&targets).unwrap();
        let gpu_loss = loss.tensor().get(0);
        let cpu_loss = {
            let cpu_logits =
                Value::leaf(Tensor::from_vec(logits_data.clone(), vec![2, 3]).unwrap());
            cpu_logits
                .cross_entropy_loss(&targets)
                .unwrap()
                .tensor()
                .get(0)
        };
        assert!(
            (gpu_loss - cpu_loss).abs() <= 1e-5,
            "gpu={gpu_loss}, cpu={cpu_loss}"
        );

        loss.backward();
        assert_eq!(logits.grad_tensor().unwrap().device, super::Device::Cuda(0));
        let analytical = logits.grad();
        let delta = 1e-3;
        for index in 0..logits_data.len() {
            let mut plus = logits_data.clone();
            let mut minus = logits_data.clone();
            plus[index] += delta;
            minus[index] -= delta;
            let numerical = (cuda_cross_entropy_objective(&plus, &targets)
                - cuda_cross_entropy_objective(&minus, &targets))
                / (2.0 * delta);
            assert!(
                (analytical[index] - numerical).abs() <= 2e-3,
                "index={index}, analytical={}, numerical={numerical}",
                analytical[index]
            );
        }
    }

    #[test]
    fn cuda_tiny_transformer_loss_runs_on_device() {
        use crate::nn::transformer::{GPT2Config, GPT2};

        let mut cfg = GPT2Config::tiny();
        cfg.vocab_size = 32;
        cfg.max_seq_len = 8;
        let mut model = GPT2::new(cfg);
        model.to_device(super::Device::Cuda(0)).unwrap();

        let tokens = vec![1usize, 2, 3, 4, 5, 6, 7];
        let loss = model.loss(&tokens).unwrap();
        assert_eq!(loss.tensor().device, super::Device::Cuda(0));
        assert!(loss.tensor().get(0).is_finite());

        loss.backward();
        let grads = model.parameters();
        assert!(grads.iter().all(|p| p.grad_tensor().is_some()));
        assert!(grads.iter().all(|p| p.grad_tensor().unwrap().device == super::Device::Cuda(0)));
    }
}
