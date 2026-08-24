//! Numerical gradient checker.
//!
//! Compares analytical gradients (from autograd) against numerical gradients
//! computed via finite differences: (f(x+δ) - f(x-δ)) / (2δ).
//!
//! This is a hard gate before proceeding to GPU ops.
//! Usage: `cargo run --bin grad_check`

use crate::autograd::Value;
use crate::tensor::Tensor;
use anyhow::Result;

pub const DELTA: f32 = 1e-4;
pub const ATOL: f32 = 1e-2; // absolute tolerance (float is lossy)
pub const RTOL: f32 = 1e-2; // relative tolerance

/// Result of gradient check for one tensor.
#[derive(Debug)]
pub struct GradCheckResult {
    pub param_name: String,
    pub max_abs_err: f32,
    pub max_rel_err: f32,
    pub passed: bool,
}

impl std::fmt::Display for GradCheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "✅ PASS" } else { "❌ FAIL" };
        write!(
            f,
            "{status} [{:30}] max_abs_err={:.2e}  max_rel_err={:.2e}",
            self.param_name, self.max_abs_err, self.max_rel_err
        )
    }
}

/// Check gradients for a function `f: &[Value] → Value (scalar)`.
///
/// `params`:   leaf Values whose gradients to check.
/// `f`:        computes scalar loss from params.
/// `names`:    display names for each param.
pub fn gradient_check<F>(params: &[Value], names: &[&str], f: F) -> Result<Vec<GradCheckResult>>
where
    F: Fn(&[Value]) -> Result<Value>,
{
    // --- Analytical gradient ---
    let loss = f(params)?;
    loss.backward();
    let analytical: Vec<Vec<f32>> = params.iter().map(|p| p.grad()).collect();

    // --- Numerical gradient ---
    let mut results = Vec::new();

    for (pi, param) in params.iter().enumerate() {
        let n = param.numel();
        let data = param.tensor().to_vec();
        let mut numerical = vec![0.0f32; n];

        for i in 0..n {
            // f(x + δ)
            let mut data_plus = data.clone();
            data_plus[i] += DELTA;
            let t_plus = Tensor::from_vec(data_plus, param.shape())?;
            let v_plus = Value::leaf(t_plus);

            // f(x - δ)
            let mut data_minus = data.clone();
            data_minus[i] -= DELTA;
            let t_minus = Tensor::from_vec(data_minus, param.shape())?;
            let v_minus = Value::leaf(t_minus);

            // Build param list with i-th param perturbed
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            params_plus[pi] = v_plus;
            params_minus[pi] = v_minus;

            let loss_plus = f(&params_plus)?.tensor().get(0);
            let loss_minus = f(&params_minus)?.tensor().get(0);

            numerical[i] = (loss_plus - loss_minus) / (2.0 * DELTA);
        }

        // Compare
        let analytical_i = &analytical[pi];
        let mut max_abs_err = 0.0f32;
        let mut max_rel_err = 0.0f32;

        for (a, nu) in analytical_i.iter().zip(numerical.iter()) {
            let abs_err = (a - nu).abs();
            let rel_err = abs_err / (a.abs().max(nu.abs()).max(1e-8));
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);
        }

        let passed = max_abs_err < ATOL || max_rel_err < RTOL;
        results.push(GradCheckResult {
            param_name: names[pi].to_string(),
            max_abs_err,
            max_rel_err,
            passed,
        });
    }

    Ok(results)
}

/// Print a summary and return true if all passed.
pub fn report(results: &[GradCheckResult]) -> bool {
    println!("\n=== Gradient Check Report ===");
    let mut all_pass = true;
    for r in results {
        println!("{r}");
        if !r.passed {
            all_pass = false;
        }
    }
    if all_pass {
        println!("\n✅ ALL GRADIENT CHECKS PASSED — safe to proceed to GPU\n");
    } else {
        println!("\n❌ GRADIENT CHECK FAILURES DETECTED — DO NOT proceed to GPU\n");
    }
    all_pass
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Value;

    #[test]
    fn gradient_check_covers_add_and_matmul() {
        let a = Value::leaf(Tensor::from_vec(vec![1.0, -1.0, 2.0], vec![3]).unwrap());
        let b = Value::leaf(Tensor::from_vec(vec![0.5, 1.5, -0.5], vec![3]).unwrap());
        let add_results = gradient_check(&[a, b], &["a", "b"], |params| {
            params[0].add(&params[1])?.sum()
        })
        .unwrap();

        assert!(add_results.iter().all(|result| result.passed));

        let a = Value::leaf(
            Tensor::from_vec(vec![0.2, -0.4, 0.7, 0.3, -0.1, 0.8], vec![2, 3]).unwrap(),
        );
        let b = Value::leaf(
            Tensor::from_vec(vec![0.5, -0.2, 0.1, 0.6, -0.3, 0.4], vec![3, 2]).unwrap(),
        );
        let matmul_results = gradient_check(&[a, b], &["a", "b"], |params| {
            params[0].matmul(&params[1])?.sum()
        })
        .unwrap();

        assert!(matmul_results.iter().all(|result| result.passed));
    }

    #[test]
    fn gradient_check_covers_activation_and_loss_ops() {
        let x = Value::leaf(
            Tensor::from_vec(vec![-1.2, -0.3, 0.4, 1.1, 2.0, -0.7, 0.8, 1.5], vec![2, 4]).unwrap(),
        );
        let softmax_results =
            gradient_check(&[x], &["x"], |params| params[0].softmax()?.sum()).unwrap();
        assert!(softmax_results.iter().all(|result| result.passed));

        let x = Value::leaf(
            Tensor::from_vec(vec![-1.2, -0.3, 0.4, 1.1, 2.0, -0.7, 0.8, 1.5], vec![2, 4]).unwrap(),
        );
        let gamma = Value::leaf(Tensor::ones(vec![4]));
        let beta = Value::leaf(Tensor::zeros(vec![4]));
        let layernorm_results =
            gradient_check(&[x, gamma, beta], &["x", "gamma", "beta"], |params| {
                params[0].layernorm(&params[1], &params[2])?.sum()
            })
            .unwrap();
        assert!(layernorm_results.iter().all(|result| result.passed));

        let x = Value::leaf(Tensor::from_vec(vec![-1.2, -0.3, 0.4, 1.1, 2.0], vec![5]).unwrap());
        let gelu_results = gradient_check(&[x], &["x"], |params| params[0].gelu()?.sum()).unwrap();
        assert!(gelu_results.iter().all(|result| result.passed));

        let logits = Value::leaf(
            Tensor::from_vec(
                vec![
                    1.2, -0.3, 0.4, 1.1, 2.0, -1.0, 0.7, 1.8, -0.2, 0.5, 0.1, 1.4, -0.6, 0.9, 2.2,
                ],
                vec![3, 5],
            )
            .unwrap(),
        );
        let targets = vec![0usize, 2, 4];
        let loss_results = gradient_check(&[logits], &["logits"], |params| {
            params[0].cross_entropy_loss(&targets)
        })
        .unwrap();
        assert!(loss_results.iter().all(|result| result.passed));
    }
}
