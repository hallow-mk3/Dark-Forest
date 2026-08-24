//! Binary: gradient checker for all ops.
//! Run: cargo run --bin grad_check

use anyhow::Result;
use darkforest_core::autograd::Value;
use darkforest_core::grad_check::{gradient_check, report};
use darkforest_core::tensor::Tensor;

fn main() -> Result<()> {
    env_logger::init();
    println!("Dark Forest — Gradient Checker");
    println!("================================\n");

    let mut all_pass = true;

    // -----------------------------------------------------------------------
    // Test 1: Addition
    // -----------------------------------------------------------------------
    println!("--- Test: add(a, b) ---");
    let a = Value::leaf(Tensor::from_vec(vec![1.0, -1.0, 2.0], vec![3])?);
    let b = Value::leaf(Tensor::from_vec(vec![0.5, 1.5, -0.5], vec![3])?);

    let results = gradient_check(&[a, b], &["a", "b"], |params| {
        params[0].add(&params[1])?.sum()
    })?;
    all_pass &= report(&results);

    // -----------------------------------------------------------------------
    // Test 2: Matmul
    // -----------------------------------------------------------------------
    println!("--- Test: matmul(a, b) ---");
    let a = Value::leaf(Tensor::randn(vec![3, 4], 0.5));
    let b = Value::leaf(Tensor::randn(vec![4, 3], 0.5));

    let results = gradient_check(&[a, b], &["A [3x4]", "B [4x3]"], |params| {
        params[0].matmul(&params[1])?.sum()
    })?;
    all_pass &= report(&results);

    // -----------------------------------------------------------------------
    // Test 3: Softmax
    // -----------------------------------------------------------------------
    println!("--- Test: softmax(x) ---");
    let x = Value::leaf(Tensor::randn(vec![2, 4], 1.0));

    let results = gradient_check(&[x], &["x [2x4]"], |params| params[0].softmax()?.sum())?;
    all_pass &= report(&results);

    // -----------------------------------------------------------------------
    // Test 4: LayerNorm
    // -----------------------------------------------------------------------
    println!("--- Test: layernorm(x, gamma, beta) ---");
    let x = Value::leaf(Tensor::randn(vec![2, 4], 1.0));
    let gamma = Value::leaf(Tensor::ones(vec![4]));
    let beta = Value::leaf(Tensor::zeros(vec![4]));

    let results = gradient_check(&[x, gamma, beta], &["x", "gamma", "beta"], |params| {
        params[0].layernorm(&params[1], &params[2])?.sum()
    })?;
    all_pass &= report(&results);

    // -----------------------------------------------------------------------
    // Test 5: GELU
    // -----------------------------------------------------------------------
    println!("--- Test: gelu(x) ---");
    let x = Value::leaf(Tensor::randn(vec![5], 1.0));

    let results = gradient_check(&[x], &["x [5]"], |params| params[0].gelu()?.sum())?;
    all_pass &= report(&results);

    // -----------------------------------------------------------------------
    // Test 6: Cross-entropy loss
    // -----------------------------------------------------------------------
    println!("--- Test: cross_entropy_loss(logits, targets) ---");
    let logits = Value::leaf(Tensor::randn(vec![3, 5], 1.0));
    let targets = vec![0usize, 2, 4];

    let results = gradient_check(&[logits], &["logits [3x5]"], |params| {
        params[0].cross_entropy_loss(&targets)
    })?;
    all_pass &= report(&results);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("============================");
    if all_pass {
        println!("✅ ALL CHECKS PASSED — SAFE TO PROCEED TO GPU MILESTONE");
        std::process::exit(0);
    } else {
        println!("❌ FAILURES DETECTED — DO NOT PROCEED TO GPU");
        std::process::exit(1);
    }
}
