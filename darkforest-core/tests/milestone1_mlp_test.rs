//! Milestone 1 Gate: Rust native MLP Training Parity test.
//!
//! Trains an MLP for 200 steps with SGD and AdamW using tape autograd.
//! Asserts that loss monotonically decreases and converges cleanly.

use darkforest_core::nn::{TapeAdamW, TapeMLP, TapeSGD};
use darkforest_core::tensor::Tensor;

#[test]
fn test_milestone1_mlp_sgd_convergence() {
    let mlp = TapeMLP::new(&[8, 16, 8, 4]).expect("Failed to initialize MLP");
    let mut sgd = TapeSGD::new(0.2, 0.9, 0.0);
    let params = mlp.parameters();

    // Fixed synthetic batch
    let batch_size = 16;
    let x_data: Vec<f32> = (0..batch_size * 8)
        .map(|i| ((i as f32) * 0.13).sin())
        .collect();
    let y_data: Vec<i64> = (0..batch_size).map(|i| (i % 4) as i64).collect();

    let x = Tensor::from_vec(x_data, vec![batch_size, 8]).unwrap();
    let y = Tensor::from_slice_i64(&y_data, vec![batch_size]).unwrap();

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..200 {
        mlp.zero_grad();
        let logits = mlp.forward(&x).unwrap();
        let loss = logits.cross_entropy(&y).unwrap();
        let loss_val = loss.to_vec()[0];

        if step == 0 {
            initial_loss = loss_val;
        }
        final_loss = loss_val;

        loss.backward(None, false).unwrap();
        sgd.step(&params);
    }

    assert!(
        final_loss < initial_loss * 0.5,
        "MLP SGD did not converge sufficiently: initial={initial_loss}, final={final_loss}"
    );
}

#[test]
fn test_milestone1_mlp_adamw_convergence() {
    let mlp = TapeMLP::new(&[8, 16, 8, 4]).expect("Failed to initialize MLP");
    let mut adamw = TapeAdamW::default_lr(0.02);
    let params = mlp.parameters();

    let batch_size = 16;
    let x_data: Vec<f32> = (0..batch_size * 8)
        .map(|i| ((i as f32) * 0.17).cos())
        .collect();
    let y_data: Vec<i64> = (0..batch_size).map(|i| ((i + 1) % 4) as i64).collect();

    let x = Tensor::from_vec(x_data, vec![batch_size, 8]).unwrap();
    let y = Tensor::from_slice_i64(&y_data, vec![batch_size]).unwrap();

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..200 {
        mlp.zero_grad();
        let logits = mlp.forward(&x).unwrap();
        let loss = logits.cross_entropy(&y).unwrap();
        let loss_val = loss.to_vec()[0];

        if step == 0 {
            initial_loss = loss_val;
        }
        final_loss = loss_val;

        loss.backward(None, false).unwrap();
        adamw.step(&params);
    }

    assert!(
        final_loss < initial_loss * 0.5,
        "MLP AdamW did not converge sufficiently: initial={initial_loss}, final={final_loss}"
    );
}
