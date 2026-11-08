use darkforest_core::autograd::Value;
use darkforest_core::nn::{BatchNorm2d, Conv2d, Linear, MaxPool2d};
use darkforest_core::tensor::Tensor;

#[test]
fn test_linear_autograd_flow() {
    let linear = Linear::new(4, 3, true);
    let x = Value::leaf(Tensor::ones(vec![2, 4]));
    let y = linear.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward();

    let w_grad = linear.weight.grad();
    assert_eq!(w_grad.len(), 12);
    assert!(w_grad.iter().any(|&g| g.abs() > 0.0), "Linear weight gradient must be non-zero");

    let b_grad = linear.bias.as_ref().unwrap().grad();
    assert_eq!(b_grad.len(), 3);
    for g in b_grad {
        assert_eq!(g, 2.0, "Bias gradient for batch=2 with sum loss must be 2.0");
    }
}

#[test]
fn test_diamond_graph_residual_accumulation() {
    // y = x + Linear(x)
    // loss = sum(y)
    // dy/dx = 1 + W^T
    let linear = Linear::new(3, 3, false);
    // set weights to identity
    linear.weight.update_tensor_data(vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);

    let x = Value::leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let h = linear.forward(&x).unwrap();
    let y = x.add(&h).unwrap();
    let loss = y.sum().unwrap();
    loss.backward();

    let x_grad = x.grad();
    // y = x + x * I = 2x. d(sum(y))/dx should be [2.0, 2.0, 2.0]
    assert_eq!(x_grad.len(), 3);
    for g in x_grad {
        assert!((g - 2.0).abs() < 1e-5, "Expected gradient 2.0, got {g}");
    }
}

#[test]
fn test_conv2d_autograd_flow() {
    let conv = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0), true);
    let x = Value::leaf(Tensor::ones(vec![1, 1, 3, 3]));
    let y = conv.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward();

    let w_grad = conv.weight.grad();
    assert_eq!(w_grad.len(), 4);
    assert!(w_grad.iter().any(|&g| g.abs() > 0.0), "Conv2d weight gradient must be non-zero");

    let x_grad = x.grad();
    assert_eq!(x_grad.len(), 9);
    assert!(x_grad.iter().any(|&g| g.abs() > 0.0), "Conv2d input gradient must be non-zero");
}

#[test]
fn test_batchnorm2d_autograd_flow() {
    let mut bn = BatchNorm2d::new(2);
    let x = Value::leaf(Tensor::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, // ch0
        5.0, 6.0, 7.0, 8.0, // ch1
    ], vec![1, 2, 2, 2]).unwrap());
    let y = bn.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward();

    let g_grad = bn.weight.grad();
    assert_eq!(g_grad.len(), 2);
    let b_grad = bn.bias.grad();
    assert_eq!(b_grad.len(), 2);
    assert!((b_grad[0] - 4.0).abs() < 1e-4);
    assert!((b_grad[1] - 4.0).abs() < 1e-4);
}

#[test]
fn test_maxpool2d_autograd_flow() {
    let pool = MaxPool2d::new((2, 2), (2, 2), (0, 0));
    let x = Value::leaf(Tensor::from_vec(vec![
        1.0, 2.0,
        3.0, 4.0, // max is 4.0 at flat index 3
    ], vec![1, 1, 2, 2]).unwrap());
    let y = pool.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward();

    let x_grad = x.grad();
    assert_eq!(x_grad, vec![0.0, 0.0, 0.0, 1.0], "Gradient should only route to max position");
}
