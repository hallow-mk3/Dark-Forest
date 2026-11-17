use darkforest_core::tensor::Tensor;

#[test]
fn test_tape_basic_arithmetic_backward() {
    let x = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3])
        .unwrap()
        .requires_grad_(true);
    let y = Tensor::from_vec(vec![5.0, 6.0, 7.0], vec![3])
        .unwrap()
        .requires_grad_(true);

    // z = x * y + x
    let mul = x.mul(&y).unwrap();
    let z = mul.add(&x).unwrap();

    // loss = sum(z)
    let loss = z.sum(None, false).unwrap();
    loss.backward(None, false).unwrap();

    let grad_x = x.grad().unwrap().to_vec();
    let grad_y = y.grad().unwrap().to_vec();

    // dz/dx = y + 1 => [6, 7, 8]
    // dz/dy = x => [2, 3, 4]
    assert_eq!(grad_x, vec![6.0, 7.0, 8.0]);
    assert_eq!(grad_y, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_tape_broadcast_sum_to_size_backward() {
    // a: [3, 1]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1])
        .unwrap()
        .requires_grad_(true);
    // b: [3, 4]
    let b = Tensor::ones(vec![3, 4]).requires_grad_(true);

    // c = a + b -> shape [3, 4]
    let c = a.add(&b).unwrap();
    let loss = c.sum(None, false).unwrap();
    loss.backward(None, false).unwrap();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    assert_eq!(grad_a.shape(), &vec![3, 1]);
    assert_eq!(grad_a.to_vec(), vec![4.0, 4.0, 4.0]); // Each row summed 4 times

    assert_eq!(grad_b.shape(), &vec![3, 4]);
    assert_eq!(grad_b.to_vec(), vec![1.0; 12]);
}

#[test]
fn test_tape_dag_multi_consumer() {
    let x = Tensor::from_vec(vec![3.0], vec![1])
        .unwrap()
        .requires_grad_(true);

    // y = x * x + x * x
    let x2 = x.mul(&x).unwrap();
    let y = x2.add(&x2).unwrap();

    y.backward(None, false).unwrap();

    // dy/dx = 2*(2*x) = 4 * 3 = 12
    let grad_x = x.grad().unwrap().to_vec();
    assert_eq!(grad_x, vec![12.0]);
}

#[test]
fn test_tape_in_place_mutation_fails_backward() {
    let mut x = Tensor::from_vec(vec![2.0, 3.0], vec![2])
        .unwrap()
        .requires_grad_(true);
    let y = Tensor::from_vec(vec![4.0, 5.0], vec![2]).unwrap();

    let z = x.mul(&y).unwrap();

    // In-place modify x after it was saved on the tape
    let ones = Tensor::ones(vec![2]);
    x.add_(&ones).unwrap();

    // Backward must catch version mismatch and error
    let res = z.sum(None, false).unwrap().backward(None, false);
    assert!(res.is_err());
    let err_msg = res.unwrap_err().to_string();
    assert!(
        err_msg.contains("one of the variables needed for gradient computation has been modified"),
        "Unexpected error: {}",
        err_msg
    );
}

#[test]
fn test_tape_matmul_and_relu_backward() {
    // x: [2, 3]
    let x = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], vec![2, 3])
        .unwrap()
        .requires_grad_(true);
    // w: [3, 2]
    let w = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
        .unwrap()
        .requires_grad_(true);

    let mm = x.matmul(&w).unwrap(); // [2, 2]
    let relu_out = mm.relu().unwrap();
    let loss = relu_out.sum(None, false).unwrap();

    loss.backward(None, false).unwrap();

    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
    assert_eq!(x.grad().unwrap().shape(), &vec![2, 3]);
    assert_eq!(w.grad().unwrap().shape(), &vec![3, 2]);
}

#[test]
fn test_tape_cross_entropy_backward() {
    // logits: [2, 3]
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.1, 0.5, 3.0, 0.2], vec![2, 3])
        .unwrap()
        .requires_grad_(true);
    // targets: [0, 1]
    let targets = Tensor::from_slice_i64(&[0, 1], vec![2]).unwrap();

    let loss = logits.cross_entropy(&targets).unwrap();
    loss.backward(None, false).unwrap();

    let grad_logits = logits.grad().unwrap();
    assert_eq!(grad_logits.shape(), &vec![2, 3]);

    // Sum of gradients along class dimension should be ~0 (since sum(p_i - 1_{y=i}) = 1 - 1 = 0)
    let g = grad_logits.to_vec();
    let sum_row0 = g[0] + g[1] + g[2];
    let sum_row1 = g[3] + g[4] + g[5];
    assert!((sum_row0.abs()) < 1e-5, "sum_row0 = {}", sum_row0);
    assert!((sum_row1.abs()) < 1e-5, "sum_row1 = {}", sum_row1);
}
