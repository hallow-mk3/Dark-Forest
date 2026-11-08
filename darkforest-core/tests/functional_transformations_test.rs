//! Integration Tests: Dark Forest Functional Transformations Suite (JAX & beyond)

use darkforest_core::functional::*;
use darkforest_core::autograd::Value;
use darkforest_core::Tensor;

#[test]
fn test_jnp_and_immutable_at_indexing() {
    // 1. Array creation via jnp
    let x = jnp::arange(0.0, 6.0, 1.0).unwrap();
    assert_eq!(x.shape(), &[6]);
    assert_eq!(x.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // 2. Reshape to 2D
    let mat = x.reshape(vec![2, 3]).unwrap();
    assert_eq!(mat.shape(), &[2, 3]);

    // 3. Functional update: mat.at[1, 2].set(99.0)
    let updated = mat.at(IndexSpec::Point(vec![1, 2])).set(99.0).unwrap();

    // Verify immutability of original mat
    assert_eq!(mat.to_vec()[5], 5.0);
    // Verify updated copy
    assert_eq!(updated.to_vec()[5], 99.0);

    // 4. In-place arithmetic: mat.at[0..2].add(10.0)
    let sliced_add = mat.at(IndexSpec::Slice { axis: 1, start: 0, end: 2 }).add(10.0).unwrap();
    assert_eq!(sliced_add.to_vec(), vec![10.0, 11.0, 2.0, 13.0, 14.0, 5.0]);
}

#[test]
fn test_stateless_prng_key_parallel_splits() {
    let root_key = Key::new(2026);
    let subkeys = root_key.split_n(4);
    assert_eq!(subkeys.len(), 4);

    // Verify all keys are statistically distinct
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(subkeys[i], subkeys[j]);
        }
    }

    // Deterministic random sampling
    let sample1 = uniform(&subkeys[0], vec![100], 0.0, 1.0);
    let sample2 = uniform(&subkeys[0], vec![100], 0.0, 1.0);
    // Same key => exactly identical arrays
    assert_eq!(sample1.to_vec(), sample2.to_vec());

    // Different key => different arrays
    let sample3 = uniform(&subkeys[1], vec![100], 0.0, 1.0);
    assert_ne!(sample1.to_vec(), sample3.to_vec());
}

#[test]
fn test_vmap_over_grad_composition() {
    // Compute gradient of f(x) = (x - 2.0)²
    // f'(x) = 2*(x - 2.0)
    let loss = |v: &Value| {
        let two = Value::leaf(Tensor::from_vec(vec![2.0], vec![1])?);
        let diff = v.sub(&two)?;
        diff.mul(&diff)?.sum()
    };

    let sample_grad = grad(loss);

    // Wrap sample_grad in vmap to evaluate batched gradient across 4 samples simultaneously!
    let batched_grad_fn = vmap(move |x: &Array| sample_grad(x));

    // Inputs: [0.0, 1.0, 2.0, 3.0]
    // Expected grads: 2*(x - 2) => [-4.0, -2.0, 0.0, 2.0]
    let batch_input = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let batch_grads = batched_grad_fn(&batch_input).unwrap();

    assert_eq!(batch_grads.shape(), &[4, 1]);
    let grads_flat = batch_grads.to_vec();
    assert!((grads_flat[0] - (-4.0)).abs() < 1e-4);
    assert!((grads_flat[1] - (-2.0)).abs() < 1e-4);
    assert!((grads_flat[2] - 0.0).abs() < 1e-4);
    assert!((grads_flat[3] - 2.0).abs() < 1e-4);
}

#[test]
fn test_jit_expression_fusion_and_memory_reuse() {
    let mut graph = IrGraph::new();
    let a = graph.add_input(vec![3]);
    let b = graph.add_input(vec![3]);
    let c = graph.add_input(vec![3]);

    // Express affine transformation: a * b + c
    let prod = graph.add_op(IrOp::Mul { left: a, right: b }, vec![3]);
    let out = graph.add_op(IrOp::Add { left: prod, right: c }, vec![3]);
    graph.mark_output(out);

    let plan = CompiledPlan::new(graph);

    let in_a = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let in_b = Array::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
    let in_c = Array::from_vec(vec![7.0, 8.0, 9.0], vec![3]).unwrap();

    let res = plan.execute(&[in_a, in_b, in_c]).unwrap();
    // [1*4 + 7, 2*5 + 8, 3*6 + 9] = [11, 18, 27]
    assert_eq!(res.to_vec(), vec![11.0, 18.0, 27.0]);
}

#[test]
fn test_optax_pure_training_loop() {
    // Minimize f(w) = w² starting from w = 5.0
    // Gradients: ∇f(w) = 2*w
    let opt = AdamWOpt::new(0.15, 0.9, 0.999, 1e-8, 0.0);
    let mut params = vec![Array::from_vec(vec![3.0], vec![1]).unwrap()];
    let mut state = opt.init(&params);

    for _ in 0..30 {
        let w_val = params[0].item().unwrap();
        let grads = vec![Array::from_vec(vec![2.0 * w_val], vec![1]).unwrap()];
        let (updates, next_state) = opt.update(&grads, &state, &params).unwrap();
        params = opt.apply_updates(&params, &updates).unwrap();
        state = next_state;
    }

    let final_w = params[0].item().unwrap();
    assert!(final_w < 0.5, "Optimizer should converge towards 0, got {}", final_w);
}

#[test]
fn test_nan_trapping_in_debug_monitor() {
    let clean_tensor = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert!(debug_check_nans(&clean_tensor, "safe_layer").is_ok());

    let poisoned_tensor = Array::from_vec(vec![1.0, f32::NAN, 3.0], vec![3]).unwrap();
    let trap_result = debug_check_nans(&poisoned_tensor, "exploding_layer");
    assert!(trap_result.is_err());
    let err_msg = trap_result.unwrap_err().to_string();
    assert!(err_msg.contains("NaN/Inf TRAP"));
    assert!(err_msg.contains("exploding_layer"));
}
