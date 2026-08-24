//! Python bindings for the Dark Forest runtime.
//!
//! The product goal is not to clone every PyTorch feature blindly, but to ship
//! a smaller, faster, more opinionated ML engine for the workloads we actually
//! need: tight CUDA control, explicit device residency, deterministic training,
//! and a predictable Rust core.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::collections::HashMap;

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

#[cfg(feature = "python")]
#[pyfunction]
fn product_name() -> PyResult<String> {
    Ok("Dark Forest".to_string())
}

#[cfg(feature = "python")]
#[pyfunction]
fn status() -> PyResult<String> {
    Ok("alpha-ready: Rust autograd core, CUDA fallback path, Python bindings scaffolded".to_string())
}

#[cfg(feature = "python")]
#[pyfunction]
fn smoke_test() -> PyResult<String> {
    use darkforest_core::autograd::Value;
    use darkforest_core::tensor::Tensor;

    let a = Value::leaf(
        Tensor::from_vec(vec![1.0, -1.0, 2.0], vec![3])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("tensor init failed: {e}")))?,
    );
    let b = Value::leaf(
        Tensor::from_vec(vec![0.5, 1.5, -0.5], vec![3])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("tensor init failed: {e}")))?,
    );

    let loss = a
        .add(&b)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("op failed: {e}")))?
        .sum()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("sum failed: {e}")))?;
    loss.backward();

    let grad_a = a.grad();
    let grad_b = b.grad();
    let loss_val = loss.tensor().get(0);

    Ok(format!(
        "loss={loss_val:.6}, grad_a={grad_a:?}, grad_b={grad_b:?}"
    ))
}

#[cfg(feature = "python")]
#[pyfunction]
fn benchmark_matrix_step(steps: usize, rows: usize, cols: usize) -> PyResult<HashMap<String, f64>> {
    use darkforest_core::autograd::Value;
    use darkforest_core::tensor::Tensor;
    use std::time::Instant;

    if rows == 0 || cols == 0 || steps == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "steps, rows, and cols must all be > 0",
        ));
    }

    let device = {
        #[cfg(feature = "cuda")]
        {
            darkforest_core::tensor::Device::Cuda(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            darkforest_core::tensor::Device::Cpu
        }
    };

    let a_vec: Vec<f32> = (0..rows * cols)
        .map(|idx| ((idx as f32) + 1.0) % 7.0 - 3.0)
        .collect();
    let b_vec: Vec<f32> = (0..cols * rows)
        .map(|idx| (((idx as f32) + 2.0) % 9.0) - 4.0)
        .collect();

    let a_tensor = Tensor::from_vec_device(a_vec, vec![rows, cols], device.clone()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("a tensor init failed: {e}"))
    })?;
    let b_tensor = Tensor::from_vec_device(b_vec, vec![cols, rows], device.clone()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("b tensor init failed: {e}"))
    })?;

    // Warmup (5 iterations)
    for _ in 0..5 {
        let a = Value::leaf(a_tensor.clone());
        let b = Value::leaf(b_tensor.clone());
        let out = a.matmul(&b).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("matmul failed: {e}"))
        })?;
        let loss = out.sum().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("sum failed: {e}"))
        })?;
        loss.backward();
        let _ = darkforest_core::cuda_sync();
    }

    #[cfg(feature = "cuda")]
    let timer = darkforest_core::CudaEventTimer::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CudaEventTimer init failed: {e}"))
    })?;

    let mut samples = Vec::with_capacity(steps);
    for _ in 0..steps {
        let a = Value::leaf(a_tensor.clone());
        let b = Value::leaf(b_tensor.clone());

        #[cfg(feature = "cuda")]
        timer.start().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("timer start failed: {e}"))
        })?;
        #[cfg(not(feature = "cuda"))]
        let start = Instant::now();

        let out = a.matmul(&b).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("matmul failed: {e}"))
        })?;
        let loss = out.sum().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("sum failed: {e}"))
        })?;
        loss.backward();

        #[cfg(feature = "cuda")]
        let elapsed_ms = timer.stop_and_elapsed_ms().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("timer stop failed: {e}"))
        })?;
        #[cfg(not(feature = "cuda"))]
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        samples.push(elapsed_ms);
    }

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    let mut result = HashMap::new();
    result.insert("steps".to_string(), steps as f64);
    result.insert("rows".to_string(), rows as f64);
    result.insert("cols".to_string(), cols as f64);
    result.insert("mean_ms".to_string(), mean);
    result.insert("median_ms".to_string(), median);
    result.insert("min_ms".to_string(), samples.iter().copied().fold(f64::INFINITY, f64::min));
    result.insert("max_ms".to_string(), samples.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    Ok(result)
}

#[cfg(feature = "python")]
#[pymodule]
fn _darkforest_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(product_name, m)?)?;
    m.add_function(wrap_pyfunction!(status, m)?)?;
    m.add_function(wrap_pyfunction!(smoke_test, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_matrix_step, m)?)?;
    Ok(())
}
