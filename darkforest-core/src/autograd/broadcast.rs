use anyhow::Result;
use crate::tensor::Tensor;

/// Sum a gradient tensor back to its target unbroadcasted shape.
///
/// In reverse-mode autodiff, when an input is broadcasted to a larger output shape,
/// the upstream gradient must be summed along all broadcasted axes.
pub fn sum_to_shape(grad: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    if grad.shape() == target_shape {
        return Ok(grad.clone());
    }

    if target_shape.is_empty() {
        return grad.sum(None, false);
    }

    let grad_ndim = grad.ndim();
    let target_ndim = target_shape.len();

    let mut result = grad.clone();

    // 1. If grad has more dimensions than target, sum out leading dimensions.
    if grad_ndim > target_ndim {
        let diff = grad_ndim - target_ndim;
        let leading_dims: Vec<usize> = (0..diff).collect();
        result = result.sum(Some(&leading_dims), false)?;
    }

    // 2. Now ranks match. For dimensions where target_shape[i] == 1 and result.shape()[i] > 1,
    // sum along dimension i with keepdim = true.
    let current_shape = result.shape().to_vec();
    let mut dims_to_sum = Vec::new();
    for i in 0..target_ndim {
        if target_shape[i] == 1 && current_shape[i] > 1 {
            dims_to_sum.push(i);
        }
    }

    if !dims_to_sum.is_empty() {
        result = result.sum(Some(&dims_to_sum), true)?;
    }

    // 3. Final reshape if needed
    if result.shape() != target_shape {
        result = result.reshape(target_shape.to_vec())?;
    }

    Ok(result)
}
