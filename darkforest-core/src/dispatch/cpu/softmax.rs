use anyhow::{anyhow, Result};
use crate::tensor::Tensor;

pub fn log_softmax(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    let ndim = tensor.ndim();
    if ndim == 0 {
        return Err(anyhow!("log_softmax requires ndim >= 1"));
    }
    let d = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };
    if d >= ndim {
        return Err(anyhow!("Dimension {} out of range for ndim {}", dim, ndim));
    }

    let out = Tensor::zeros_device(tensor.shape().to_vec(), tensor.device());
    let shape = tensor.shape();
    let class_dim_size = shape[d];
    let total_elements = tensor.numel();
    let num_slices = total_elements / class_dim_size;

    let in_strides = tensor.strides();
    let out_strides = out.strides().to_vec();

    let in_ptr = tensor.data_ptr() as *const f32;
    let out_ptr = out.data_ptr() as *mut f32;

    let in_stride_c = in_strides[d];
    let out_stride_c = out_strides[d];

    // Iterate through all slices along dim `d`
    let mut coord = vec![0usize; ndim];

    for slice_idx in 0..num_slices {
        // Map slice_idx to non-d coordinates
        let mut rem = slice_idx;
        for dim_i in (0..ndim).rev() {
            if dim_i == d {
                coord[dim_i] = 0;
            } else {
                coord[dim_i] = rem % shape[dim_i];
                rem /= shape[dim_i];
            }
        }

        let mut base_in_off = tensor.storage_offset() as isize;
        let mut base_out_off = out.storage_offset() as isize;
        for dim_i in 0..ndim {
            base_in_off += (coord[dim_i] as isize) * in_strides[dim_i];
            base_out_off += (coord[dim_i] as isize) * out_strides[dim_i];
        }

        // 1. Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        unsafe {
            for c in 0..class_dim_size {
                let val = *in_ptr.offset(base_in_off + (c as isize) * in_stride_c);
                if val > max_val {
                    max_val = val;
                }
            }

            // 2. Compute sum(exp(x - max))
            let mut sum_exp = 0.0f32;
            for c in 0..class_dim_size {
                let val = *in_ptr.offset(base_in_off + (c as isize) * in_stride_c);
                sum_exp += (val - max_val).exp();
            }
            let log_sum_exp = max_val + sum_exp.ln();

            // 3. Output log_softmax
            for c in 0..class_dim_size {
                let val = *in_ptr.offset(base_in_off + (c as isize) * in_stride_c);
                *out_ptr.offset(base_out_off + (c as isize) * out_stride_c) = val - log_sum_exp;
            }
        }
    }

    Ok(out)
}

pub fn softmax(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    let ls = log_softmax(tensor, dim)?;
    crate::dispatch::iterator::TensorIterator::unary_f32(&ls, |x| x.exp())
}
