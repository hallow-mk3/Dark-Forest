use anyhow::{anyhow, Result};
use crate::tensor::{numel, Tensor};

/// Calculate common broadcasted shape for a list of input shapes.
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(vec![]);
    }
    let mut max_ndim = 0;
    for s in shapes {
        max_ndim = max_ndim.max(s.len());
    }
    let mut out_shape = vec![1usize; max_ndim];
    for s in shapes {
        let offset = max_ndim - s.len();
        for (i, &dim) in s.iter().enumerate() {
            let target_idx = offset + i;
            if out_shape[target_idx] == 1 {
                out_shape[target_idx] = dim;
            } else if dim != 1 && dim != out_shape[target_idx] {
                return Err(anyhow!(
                    "The size of tensor ({}) must match the size of tensor ({}) at dimension {}",
                    dim,
                    out_shape[target_idx],
                    target_idx
                ));
            }
        }
    }
    Ok(out_shape)
}

/// Compute effective broadcast strides for a tensor mapped to target output shape.
pub fn effective_strides(tensor: &Tensor, target_shape: &[usize]) -> Vec<isize> {
    let target_ndim = target_shape.len();
    let src_ndim = tensor.ndim();
    let lead = target_ndim.saturating_sub(src_ndim);
    let mut eff = vec![0isize; target_ndim];

    for d in lead..target_ndim {
        let src_d = d - lead;
        if tensor.shape()[src_d] == 1 {
            eff[d] = 0; // Broadcasted dimension has stride 0
        } else {
            eff[d] = tensor.strides()[src_d];
        }
    }
    eff
}

/// Multi-tensor strided ND elementwise iterator.
/// Walks N inputs by their strides simultaneously, handling broadcasting and non-contiguous layouts
/// without any intermediate memory copies or contiguous conversions.
pub struct TensorIterator;

impl TensorIterator {
    /// Binary elementwise operation on two tensors producing a new broadcasted tensor.
    pub fn binary_f32<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor>
    where
        F: Fn(f32, f32) -> f32,
    {
        let out_shape = broadcast_shapes(&[a.shape(), b.shape()])?;
        let out_numel = numel(&out_shape);
        let out = Tensor::zeros_device(out_shape.clone(), a.device());

        if out_numel == 0 {
            return Ok(out);
        }

        let eff_a = effective_strides(a, &out_shape);
        let eff_b = effective_strides(b, &out_shape);
        let eff_out = out.strides().to_vec();

        let m = out_shape.len();
        let a_ptr = a.data_ptr() as *const f32;
        let b_ptr = b.data_ptr() as *const f32;
        let out_ptr = out.data_ptr() as *mut f32;

        let a_offset = a.storage_offset() as isize;
        let b_offset = b.storage_offset() as isize;
        let out_offset = out.storage_offset() as isize;

        if m == 0 {
            unsafe {
                *out_ptr.offset(out_offset) = op(*a_ptr.offset(a_offset), *b_ptr.offset(b_offset));
            }
            return Ok(out);
        }

        let mut coord = vec![0usize; m];
        let mut off_a = a_offset;
        let mut off_b = b_offset;
        let mut off_out = out_offset;

        for _ in 0..out_numel {
            unsafe {
                *out_ptr.offset(off_out) = op(*a_ptr.offset(off_a), *b_ptr.offset(off_b));
            }

            // O(1) amortized ND index coordinate and pointer offset advance
            for d in (0..m).rev() {
                coord[d] += 1;
                off_a += eff_a[d];
                off_b += eff_b[d];
                off_out += eff_out[d];

                if coord[d] < out_shape[d] {
                    break;
                }

                coord[d] = 0;
                let step = out_shape[d] as isize;
                off_a -= eff_a[d] * step;
                off_b -= eff_b[d] * step;
                off_out -= eff_out[d] * step;
            }
        }

        Ok(out)
    }

    /// Unary elementwise operation on a tensor.
    pub fn unary_f32<F>(a: &Tensor, op: F) -> Result<Tensor>
    where
        F: Fn(f32) -> f32,
    {
        let out_shape = a.shape().to_vec();
        let out_numel = numel(&out_shape);
        let out = Tensor::zeros_device(out_shape.clone(), a.device());

        if out_numel == 0 {
            return Ok(out);
        }

        let eff_a = a.strides();
        let eff_out = out.strides().to_vec();

        let m = out_shape.len();
        let a_ptr = a.data_ptr() as *const f32;
        let out_ptr = out.data_ptr() as *mut f32;

        let a_offset = a.storage_offset() as isize;
        let out_offset = out.storage_offset() as isize;

        if m == 0 {
            unsafe {
                *out_ptr.offset(out_offset) = op(*a_ptr.offset(a_offset));
            }
            return Ok(out);
        }

        let mut coord = vec![0usize; m];
        let mut off_a = a_offset;
        let mut off_out = out_offset;

        for _ in 0..out_numel {
            unsafe {
                *out_ptr.offset(off_out) = op(*a_ptr.offset(off_a));
            }

            for d in (0..m).rev() {
                coord[d] += 1;
                off_a += eff_a[d];
                off_out += eff_out[d];

                if coord[d] < out_shape[d] {
                    break;
                }

                coord[d] = 0;
                let step = out_shape[d] as isize;
                off_a -= eff_a[d] * step;
                off_out -= eff_out[d] * step;
            }
        }

        Ok(out)
    }

    /// Ternary elementwise operation (e.g. for where_op / clamp / fused ops).
    pub fn ternary_f32<F>(a: &Tensor, b: &Tensor, c: &Tensor, op: F) -> Result<Tensor>
    where
        F: Fn(f32, f32, f32) -> f32,
    {
        let out_shape = broadcast_shapes(&[a.shape(), b.shape(), c.shape()])?;
        let out_numel = numel(&out_shape);
        let out = Tensor::zeros_device(out_shape.clone(), a.device());

        if out_numel == 0 {
            return Ok(out);
        }

        let eff_a = effective_strides(a, &out_shape);
        let eff_b = effective_strides(b, &out_shape);
        let eff_c = effective_strides(c, &out_shape);
        let eff_out = out.strides().to_vec();

        let m = out_shape.len();
        let a_ptr = a.data_ptr() as *const f32;
        let b_ptr = b.data_ptr() as *const f32;
        let c_ptr = c.data_ptr() as *const f32;
        let out_ptr = out.data_ptr() as *mut f32;

        let a_offset = a.storage_offset() as isize;
        let b_offset = b.storage_offset() as isize;
        let c_offset = c.storage_offset() as isize;
        let out_offset = out.storage_offset() as isize;

        if m == 0 {
            unsafe {
                *out_ptr.offset(out_offset) = op(
                    *a_ptr.offset(a_offset),
                    *b_ptr.offset(b_offset),
                    *c_ptr.offset(c_offset),
                );
            }
            return Ok(out);
        }

        let mut coord = vec![0usize; m];
        let mut off_a = a_offset;
        let mut off_b = b_offset;
        let mut off_c = c_offset;
        let mut off_out = out_offset;

        for _ in 0..out_numel {
            unsafe {
                *out_ptr.offset(off_out) = op(
                    *a_ptr.offset(off_a),
                    *b_ptr.offset(off_b),
                    *c_ptr.offset(off_c),
                );
            }

            for d in (0..m).rev() {
                coord[d] += 1;
                off_a += eff_a[d];
                off_b += eff_b[d];
                off_c += eff_c[d];
                off_out += eff_out[d];

                if coord[d] < out_shape[d] {
                    break;
                }

                coord[d] = 0;
                let step = out_shape[d] as isize;
                off_a -= eff_a[d] * step;
                off_b -= eff_b[d] * step;
                off_c -= eff_c[d] * step;
                off_out -= eff_out[d] * step;
            }
        }

        Ok(out)
    }

    /// In-place binary operation: `a.op_(b)`.
    pub fn binary_inplace_f32<F>(a: &mut Tensor, b: &Tensor, op: F) -> Result<()>
    where
        F: Fn(f32, f32) -> f32,
    {
        if a.has_internal_overlap() {
            return Err(anyhow!(
                "unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation."
            ));
        }

        let out_shape = a.shape().to_vec();
        let eff_b = effective_strides(b, &out_shape);
        let eff_a = a.strides().to_vec();
        let out_numel = numel(&out_shape);

        let m = out_shape.len();
        let a_ptr = a.data_ptr() as *mut f32;
        let b_ptr = b.data_ptr() as *const f32;

        let a_offset = a.storage_offset() as isize;
        let b_offset = b.storage_offset() as isize;

        if m == 0 {
            unsafe {
                *a_ptr.offset(a_offset) = op(*a_ptr.offset(a_offset), *b_ptr.offset(b_offset));
            }
            a.bump_version();
            return Ok(());
        }

        let mut coord = vec![0usize; m];
        let mut off_a = a_offset;
        let mut off_b = b_offset;

        for _ in 0..out_numel {
            unsafe {
                *a_ptr.offset(off_a) = op(*a_ptr.offset(off_a), *b_ptr.offset(off_b));
            }

            for d in (0..m).rev() {
                coord[d] += 1;
                off_a += eff_a[d];
                off_b += eff_b[d];

                if coord[d] < out_shape[d] {
                    break;
                }

                coord[d] = 0;
                let step = out_shape[d] as isize;
                off_a -= eff_a[d] * step;
                off_b -= eff_b[d] * step;
            }
        }

        a.bump_version();
        Ok(())
    }
}
