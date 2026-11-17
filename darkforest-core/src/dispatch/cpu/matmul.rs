use anyhow::{anyhow, Result};
use crate::dispatch::iterator::broadcast_shapes;
use crate::tensor::{numel, Tensor};

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(anyhow!(
            "matmul requires both tensors to have ndim >= 2, got {:?} and {:?}",
            a.shape(),
            b.shape()
        ));
    }

    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    let m = a.shape()[a_ndim - 2];
    let k_a = a.shape()[a_ndim - 1];
    let k_b = b.shape()[b_ndim - 2];
    let n = b.shape()[b_ndim - 1];

    if k_a != k_b {
        return Err(anyhow!(
            "matmul dimension mismatch: K dimension {} != {}",
            k_a,
            k_b
        ));
    }

    let a_batch = &a.shape()[..a_ndim - 2];
    let b_batch = &b.shape()[..b_ndim - 2];
    let out_batch = broadcast_shapes(&[a_batch, b_batch])?;

    let mut out_shape = out_batch.clone();
    out_shape.push(m);
    out_shape.push(n);

    let out = Tensor::zeros_device(out_shape.clone(), a.device());
    let total_batches = numel(&out_batch);

    let a_strides = a.strides();
    let b_strides = b.strides();
    let out_strides = out.strides().to_vec();

    let a_stride_m = a_strides[a_ndim - 2];
    let a_stride_k = a_strides[a_ndim - 1];
    let b_stride_k = b_strides[b_ndim - 2];
    let b_stride_n = b_strides[b_ndim - 1];

    let out_stride_m = out_strides[out_shape.len() - 2];
    let out_stride_n = out_strides[out_shape.len() - 1];

    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;
    let out_ptr = out.data_ptr() as *mut f32;

    let a_offset = a.storage_offset() as isize;
    let b_offset = b.storage_offset() as isize;
    let out_offset = out.storage_offset() as isize;

    // Helper to compute batch offset
    let batch_ndim = out_batch.len();
    let mut coord = vec![0usize; batch_ndim];

    for batch_idx in 0..total_batches.max(1) {
        let mut cur_a_off = a_offset;
        let mut cur_b_off = b_offset;
        let mut cur_out_off = out_offset;

        if batch_ndim > 0 {
            // Unravel batch_idx
            let mut rem = batch_idx;
            for d in (0..batch_ndim).rev() {
                coord[d] = rem % out_batch[d];
                rem /= out_batch[d];
            }

            // Add batch strides
            let a_lead = batch_ndim.saturating_sub(a_batch.len());
            for d in a_lead..batch_ndim {
                let src_d = d - a_lead;
                if a_batch[src_d] > 1 {
                    cur_a_off += (coord[d] as isize) * a_strides[src_d];
                }
            }

            let b_lead = batch_ndim.saturating_sub(b_batch.len());
            for d in b_lead..batch_ndim {
                let src_d = d - b_lead;
                if b_batch[src_d] > 1 {
                    cur_b_off += (coord[d] as isize) * b_strides[src_d];
                }
            }

            for d in 0..batch_ndim {
                cur_out_off += (coord[d] as isize) * out_strides[d];
            }
        }

        // 2D Matmul kernel for this batch slice
        unsafe {
            for row in 0..m {
                let a_row_off = cur_a_off + (row as isize) * a_stride_m;
                let out_row_off = cur_out_off + (row as isize) * out_stride_m;

                for col in 0..n {
                    let b_col_off = cur_b_off + (col as isize) * b_stride_n;
                    let mut sum = 0.0f32;

                    for k in 0..k_a {
                        let val_a = *a_ptr.offset(a_row_off + (k as isize) * a_stride_k);
                        let val_b = *b_ptr.offset(b_col_off + (k as isize) * b_stride_k);
                        sum += val_a * val_b;
                    }

                    *out_ptr.offset(out_row_off + (col as isize) * out_stride_n) = sum;
                }
            }
        }
    }

    Ok(out)
}
