use anyhow::{anyhow, Result};
use crate::tensor::Tensor;

pub fn sum(tensor: &Tensor, dims: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
    let ndim = tensor.ndim();

    // Full reduction to scalar
    if dims.is_none() || ndim == 0 {
        let total = tensor.numel();
        let mut acc = 0.0f32;

        let eff_strides = tensor.strides();
        let m = ndim;
        let ptr = tensor.data_ptr() as *const f32;
        let offset = tensor.storage_offset() as isize;

        if m == 0 {
            acc = unsafe { *ptr.offset(offset) };
        } else {
            let mut coord = vec![0usize; m];
            let mut off = offset;

            for _ in 0..total {
                unsafe {
                    acc += *ptr.offset(off);
                }

                for d in (0..m).rev() {
                    coord[d] += 1;
                    off += eff_strides[d];

                    if coord[d] < tensor.shape()[d] {
                        break;
                    }

                    coord[d] = 0;
                    let step = tensor.shape()[d] as isize;
                    off -= eff_strides[d] * step;
                }
            }
        }

        let out = Tensor::zeros_device(vec![], tensor.device());
        unsafe {
            *(out.data_ptr() as *mut f32) = acc;
        }
        return Ok(out);
    }

    let dims = dims.unwrap();
    for &d in dims {
        if d >= ndim {
            return Err(anyhow!("Dimension out of range (got {}, ndim {})", d, ndim));
        }
    }

    let mut out_shape = Vec::new();
    for d in 0..ndim {
        if dims.contains(&d) {
            if keepdim {
                out_shape.push(1);
            }
        } else {
            out_shape.push(tensor.shape()[d]);
        }
    }

    let out = Tensor::zeros_device(out_shape.clone(), tensor.device());
    let total_in = tensor.numel();
    if total_in == 0 {
        return Ok(out);
    }

    let in_strides = tensor.strides();
    let out_strides = out.strides().to_vec();

    let in_ptr = tensor.data_ptr() as *const f32;
    let out_ptr = out.data_ptr() as *mut f32;

    let in_offset = tensor.storage_offset() as isize;
    let out_offset = out.storage_offset() as isize;

    let mut in_coord = vec![0usize; ndim];
    let mut in_off = in_offset;

    for _ in 0..total_in {
        // Map in_coord to out offset
        let mut out_off = out_offset;
        let mut out_d = 0;
        for d in 0..ndim {
            if dims.contains(&d) {
                if keepdim {
                    // coordinate is 0 for size-1 dim
                    out_d += 1;
                }
            } else {
                out_off += (in_coord[d] as isize) * out_strides[out_d];
                out_d += 1;
            }
        }

        unsafe {
            let val = *in_ptr.offset(in_off);
            *out_ptr.offset(out_off) += val;
        }

        // Advance in_coord
        for d in (0..ndim).rev() {
            in_coord[d] += 1;
            in_off += in_strides[d];

            if in_coord[d] < tensor.shape()[d] {
                break;
            }

            in_coord[d] = 0;
            let step = tensor.shape()[d] as isize;
            in_off -= in_strides[d] * step;
        }
    }

    Ok(out)
}

pub fn mean(tensor: &Tensor, dims: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
    let s = sum(tensor, dims, keepdim)?;
    let count = (tensor.numel() / s.numel().max(1)) as f32;
    s.div_scalar(count)
}
