use anyhow::{anyhow, Result};
use crate::dispatch::cpu::softmax::log_softmax;
use crate::tensor::Tensor;

pub fn cross_entropy(logits: &Tensor, target: &Tensor) -> Result<Tensor> {
    if logits.ndim() != 2 || target.ndim() != 1 {
        return Err(anyhow!(
            "cross_entropy currently expects 2D logits [batch_size, num_classes] and 1D target [batch_size], got {:?} and {:?}",
            logits.shape(),
            target.shape()
        ));
    }

    let batch_size = logits.shape()[0];
    let num_classes = logits.shape()[1];

    if target.shape()[0] != batch_size {
        return Err(anyhow!(
            "cross_entropy batch size mismatch: logits {} != target {}",
            batch_size,
            target.shape()[0]
        ));
    }

    let log_probs = log_softmax(logits, -1)?;
    let target_labels = target.to_vec_i64()?;

    let lp_strides = log_probs.strides();
    let lp_ptr = log_probs.data_ptr() as *const f32;
    let lp_offset = log_probs.storage_offset() as isize;

    let stride_b = lp_strides[0];
    let stride_c = lp_strides[1];

    let mut total_loss = 0.0f32;

    for (b, &label) in target_labels.iter().enumerate() {
        if label < 0 || (label as usize) >= num_classes {
            return Err(anyhow!(
                "target label {} out of bounds for num_classes {}",
                label,
                num_classes
            ));
        }
        let c = label as usize;
        let off = lp_offset + (b as isize) * stride_b + (c as isize) * stride_c;
        let lp = unsafe { *lp_ptr.offset(off) };
        total_loss -= lp;
    }

    let mean_loss = total_loss / (batch_size as f32);
    let out = Tensor::zeros_device(vec![], logits.device());
    unsafe {
        *(out.data_ptr() as *mut f32) = mean_loss;
    }
    Ok(out)
}
