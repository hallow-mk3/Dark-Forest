//! Shape ops — full PyTorch parity.
//!
//! unsqueeze, squeeze, flatten, cat, stack, split, chunk, permute, expand, view.

use crate::tensor::{numel, Tensor};
use anyhow::{anyhow, Result};

// ---------------------------------------------------------------------------
// unsqueeze: insert a dim of size 1 at position dim
// ---------------------------------------------------------------------------
pub fn unsqueeze(x: &Tensor, dim: i64) -> Result<Tensor> {
    let ndim = x.ndim();
    let d = if dim < 0 {
        (ndim as i64 + 1 + dim) as usize
    } else {
        dim as usize
    };
    if d > ndim {
        return Err(anyhow!(
            "unsqueeze dim {} out of range for ndim {}",
            dim,
            ndim
        ));
    }
    let mut new_shape = x.shape.clone();
    new_shape.insert(d, 1);
    x.reshape(new_shape)
}

// ---------------------------------------------------------------------------
// squeeze: remove all dims of size 1, or just dim dim
// ---------------------------------------------------------------------------
pub fn squeeze(x: &Tensor, dim: Option<i64>) -> Result<Tensor> {
    let new_shape = match dim {
        None => x
            .shape
            .iter()
            .cloned()
            .filter(|&d| d != 1)
            .collect::<Vec<_>>(),
        Some(d) => {
            let ndim = x.ndim();
            let idx = if d < 0 {
                (ndim as i64 + d) as usize
            } else {
                d as usize
            };
            if idx >= ndim {
                return Err(anyhow!("squeeze dim {} out of range for ndim {}", d, ndim));
            }
            if x.shape[idx] == 1 {
                let mut s = x.shape.clone();
                s.remove(idx);
                s
            } else {
                x.shape.clone()
            }
        }
    };
    let new_shape = if new_shape.is_empty() {
        vec![1]
    } else {
        new_shape
    };
    x.reshape(new_shape)
}

// ---------------------------------------------------------------------------
// flatten: merge dims [start_dim, end_dim] into one
// ---------------------------------------------------------------------------
pub fn flatten(x: &Tensor, start_dim: i64, end_dim: i64) -> Result<Tensor> {
    let ndim = x.ndim();
    let s = if start_dim < 0 {
        (ndim as i64 + start_dim) as usize
    } else {
        start_dim as usize
    };
    let e = if end_dim < 0 {
        (ndim as i64 + end_dim) as usize
    } else {
        end_dim as usize
    };
    if s > e || e >= ndim {
        return Err(anyhow!(
            "flatten: invalid dims [{}, {}] for ndim {}",
            start_dim,
            end_dim,
            ndim
        ));
    }
    let flat: usize = x.shape[s..=e].iter().product();
    let mut new_shape: Vec<usize> = x.shape[..s].to_vec();
    new_shape.push(flat);
    new_shape.extend_from_slice(&x.shape[e + 1..]);
    x.reshape(new_shape)
}

// ---------------------------------------------------------------------------
// permute: reorder dimensions
// ---------------------------------------------------------------------------
pub fn permute(x: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let ndim = x.ndim();
    if dims.len() != ndim {
        return Err(anyhow!(
            "permute: dims length {} != ndim {}",
            dims.len(),
            ndim
        ));
    }
    let mut used = vec![false; ndim];
    for &d in dims {
        if d >= ndim || used[d] {
            return Err(anyhow!("permute: invalid dims {:?}", dims));
        }
        used[d] = true;
    }

    let src = x.to_vec();
    let new_shape: Vec<usize> = dims.iter().map(|&d| x.shape[d]).collect();
    let n = numel(&x.shape);
    let mut dst = vec![0.0f32; n];

    // Compute strides for old layout
    let old_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            s[i] = s[i + 1] * x.shape[i + 1];
        }
        s
    };
    let new_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            s[i] = s[i + 1] * new_shape[i + 1];
        }
        s
    };

    for flat in 0..n {
        // Decompose flat into new coords
        let mut rem = flat;
        let mut old_flat = 0;
        for (new_dim, &old_dim) in dims.iter().enumerate() {
            let coord = rem / new_strides[new_dim];
            rem %= new_strides[new_dim];
            old_flat += coord * old_strides[old_dim];
        }
        dst[flat] = src[old_flat];
    }

    Tensor::from_vec_device(dst, new_shape, x.device.clone())
}

// ---------------------------------------------------------------------------
// cat: concatenate tensors along dim
// ---------------------------------------------------------------------------
pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(anyhow!("cat: empty tensor list"));
    }
    let ndim = tensors[0].ndim();
    if dim >= ndim {
        return Err(anyhow!("cat: dim {} out of range for ndim {}", dim, ndim));
    }
    // Validate shapes are compatible
    for t in &tensors[1..] {
        if t.ndim() != ndim {
            return Err(anyhow!("cat: ndim mismatch"));
        }
        for d in 0..ndim {
            if d != dim && t.shape[d] != tensors[0].shape[d] {
                return Err(anyhow!("cat: shape mismatch at dim {}", d));
            }
        }
    }

    let outer: usize = tensors[0].shape[..dim].iter().product();
    let inner: usize = tensors[0].shape[dim + 1..].iter().product();
    let total_cat: usize = tensors.iter().map(|t| t.shape[dim]).sum();

    let mut out_shape = tensors[0].shape.clone();
    out_shape[dim] = total_cat;
    let mut out = vec![0.0f32; numel(&out_shape)];

    let out_inner_stride = total_cat * inner;
    let mut offset = 0usize;
    for t in tensors {
        let cat_size = t.shape[dim];
        let src = t.to_vec();
        for o in 0..outer {
            for c in 0..cat_size {
                for i in 0..inner {
                    out[o * out_inner_stride + (offset + c) * inner + i] =
                        src[o * cat_size * inner + c * inner + i];
                }
            }
        }
        offset += cat_size;
    }

    Tensor::from_vec_device(out, out_shape, tensors[0].device.clone())
}

// ---------------------------------------------------------------------------
// stack: create a new dimension and cat
// ---------------------------------------------------------------------------
pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    let unsqueezed: Vec<Tensor> = tensors
        .iter()
        .map(|t| unsqueeze(t, dim as i64))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Tensor> = unsqueezed.iter().collect();
    cat(&refs, dim)
}

// ---------------------------------------------------------------------------
// split: split tensor into chunks of split_size along dim
// ---------------------------------------------------------------------------
pub fn split(x: &Tensor, split_size: usize, dim: usize) -> Result<Vec<Tensor>> {
    if dim >= x.ndim() {
        return Err(anyhow!(
            "split: dim {} out of range for ndim {}",
            dim,
            x.ndim()
        ));
    }
    let total = x.shape[dim];
    let mut chunks = Vec::new();
    let mut offset = 0;
    while offset < total {
        let size = split_size.min(total - offset);
        let t = narrow(x, dim, offset, size)?;
        chunks.push(t);
        offset += size;
    }
    Ok(chunks)
}

// ---------------------------------------------------------------------------
// chunk: split into n equal (or near-equal) chunks
// ---------------------------------------------------------------------------
pub fn chunk(x: &Tensor, n_chunks: usize, dim: usize) -> Result<Vec<Tensor>> {
    if dim >= x.ndim() {
        return Err(anyhow!(
            "chunk: dim {} out of range for ndim {}",
            dim,
            x.ndim()
        ));
    }
    let total = x.shape[dim];
    let chunk_size = (total + n_chunks - 1) / n_chunks;
    split(x, chunk_size, dim)
}

// ---------------------------------------------------------------------------
// narrow: select a slice [start, start+length) along dim
// ---------------------------------------------------------------------------
pub fn narrow(x: &Tensor, dim: usize, start: usize, length: usize) -> Result<Tensor> {
    if dim >= x.ndim() {
        return Err(anyhow!(
            "narrow: dim {} out of range for ndim {}",
            dim,
            x.ndim()
        ));
    }
    if start + length > x.shape[dim] {
        return Err(anyhow!(
            "narrow: start({}) + length({}) > shape[{}]({})",
            start,
            length,
            dim,
            x.shape[dim]
        ));
    }
    let src = x.to_vec();
    let shape = &x.shape;
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0.0f32; outer * length * inner];
    for o in 0..outer {
        for c in 0..length {
            for i in 0..inner {
                out[o * length * inner + c * inner + i] =
                    src[o * shape[dim] * inner + (start + c) * inner + i];
            }
        }
    }
    let mut out_shape = shape.clone();
    out_shape[dim] = length;
    Tensor::from_vec_device(out, out_shape, x.device.clone())
}

// ---------------------------------------------------------------------------
// expand: broadcast a tensor to a target shape (dims of size 1 can be expanded)
// ---------------------------------------------------------------------------
pub fn expand(x: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    // Align shapes from the right
    let ndim_x = x.ndim();
    let ndim_t = target_shape.len();
    if ndim_t < ndim_x {
        return Err(anyhow!(
            "expand: target ndim {} < source ndim {}",
            ndim_t,
            ndim_x
        ));
    }
    let pad = ndim_t - ndim_x;
    let src = x.to_vec();
    let n_out = numel(target_shape);
    let mut out = vec![0.0f32; n_out];

    // Compute strides for source (padded to ndim_t; padded dims have stride 0)
    let mut src_strides = vec![0usize; ndim_t];
    {
        let mut s = 1usize;
        for i in (0..ndim_x).rev() {
            let src_dim = x.shape[i];
            if src_dim == 1 {
                src_strides[i + pad] = 0; // broadcast dim
            } else if src_dim == target_shape[i + pad] {
                src_strides[i + pad] = s;
            } else {
                return Err(anyhow!(
                    "expand: cannot expand dim {} from {} to {}",
                    i,
                    src_dim,
                    target_shape[i + pad]
                ));
            }
            s *= src_dim;
        }
    }
    // Compute strides for output
    let mut out_strides = vec![1usize; ndim_t];
    for i in (0..ndim_t - 1).rev() {
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1];
    }

    for flat in 0..n_out {
        let mut rem = flat;
        let mut src_flat = 0;
        for d in 0..ndim_t {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            src_flat += coord * src_strides[d];
        }
        out[flat] = src[src_flat];
    }

    Tensor::from_vec_device(out, target_shape.to_vec(), x.device.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_unsqueeze_squeeze() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let y = unsqueeze(&x, 0).unwrap();
        assert_eq!(y.shape, vec![1, 3]);
        let z = squeeze(&y, Some(0)).unwrap();
        assert_eq!(z.shape, vec![3]);
    }

    #[test]
    fn test_flatten() {
        let x = Tensor::from_vec(vec![1.0; 12], vec![2, 3, 2]).unwrap();
        let y = flatten(&x, 1, -1).unwrap();
        assert_eq!(y.shape, vec![2, 6]);
    }

    #[test]
    fn test_cat() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]).unwrap();
        let c = cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape, vec![5]);
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_split() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let chunks = split(&x, 2, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].to_vec(), vec![1.0, 2.0]);
        assert_eq!(chunks[2].to_vec(), vec![5.0]);
    }

    #[test]
    fn test_permute() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let y = permute(&x, &[1, 0]).unwrap();
        assert_eq!(y.shape, vec![3, 2]);
        let v = y.to_vec();
        assert_eq!(v, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_expand() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = expand(&x, &[4, 3]).unwrap();
        assert_eq!(y.shape, vec![4, 3]);
        assert_eq!(
            y.to_vec(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }
}
