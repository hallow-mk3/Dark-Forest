//! Index ops — gather, scatter, index_select, masked_fill, where_, slice, narrow.

use crate::tensor::{numel, Tensor};
use anyhow::{anyhow, Result};

// ---------------------------------------------------------------------------
// index_select: select indices along a dim
// ---------------------------------------------------------------------------
pub fn index_select(x: &Tensor, dim: usize, indices: &[usize]) -> Result<Tensor> {
    if dim >= x.ndim() {
        return Err(anyhow!("index_select: dim {} out of range", dim));
    }
    let src = x.to_vec();
    let shape = &x.shape;
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let dim_size = shape[dim];
    for &idx in indices {
        if idx >= dim_size {
            return Err(anyhow!("index_select: index {} out of range for dim {} with size {}", idx, dim, dim_size));
        }
    }
    let n_sel = indices.len();
    let mut out = vec![0.0f32; outer * n_sel * inner];
    for o in 0..outer {
        for (new_c, &idx) in indices.iter().enumerate() {
            for i in 0..inner {
                out[o * n_sel * inner + new_c * inner + i] =
                    src[o * dim_size * inner + idx * inner + i];
            }
        }
    }
    let mut out_shape = shape.clone();
    out_shape[dim] = n_sel;
    Tensor::from_vec_device(out, out_shape, x.device.clone())
}

// ---------------------------------------------------------------------------
// gather: out[i][j][k] = input[i][index[i][j][k]][k]  (for dim=1, 3D case)
//   General: works for any dim. shapes of input and index must match except at dim.
// ---------------------------------------------------------------------------
pub fn gather(x: &Tensor, dim: usize, index: &[usize]) -> Result<Tensor> {
    if dim >= x.ndim() {
        return Err(anyhow!("gather: dim {} out of range", dim));
    }
    let src = x.to_vec();
    let shape = &x.shape;
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let dim_size = shape[dim];
    let n_out = outer * inner; // index.len() should equal this if index is [outer * inner]

    if index.len() != n_out {
        // Alternatively accept a flat index over the full output
        // For simplicity: treat index as flat indices into dim
    }

    let mut out = vec![0.0f32; index.len()];
    for (flat, &idx) in index.iter().enumerate() {
        // Decompose flat into (outer_coord, inner_coord)
        let o = flat / inner;
        let i = flat % inner;
        if idx >= dim_size {
            return Err(anyhow!("gather: index {} out of range for dim size {}", idx, dim_size));
        }
        out[flat] = src[o * dim_size * inner + idx * inner + i];
    }
    Tensor::from_vec_device(out, vec![index.len()], x.device.clone())
}

// ---------------------------------------------------------------------------
// scatter_add: accumulate src values at positions given by index along dim
// ---------------------------------------------------------------------------
pub fn scatter_add(x: &Tensor, dim: usize, index: &[usize], src: &[f32]) -> Result<Tensor> {
    if dim >= x.ndim() {
        return Err(anyhow!("scatter_add: dim {} out of range", dim));
    }
    let mut out = x.to_vec();
    let shape = &x.shape;
    let inner: usize = shape[dim + 1..].iter().product();
    let dim_size = shape[dim];
    for (flat, (&idx, &val)) in index.iter().zip(src.iter()).enumerate() {
        let o = flat / inner;
        let i = flat % inner;
        if idx >= dim_size {
            return Err(anyhow!("scatter_add: index {} out of range", idx));
        }
        out[o * dim_size * inner + idx * inner + i] += val;
    }
    Tensor::from_vec_device(out, shape.clone(), x.device.clone())
}

// ---------------------------------------------------------------------------
// masked_fill: fill positions where mask is true with alue
// ---------------------------------------------------------------------------
pub fn masked_fill(x: &Tensor, mask: &[bool], value: f32) -> Result<Tensor> {
    if mask.len() != x.numel() {
        return Err(anyhow!("masked_fill: mask length {} != numel {}", mask.len(), x.numel()));
    }
    let mut out = x.to_vec();
    for (v, &m) in out.iter_mut().zip(mask.iter()) {
        if m {
            *v = value;
        }
    }
    Tensor::from_vec_device(out, x.shape.clone(), x.device.clone())
}

// ---------------------------------------------------------------------------
// where_: element-wise conditional select
// ---------------------------------------------------------------------------
pub fn where_op(condition: &[bool], a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(anyhow!("where: shape mismatch {:?} vs {:?}", a.shape, b.shape));
    }
    if condition.len() != a.numel() {
        return Err(anyhow!("where: condition length {} != numel {}", condition.len(), a.numel()));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let out: Vec<f32> = condition
        .iter()
        .zip(a_data.iter().zip(b_data.iter()))
        .map(|(&c, (&av, &bv))| if c { av } else { bv })
        .collect();
    Tensor::from_vec_device(out, a.shape.clone(), a.device.clone())
}

// ---------------------------------------------------------------------------
// slice: Python-style slice along one dim [start:stop:step]
// ---------------------------------------------------------------------------
pub fn slice_dim(x: &Tensor, dim: usize, start: usize, stop: usize, step: usize) -> Result<Tensor> {
    if dim >= x.ndim() {
        return Err(anyhow!("slice: dim {} out of range", dim));
    }
    let size = x.shape[dim];
    let stop = stop.min(size);
    if step == 0 {
        return Err(anyhow!("slice: step cannot be 0"));
    }
    let indices: Vec<usize> = (start..stop).step_by(step).collect();
    index_select(x, dim, &indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_index_select() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let y = index_select(&x, 0, &[0, 2, 4]).unwrap();
        assert_eq!(y.to_vec(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_masked_fill() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let mask = vec![true, false, true];
        let y = masked_fill(&x, &mask, -999.0).unwrap();
        assert_eq!(y.to_vec(), vec![-999.0, 2.0, -999.0]);
    }

    #[test]
    fn test_where_op() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let cond = vec![true, false, true];
        let y = where_op(&cond, &a, &b).unwrap();
        assert_eq!(y.to_vec(), vec![1.0, 5.0, 3.0]);
    }
}
