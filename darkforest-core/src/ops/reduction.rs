//! Reduction ops — full PyTorch parity.
//!
//! Includes: mean, var, std, sum_dim, mean_dim, max, min, argmax, argmin.
//! All with optional dimension and keepdim semantics.

use crate::tensor::{numel, Tensor};
use anyhow::{anyhow, Result};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_dim(ndim: usize, dim: i64) -> Result<usize> {
    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    if d >= ndim {
        return Err(anyhow!("dim {} out of range for ndim {}", dim, ndim));
    }
    Ok(d)
}

// ---------------------------------------------------------------------------
// mean(x) — scalar mean over all elements
// ---------------------------------------------------------------------------
pub fn mean(x: &Tensor) -> Result<Tensor> {
    let data = x.to_vec();
    let n = data.len() as f32;
    let m: f32 = data.iter().sum::<f32>() / n;
    Tensor::from_vec_device(vec![m], vec![1], x.device.clone())
}

pub fn mean_backward(grad_out: &[f32], n: usize, shape: &[usize]) -> Vec<f32> {
    let g = grad_out[0] / n as f32;
    vec![g; numel(shape)]
}

// ---------------------------------------------------------------------------
// mean_dim(x, dim, keepdim) — mean along one dimension
// ---------------------------------------------------------------------------
pub fn mean_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<Tensor> {
    let d = check_dim(x.ndim(), dim)?;
    let out = reduce_dim(x, d, |acc, v| acc + v, |s, n| s / n as f32)?;
    if keepdim {
        let mut new_shape = x.shape.clone();
        new_shape[d] = 1;
        out.reshape(new_shape)
    } else {
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// sum_dim(x, dim, keepdim)
// ---------------------------------------------------------------------------
pub fn sum_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<Tensor> {
    let d = check_dim(x.ndim(), dim)?;
    let out = reduce_dim(x, d, |acc, v| acc + v, |s, _| s)?;
    if keepdim {
        let mut new_shape = x.shape.clone();
        new_shape[d] = 1;
        out.reshape(new_shape)
    } else {
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// var(x, unbiased) — variance over all elements
// ---------------------------------------------------------------------------
pub fn var(x: &Tensor, unbiased: bool) -> Result<Tensor> {
    let data = x.to_vec();
    let n = data.len() as f32;
    let m: f32 = data.iter().sum::<f32>() / n;
    let v: f32 = data.iter().map(|&x| (x - m) * (x - m)).sum::<f32>()
        / if unbiased { (n - 1.0).max(1.0) } else { n };
    Tensor::from_vec_device(vec![v], vec![1], x.device.clone())
}

// ---------------------------------------------------------------------------
// std(x, unbiased)
// ---------------------------------------------------------------------------
pub fn std_dev(x: &Tensor, unbiased: bool) -> Result<Tensor> {
    let v = var(x, unbiased)?.to_vec()[0];
    Tensor::from_vec_device(vec![v.sqrt()], vec![1], x.device.clone())
}

// ---------------------------------------------------------------------------
// max / min — scalar
// ---------------------------------------------------------------------------
pub fn max(x: &Tensor) -> Result<Tensor> {
    let v = x
        .to_vec()
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    Tensor::from_vec_device(vec![v], vec![1], x.device.clone())
}

pub fn min(x: &Tensor) -> Result<Tensor> {
    let v = x
        .to_vec()
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    Tensor::from_vec_device(vec![v], vec![1], x.device.clone())
}

// ---------------------------------------------------------------------------
// argmax / argmin — scalar index
// ---------------------------------------------------------------------------
pub fn argmax(x: &Tensor) -> usize {
    let data = x.to_vec();
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn argmin(x: &Tensor) -> usize {
    let data = x.to_vec();
    data.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// max_dim / min_dim — max/min along a dimension, returning (values, indices)
// ---------------------------------------------------------------------------
pub fn max_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<(Tensor, Vec<usize>)> {
    let d = check_dim(x.ndim(), dim)?;
    let (values, indices) = reduce_dim_with_index(x, d, f32::NEG_INFINITY, |a, b| a > b)?;
    let out = if keepdim {
        let mut new_shape = x.shape.clone();
        new_shape[d] = 1;
        values.reshape(new_shape)?
    } else {
        values
    };
    Ok((out, indices))
}

pub fn min_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<(Tensor, Vec<usize>)> {
    let d = check_dim(x.ndim(), dim)?;
    let (values, indices) = reduce_dim_with_index(x, d, f32::INFINITY, |a, b| a < b)?;
    let out = if keepdim {
        let mut new_shape = x.shape.clone();
        new_shape[d] = 1;
        values.reshape(new_shape)?
    } else {
        values
    };
    Ok((out, indices))
}

// ---------------------------------------------------------------------------
// Internal: generic dimension reduction
// ---------------------------------------------------------------------------
fn reduce_dim<F, G>(x: &Tensor, dim: usize, combine: F, finalize: G) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
    G: Fn(f32, usize) -> f32,
{
    let data = x.to_vec();
    let shape = &x.shape;
    let ndim = shape.len();

    // outer = product of dims before dim
    let outer: usize = shape[..dim].iter().product();
    // inner = product of dims after dim
    let inner: usize = shape[dim + 1..].iter().product();
    let reduce_size = shape[dim];

    let mut out_data = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f32;
            for r in 0..reduce_size {
                let idx = o * reduce_size * inner + r * inner + i;
                acc = combine(acc, data[idx]);
            }
            out_data[o * inner + i] = finalize(acc, reduce_size);
        }
    }

    let mut out_shape = shape.clone();
    out_shape.remove(dim);
    if out_shape.is_empty() {
        out_shape = vec![1];
    }

    let _ = ndim; // suppress unused warning
    Tensor::from_vec_device(out_data, out_shape, x.device.clone())
}

fn reduce_dim_with_index<F>(
    x: &Tensor,
    dim: usize,
    init: f32,
    better: F,
) -> Result<(Tensor, Vec<usize>)>
where
    F: Fn(f32, f32) -> bool,
{
    let data = x.to_vec();
    let shape = &x.shape;
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let reduce_size = shape[dim];

    let mut values = vec![init; outer * inner];
    let mut indices = vec![0usize; outer * inner];

    for o in 0..outer {
        for i in 0..inner {
            let out_idx = o * inner + i;
            let mut best_val = init;
            let mut best_idx = 0;
            for r in 0..reduce_size {
                let idx = o * reduce_size * inner + r * inner + i;
                let v = data[idx];
                if better(v, best_val) {
                    best_val = v;
                    best_idx = r;
                }
            }
            values[out_idx] = best_val;
            indices[out_idx] = best_idx;
        }
    }

    let mut out_shape = shape.clone();
    out_shape.remove(dim);
    if out_shape.is_empty() {
        out_shape = vec![1];
    }

    let t = Tensor::from_vec_device(values, out_shape, x.device.clone())?;
    Ok((t, indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_mean() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let m = mean(&x).unwrap();
        assert!((m.get(0) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_var_biased() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let v = var(&x, false).unwrap().get(0);
        // mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        assert!((v - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_min() {
        let x = Tensor::from_vec(vec![-1.0, 5.0, 3.0], vec![3]).unwrap();
        assert!((max(&x).unwrap().get(0) - 5.0).abs() < 1e-6);
        assert!((min(&x).unwrap().get(0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax() {
        let x = Tensor::from_vec(vec![1.0, 3.0, 2.0], vec![3]).unwrap();
        assert_eq!(argmax(&x), 1);
    }

    #[test]
    fn test_sum_dim() {
        // shape [2, 3], sum along dim=0 → shape [3]
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let out = sum_dim(&x, 0, false).unwrap();
        let v = out.to_vec();
        assert!((v[0] - 5.0).abs() < 1e-6);
        assert!((v[1] - 7.0).abs() < 1e-6);
        assert!((v[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_dim() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let out = mean_dim(&x, 0, false).unwrap();
        let v = out.to_vec();
        assert!((v[0] - 2.5).abs() < 1e-6);
        assert!((v[1] - 3.5).abs() < 1e-6);
        assert!((v[2] - 4.5).abs() < 1e-6);
    }
}
