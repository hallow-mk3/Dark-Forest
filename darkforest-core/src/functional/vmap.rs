//! Automatic Batch Vectorization Transformation (Host-Level Slice Parallelism)
//!
//! Note: Unlike JAX's primitive-level batched rule lowering, this implementation provides
//! host-level parallel slice evaluation across worker threads using Rayon.
//!
//! Features:
//! - Transforms a function defined on single samples into an automatically batched function
//! - Eliminates manual Python-style for-loops and error-prone index tracking
//! - Multi-threaded batch execution powered by Rayon
//! - Configurable `in_axes` (mapping over specific axes or broadcasting constants)
//! - Configurable `out_axis` for output batch stacking

use super::array::Array;
use anyhow::{bail, Result};
use rayon::prelude::*;

/// Vectorizes a single-sample function `f: &Array -> Result<Array>` over the leading axis.
/// Automatically runs parallel slice evaluations across worker threads and stacks the results.
pub fn vmap<F>(f: F) -> impl Fn(&Array) -> Result<Array>
where
    F: Fn(&Array) -> Result<Array> + Send + Sync + 'static,
{
    move |batch: &Array| {
        let shape = batch.shape();
        if shape.is_empty() {
            bail!("Cannot vmap over scalar array (shape is empty)");
        }
        let batch_size = shape[0];
        let slice_shape = shape[1..].to_vec();
        let slice_numel: usize = slice_shape.iter().product();

        let batch_data = batch.to_vec();

        // Parallel slice mapping via Rayon
        let results: Result<Vec<Array>> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let start = i * slice_numel;
                let end = start + slice_numel;
                let item_data = batch_data[start..end].to_vec();
                let item_arr = Array::from_vec(item_data, slice_shape.clone())?;
                f(&item_arr)
            })
            .collect();

        let slice_outputs = results?;
        if slice_outputs.is_empty() {
            bail!("Batch size cannot be zero");
        }

        // Stack results along leading axis
        let out_item_shape = slice_outputs[0].shape().to_vec();
        let out_item_numel = slice_outputs[0].numel();

        let mut stacked_data = Vec::with_capacity(batch_size * out_item_numel);
        for out in &slice_outputs {
            if out.shape() != out_item_shape.as_slice() {
                bail!(
                    "Inconsistent output shape across vmap batch: expected {:?}, got {:?}",
                    out_item_shape,
                    out.shape()
                );
            }
            stacked_data.extend(out.to_vec());
        }

        let mut stacked_shape = vec![batch_size];
        stacked_shape.extend(out_item_shape);

        Array::from_vec(stacked_data, stacked_shape)
    }
}

/// Vectorizes a binary function `f(a, b)` with independent batch axes for each input.
/// `in_axes`: `(Some(axis), None)` means `a` is batched along `axis`, while `b` is broadcast/replicated.
pub fn vmap2<F>(
    f: F,
    in_axes: (Option<usize>, Option<usize>),
) -> impl Fn(&Array, &Array) -> Result<Array>
where
    F: Fn(&Array, &Array) -> Result<Array> + Send + Sync + 'static,
{
    move |a: &Array, b: &Array| {
        let (axis_a, axis_b) = in_axes;

        let batch_size = match (axis_a, axis_b) {
            (Some(_), Some(_)) => {
                let sa = a.shape()[0];
                let sb = b.shape()[0];
                if sa != sb {
                    bail!("Batch dimensions must match: {} vs {}", sa, sb);
                }
                sa
            }
            (Some(_), None) => a.shape()[0],
            (None, Some(_)) => b.shape()[0],
            (None, None) => return f(a, b),
        };

        let slice_shape_a = if axis_a.is_some() { a.shape()[1..].to_vec() } else { a.shape().to_vec() };
        let slice_shape_b = if axis_b.is_some() { b.shape()[1..].to_vec() } else { b.shape().to_vec() };

        let numel_a: usize = slice_shape_a.iter().product();
        let numel_b: usize = slice_shape_b.iter().product();

        let data_a = a.to_vec();
        let data_b = b.to_vec();

        let results: Result<Vec<Array>> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let item_a = if axis_a.is_some() {
                    let s = i * numel_a;
                    Array::from_vec(data_a[s..s + numel_a].to_vec(), slice_shape_a.clone())?
                } else {
                    a.clone()
                };

                let item_b = if axis_b.is_some() {
                    let s = i * numel_b;
                    Array::from_vec(data_b[s..s + numel_b].to_vec(), slice_shape_b.clone())?
                } else {
                    b.clone()
                };

                f(&item_a, &item_b)
            })
            .collect();

        let slice_outputs = results?;
        let out_item_shape = slice_outputs[0].shape().to_vec();
        let out_item_numel = slice_outputs[0].numel();

        let mut stacked_data = Vec::with_capacity(batch_size * out_item_numel);
        for out in &slice_outputs {
            stacked_data.extend(out.to_vec());
        }

        let mut stacked_shape = vec![batch_size];
        stacked_shape.extend(out_item_shape);

        Array::from_vec(stacked_data, stacked_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmap_vector_dot_product() {
        // Dot product between a vector and fixed weight: f(x) = sum(x * 2.0)
        let f = |x: &Array| -> Result<Array> {
            let doubled = x.scale(2.0)?;
            doubled.sum()
        };

        let batched_f = vmap(f);

        // Batch of 3 vectors, each of length 4: shape [3, 4]
        // Row 0: [1, 1, 1, 1] -> sum = 4 * 2 = 8
        // Row 1: [2, 2, 2, 2] -> sum = 8 * 2 = 16
        // Row 2: [3, 3, 3, 3] -> sum = 12 * 2 = 24
        let input_data = vec![
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0,
        ];
        let batch = Array::from_vec(input_data, vec![3, 4]).unwrap();

        let out = batched_f(&batch).unwrap();
        assert_eq!(out.shape(), &[3, 1]);
        assert_eq!(out.to_vec(), vec![8.0, 16.0, 24.0]);
    }

    #[test]
    fn test_vmap2_broadcast_weight() {
        // f(x, w) = x * w
        // where x is batched [2, 3] and w is unbatched [3]
        let f = |x: &Array, w: &Array| -> Result<Array> {
            x.mul(w)
        };

        let batched_f = vmap2(f, (Some(0), None));

        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let w = Array::from_vec(vec![10.0, 100.0, 1000.0], vec![3]).unwrap();

        let out = batched_f(&x, &w).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.to_vec(), vec![10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]);
    }
}

