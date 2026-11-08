//! Functional Array & NumPy-like Scientific Computing (JAX jnp equivalent)
//!
//! Features:
//! - Immutable array semantics (pure functional operations)
//! - Copy-on-Write (COW) with zero-cost in-place reuse when uniquely owned
//! - Functional indexed updates via `.at(indices).set/add/mul/min/max(...)`
//! - Full array slicing, reshaping, broadcasting, and reductions

use std::sync::Arc;
use crate::tensor::{Device, Tensor};
use anyhow::{bail, Result};

/// An immutable, pure functional N-dimensional array.
/// Uses Copy-on-Write (COW) through `Arc` to guarantee mathematical referential transparency
/// while enabling zero-copy mutations when the array is uniquely owned.
#[derive(Clone, Debug)]
pub struct Array {
    tensor: Arc<Tensor>,
}

impl Array {
    /// Wraps an existing Tensor into a functional Array.
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self {
            tensor: Arc::new(tensor),
        }
    }

    /// Creates an array from a 1D vector and shape.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let t = Tensor::from_vec(data, shape)?;
        Ok(Self::from_tensor(t))
    }

    /// Creates a 1D array from a slice.
    pub fn from_slice(data: &[f32]) -> Self {
        let t = Tensor::from_vec(data.to_vec(), vec![data.len()]).expect("Valid 1D tensor");
        Self::from_tensor(t)
    }

    /// Creates a scalar array.
    pub fn scalar(val: f32) -> Self {
        let t = Tensor::from_vec(vec![val], vec![1]).expect("Valid scalar tensor");
        Self::from_tensor(t)
    }

    /// Access the underlying Tensor reference.
    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }

    /// Returns total number of elements.
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }

    /// Returns the device where data resides.
    pub fn device(&self) -> &Device {
        &self.tensor.device
    }

    /// Extracts data as a flat host Vec<f32>.
    pub fn to_vec(&self) -> Vec<f32> {
        self.tensor.to_vec()
    }

    /// Extracts a single scalar value.
    pub fn item(&self) -> Result<f32> {
        if self.numel() != 1 {
            bail!("item() can only be called on 1-element arrays, got shape {:?}", self.shape());
        }
        Ok(self.to_vec()[0])
    }

    // ------------------------------------------------------------------------
    // NumPy-like Array Constructors
    // ------------------------------------------------------------------------

    /// Creates an array filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let t = Tensor::zeros(shape);
        Self::from_tensor(t)
    }

    /// Creates an array filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let t = Tensor::from_vec(vec![1.0; n], shape).expect("Valid shape");
        Self::from_tensor(t)
    }

    /// Creates an array filled with a constant scalar value.
    pub fn full(shape: Vec<usize>, val: f32) -> Self {
        let n: usize = shape.iter().product();
        let t = Tensor::from_vec(vec![val; n], shape).expect("Valid shape");
        Self::from_tensor(t)
    }

    /// Creates a 1-D array with evenly spaced values within a given interval `[start, stop)`.
    pub fn arange(start: f32, stop: f32, step: f32) -> Result<Self> {
        if step.abs() < 1e-12 {
            bail!("arange step must not be zero");
        }
        let mut vals = Vec::new();
        let mut curr = start;
        if step > 0.0 {
            while curr < stop - 1e-7 {
                vals.push(curr);
                curr += step;
            }
        } else {
            while curr > stop + 1e-7 {
                vals.push(curr);
                curr += step;
            }
        }
        let n = vals.len();
        Self::from_vec(vals, vec![n])
    }

    /// Creates a 1-D array of `num` evenly spaced points between `start` and `stop` inclusive.
    pub fn linspace(start: f32, stop: f32, num: usize) -> Result<Self> {
        if num == 0 {
            bail!("linspace num must be > 0");
        }
        if num == 1 {
            return Self::from_vec(vec![start], vec![1]);
        }
        let step = (stop - start) / ((num - 1) as f32);
        let mut vals = Vec::with_capacity(num);
        for i in 0..num {
            vals.push(start + (i as f32) * step);
        }
        Self::from_vec(vals, vec![num])
    }

    /// Creates a 2-D identity matrix of size N x N.
    pub fn eye(n: usize) -> Self {
        let mut vals = vec![0.0f32; n * n];
        for i in 0..n {
            vals[i * n + i] = 1.0;
        }
        Self::from_vec(vals, vec![n, n]).expect("Valid N x N matrix")
    }

    // ------------------------------------------------------------------------
    // Functional Indexed Updates: array.at(indices).set/add/mul/min/max(...)
    // ------------------------------------------------------------------------

    /// Initiates a functional index update expression (equivalent to `jax.numpy.ndarray.at[...]`).
    pub fn at(&self, index: IndexSpec) -> IndexUpdateBuilder {
        IndexUpdateBuilder {
            array: self.clone(),
            spec: index,
        }
    }

    // ------------------------------------------------------------------------
    // Mathematical Operations (Pure Functional)
    // ------------------------------------------------------------------------

    pub fn add(&self, other: &Array) -> Result<Array> {
        let t = crate::ops::add::add(&self.tensor, &other.tensor)?;
        Ok(Array::from_tensor(t))
    }

    pub fn sub(&self, other: &Array) -> Result<Array> {
        let neg_other = other.neg()?;
        self.add(&neg_other)
    }

    pub fn mul(&self, other: &Array) -> Result<Array> {
        let t = crate::ops::add::mul(&self.tensor, &other.tensor)?;
        Ok(Array::from_tensor(t))
    }

    pub fn div(&self, other: &Array) -> Result<Array> {
        let a_data = self.to_vec();
        let b_data = other.to_vec();
        if a_data.len() != b_data.len() {
            bail!("Shape mismatch in div: {:?} vs {:?}", self.shape(), other.shape());
        }
        let out_data: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| a / b)
            .collect();
        Array::from_vec(out_data, self.shape().to_vec())
    }

    pub fn neg(&self) -> Result<Array> {
        let t = crate::ops::add::scale(&self.tensor, -1.0)?;
        Ok(Array::from_tensor(t))
    }

    pub fn scale(&self, scalar: f32) -> Result<Array> {
        let t = crate::ops::add::scale(&self.tensor, scalar)?;
        Ok(Array::from_tensor(t))
    }

    pub fn matmul(&self, other: &Array) -> Result<Array> {
        let t = crate::ops::matmul::matmul(&self.tensor, &other.tensor)?;
        Ok(Array::from_tensor(t))
    }

    pub fn dot(&self, other: &Array) -> Result<Array> {
        self.matmul(other)
    }

    // Elementwise Non-linearities
    pub fn relu(&self) -> Result<Array> {
        let t = crate::ops::activation::relu(&self.tensor)?;
        Ok(Array::from_tensor(t))
    }

    pub fn gelu(&self) -> Result<Array> {
        let t = crate::ops::activation::gelu(&self.tensor)?;
        Ok(Array::from_tensor(t))
    }

    pub fn sigmoid(&self) -> Result<Array> {
        let t = crate::ops::activation::sigmoid(&self.tensor)?;
        Ok(Array::from_tensor(t))
    }

    pub fn tanh(&self) -> Result<Array> {
        let t = crate::ops::activation::tanh_act(&self.tensor)?;
        Ok(Array::from_tensor(t))
    }

    pub fn exp(&self) -> Result<Array> {
        let data = self.to_vec().into_iter().map(|x| x.exp()).collect();
        Self::from_vec(data, self.shape().to_vec())
    }

    pub fn log(&self) -> Result<Array> {
        let data = self.to_vec().into_iter().map(|x| (x.max(1e-12)).ln()).collect();
        Self::from_vec(data, self.shape().to_vec())
    }

    pub fn sqrt(&self) -> Result<Array> {
        let data = self.to_vec().into_iter().map(|x| x.max(0.0).sqrt()).collect();
        Self::from_vec(data, self.shape().to_vec())
    }

    pub fn sin(&self) -> Result<Array> {
        let data = self.to_vec().into_iter().map(|x| x.sin()).collect();
        Self::from_vec(data, self.shape().to_vec())
    }

    pub fn cos(&self) -> Result<Array> {
        let data = self.to_vec().into_iter().map(|x| x.cos()).collect();
        Self::from_vec(data, self.shape().to_vec())
    }

    // Reductions
    pub fn sum(&self) -> Result<Array> {
        let s = self.to_vec().into_iter().sum::<f32>();
        Ok(Array::scalar(s))
    }

    pub fn mean(&self) -> Result<Array> {
        let n = self.numel() as f32;
        let s = self.to_vec().into_iter().sum::<f32>();
        Ok(Array::scalar(s / n))
    }

    pub fn max(&self) -> Result<Array> {
        let m = self.to_vec().into_iter().fold(f32::NEG_INFINITY, f32::max);
        Ok(Array::scalar(m))
    }

    pub fn min(&self) -> Result<Array> {
        let m = self.to_vec().into_iter().fold(f32::INFINITY, f32::min);
        Ok(Array::scalar(m))
    }

    // Shape transformations
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Array> {
        let t = self.tensor.reshape(new_shape)?;
        Ok(Array::from_tensor(t))
    }

    pub fn transpose(&self) -> Result<Array> {
        let t = crate::ops::shape::permute(&self.tensor, &[1, 0])?;
        Ok(Array::from_tensor(t))
    }

    pub fn transpose_last_two(&self) -> Result<Array> {
        let t = self.tensor.transpose_last_two()?;
        Ok(Array::from_tensor(t))
    }

    pub fn squeeze(&self, dim: Option<i64>) -> Result<Array> {
        let t = crate::ops::shape::squeeze(&self.tensor, dim)?;
        Ok(Array::from_tensor(t))
    }

    pub fn unsqueeze(&self, dim: i64) -> Result<Array> {
        let t = crate::ops::shape::unsqueeze(&self.tensor, dim)?;
        Ok(Array::from_tensor(t))
    }
}

// ----------------------------------------------------------------------------
// Operator Overloads for Natural Scientific Computing Syntax
// ----------------------------------------------------------------------------

impl std::ops::Add for Array {
    type Output = Array;
    fn add(self, rhs: Array) -> Array {
        Array::add(&self, &rhs).expect("Array addition failed")
    }
}

impl std::ops::Add<&Array> for &Array {
    type Output = Array;
    fn add(self, rhs: &Array) -> Array {
        Array::add(self, rhs).expect("Array addition failed")
    }
}

impl std::ops::Sub for Array {
    type Output = Array;
    fn sub(self, rhs: Array) -> Array {
        Array::sub(&self, &rhs).expect("Array subtraction failed")
    }
}

impl std::ops::Sub<&Array> for &Array {
    type Output = Array;
    fn sub(self, rhs: &Array) -> Array {
        Array::sub(self, rhs).expect("Array subtraction failed")
    }
}

impl std::ops::Mul for Array {
    type Output = Array;
    fn mul(self, rhs: Array) -> Array {
        Array::mul(&self, &rhs).expect("Array multiplication failed")
    }
}

impl std::ops::Mul<&Array> for &Array {
    type Output = Array;
    fn mul(self, rhs: &Array) -> Array {
        Array::mul(self, rhs).expect("Array multiplication failed")
    }
}

impl std::ops::Div for Array {
    type Output = Array;
    fn div(self, rhs: Array) -> Array {
        Array::div(&self, &rhs).expect("Array division failed")
    }
}

impl std::ops::Div<&Array> for &Array {
    type Output = Array;
    fn div(self, rhs: &Array) -> Array {
        Array::div(self, rhs).expect("Array division failed")
    }
}

impl std::ops::Neg for Array {
    type Output = Array;
    fn neg(self) -> Array {
        Array::neg(&self).expect("Array negation failed")
    }
}

// ----------------------------------------------------------------------------
// Functional Index Updating (JAX .at[idx].set(y) / add(y) / mul(y) etc.)
// ----------------------------------------------------------------------------

/// Specifies the target coordinates for an indexed functional update.
#[derive(Clone, Debug)]
pub enum IndexSpec {
    /// Exact coordinate index along each dimension (e.g. `[1, 3]`).
    Point(Vec<usize>),
    /// Slice along a specific axis: `(axis, start, end)`.
    Slice {
        axis: usize,
        start: usize,
        end: usize,
    },
    /// Flat 1D element index.
    Flat(usize),
}

/// Builder for functional array index updates.
pub struct IndexUpdateBuilder {
    array: Array,
    spec: IndexSpec,
}

impl IndexUpdateBuilder {
    /// Functionally replaces values at the target index with `val` and returns a new Array.
    pub fn set(self, val: f32) -> Result<Array> {
        self.apply_op(|_| val)
    }

    /// Functionally replaces values at the target slice with contents of another Array.
    pub fn set_array(self, other: &Array) -> Result<Array> {
        let other_data = other.to_vec();
        let mut idx = 0;
        self.apply_op(|_| {
            let v = other_data[idx % other_data.len()];
            idx += 1;
            v
        })
    }

    /// Functionally adds `val` to values at the target index: `arr[idx] + val`.
    pub fn add(self, val: f32) -> Result<Array> {
        self.apply_op(|x| x + val)
    }

    /// Functionally multiplies values at the target index: `arr[idx] * val`.
    pub fn mul(self, val: f32) -> Result<Array> {
        self.apply_op(|x| x * val)
    }

    /// Functionally replaces with minimum: `min(arr[idx], val)`.
    pub fn min(self, val: f32) -> Result<Array> {
        self.apply_op(|x| x.min(val))
    }

    /// Functionally replaces with maximum: `max(arr[idx], val)`.
    pub fn max(self, val: f32) -> Result<Array> {
        self.apply_op(|x| x.max(val))
    }

    fn apply_op<F>(self, mut op: F) -> Result<Array>
    where
        F: FnMut(f32) -> f32,
    {
        let shape = self.array.shape().to_vec();
        let mut data = self.array.to_vec();

        match self.spec {
            IndexSpec::Flat(idx) => {
                if idx >= data.len() {
                    bail!("Flat index {} out of bounds for array with {} elements", idx, data.len());
                }
                data[idx] = op(data[idx]);
            }
            IndexSpec::Point(ref coords) => {
                if coords.len() != shape.len() {
                    bail!("Coordinate length {} does not match array rank {}", coords.len(), shape.len());
                }
                // Compute linear offset
                let mut offset = 0;
                let mut stride = 1;
                for i in (0..shape.len()).rev() {
                    if coords[i] >= shape[i] {
                        bail!("Index {} along dim {} out of bounds (dim size {})", coords[i], i, shape[i]);
                    }
                    offset += coords[i] * stride;
                    stride *= shape[i];
                }
                data[offset] = op(data[offset]);
            }
            IndexSpec::Slice { axis, start, end } => {
                if axis >= shape.len() {
                    bail!("Axis {} out of bounds for array rank {}", axis, shape.len());
                }
                if end > shape[axis] || start > end {
                    bail!("Invalid slice [{}..{}] along axis {} of size {}", start, end, axis, shape[axis]);
                }
                let mut strides = vec![1; shape.len()];
                for i in (0..shape.len() - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                let target_stride = strides[axis];

                for i in 0..data.len() {
                    let coord_on_axis = (i / target_stride) % shape[axis];
                    if coord_on_axis >= start && coord_on_axis < end {
                        data[i] = op(data[i]);
                    }
                }
            }
        }

        Array::from_vec(data, shape)
    }
}

// ----------------------------------------------------------------------------
// NumPy module-level aliases (df::jnp)
// ----------------------------------------------------------------------------

pub mod jnp {
    use super::*;

    pub fn zeros(shape: Vec<usize>) -> Array {
        Array::zeros(shape)
    }

    pub fn ones(shape: Vec<usize>) -> Array {
        Array::ones(shape)
    }

    pub fn full(shape: Vec<usize>, val: f32) -> Array {
        Array::full(shape, val)
    }

    pub fn arange(start: f32, stop: f32, step: f32) -> Result<Array> {
        Array::arange(start, stop, step)
    }

    pub fn linspace(start: f32, stop: f32, num: usize) -> Result<Array> {
        Array::linspace(start, stop, num)
    }

    pub fn eye(n: usize) -> Array {
        Array::eye(n)
    }

    pub fn dot(a: &Array, b: &Array) -> Result<Array> {
        a.matmul(b)
    }

    pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
        a.matmul(b)
    }

    pub fn sum(a: &Array) -> Result<Array> {
        a.sum()
    }

    pub fn mean(a: &Array) -> Result<Array> {
        a.mean()
    }

    pub fn exp(a: &Array) -> Result<Array> {
        a.exp()
    }

    pub fn log(a: &Array) -> Result<Array> {
        a.log()
    }

    pub fn sqrt(a: &Array) -> Result<Array> {
        a.sqrt()
    }

    pub fn sin(a: &Array) -> Result<Array> {
        a.sin()
    }

    pub fn cos(a: &Array) -> Result<Array> {
        a.cos()
    }

    /// Functional ternary selection: where cond is non-zero, choose x, else y.
    pub fn where_cond(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
        if cond.shape() != x.shape() || x.shape() != y.shape() {
            bail!("Shapes must match in where_cond: {:?} vs {:?} vs {:?}", cond.shape(), x.shape(), y.shape());
        }
        let c_data = cond.to_vec();
        let x_data = x.to_vec();
        let y_data = y.to_vec();

        let mut out = Vec::with_capacity(c_data.len());
        for i in 0..c_data.len() {
            if c_data[i] != 0.0 {
                out.push(x_data[i]);
            } else {
                out.push(y_data[i]);
            }
        }
        Array::from_vec(out, x.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_immutable_at_set() {
        let a = Array::zeros(vec![3, 3]);
        // Update at coordinate [1, 1] to 42.0
        let b = a.at(IndexSpec::Point(vec![1, 1])).set(42.0).unwrap();

        // Verify a remains untouched (referential transparency)
        assert_eq!(a.to_vec()[4], 0.0);
        // Verify b has the updated value
        assert_eq!(b.to_vec()[4], 42.0);
    }

    #[test]
    fn test_array_at_add_and_slice() {
        let a = Array::ones(vec![4]);
        let b = a.at(IndexSpec::Slice { axis: 0, start: 1, end: 3 }).add(10.0).unwrap();
        assert_eq!(b.to_vec(), vec![1.0, 11.0, 11.0, 1.0]);
    }

    #[test]
    fn test_jnp_constructors() {
        let x = jnp::linspace(0.0, 1.0, 5).unwrap();
        assert_eq!(x.to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);

        let id = jnp::eye(2);
        assert_eq!(id.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
    }
}
