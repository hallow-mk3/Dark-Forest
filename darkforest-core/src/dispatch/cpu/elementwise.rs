use anyhow::Result;
use crate::dispatch::iterator::TensorIterator;
use crate::tensor::Tensor;

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    TensorIterator::binary_f32(a, b, |x, y| x + y)
}

pub fn add_(a: &mut Tensor, b: &Tensor) -> Result<()> {
    TensorIterator::binary_inplace_f32(a, b, |x, y| x + y)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    TensorIterator::binary_f32(a, b, |x, y| x - y)
}

pub fn sub_(a: &mut Tensor, b: &Tensor) -> Result<()> {
    TensorIterator::binary_inplace_f32(a, b, |x, y| x - y)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    TensorIterator::binary_f32(a, b, |x, y| x * y)
}

pub fn mul_(a: &mut Tensor, b: &Tensor) -> Result<()> {
    TensorIterator::binary_inplace_f32(a, b, |x, y| x * y)
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    TensorIterator::binary_f32(a, b, |x, y| x / y)
}

pub fn div_(a: &mut Tensor, b: &Tensor) -> Result<()> {
    TensorIterator::binary_inplace_f32(a, b, |x, y| x / y)
}

pub fn neg(a: &Tensor) -> Result<Tensor> {
    TensorIterator::unary_f32(a, |x| -x)
}

pub fn relu(a: &Tensor) -> Result<Tensor> {
    TensorIterator::unary_f32(a, |x| if x > 0.0 { x } else { 0.0 })
}
