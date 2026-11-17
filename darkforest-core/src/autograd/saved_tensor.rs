use anyhow::Result;
use crate::tensor::Tensor;

/// SavedTensor saves a Tensor for backward along with its version counter.
/// Calling `unpack()` checks that the tensor was not modified in-place since being saved.
#[derive(Clone, Debug)]
pub struct SavedTensor {
    tensor: Tensor,
    saved_version: usize,
}

impl SavedTensor {
    pub fn new(tensor: &Tensor) -> Self {
        Self {
            tensor: tensor.clone(),
            saved_version: tensor.version(),
        }
    }

    /// Unpack and verify version counter.
    /// Fails with an error if the tensor was mutated in-place.
    pub fn unpack(&self) -> Result<Tensor> {
        self.tensor.check_version(self.saved_version)?;
        Ok(self.tensor.clone())
    }
}
