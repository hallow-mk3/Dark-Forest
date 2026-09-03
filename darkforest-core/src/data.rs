//! Dataset and DataLoader abstractions for batching and dataset shuffling.

use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;

pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)>;
}

pub struct TensorDataset {
    pub inputs: Tensor,
    pub targets: Tensor,
}

impl TensorDataset {
    pub fn new(inputs: Tensor, targets: Tensor) -> Result<Self> {
        if inputs.shape[0] != targets.shape[0] {
            return Err(anyhow!(
                "TensorDataset: inputs and targets batch size mismatch"
            ));
        }
        Ok(TensorDataset { inputs, targets })
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.shape[0]
    }

    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        use crate::ops::narrow;
        let in_sample = narrow(&self.inputs, 0, index, 1)?;
        let tgt_sample = narrow(&self.targets, 0, index, 1)?;
        Ok((in_sample, tgt_sample))
    }
}

pub struct DataLoader<'a> {
    pub dataset: &'a dyn Dataset,
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    indices: Vec<usize>,
    current: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(
        dataset: &'a dyn Dataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> Self {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();
        if shuffle {
            indices.shuffle(&mut rand::thread_rng());
        }
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            indices,
            current: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let total = self.dataset.len();
        if self.current >= total {
            return None;
        }

        let remaining = total - self.current;
        if self.drop_last && remaining < self.batch_size {
            return None;
        }

        let cur_batch_size = self.batch_size.min(remaining);
        let batch_indices = &self.indices[self.current..self.current + cur_batch_size];
        self.current += cur_batch_size;

        let mut in_samples = Vec::with_capacity(cur_batch_size);
        let mut tgt_samples = Vec::with_capacity(cur_batch_size);

        for &idx in batch_indices {
            match self.dataset.get(idx) {
                Ok((inp, tgt)) => {
                    in_samples.push(inp);
                    tgt_samples.push(tgt);
                }
                Err(e) => return Some(Err(e)),
            }
        }

        let in_refs: Vec<&Tensor> = in_samples.iter().collect();
        let tgt_refs: Vec<&Tensor> = tgt_samples.iter().collect();

        match (crate::ops::cat(&in_refs, 0), crate::ops::cat(&tgt_refs, 0)) {
            (Ok(batch_x), Ok(batch_y)) => Some(Ok((batch_x, batch_y))),
            (Err(e), _) | (_, Err(e)) => Some(Err(e)),
        }
    }
}
