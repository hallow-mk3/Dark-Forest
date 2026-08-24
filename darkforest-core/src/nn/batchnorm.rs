//! BatchNorm1d and BatchNorm2d modules.

use crate::autograd::Value;
use crate::ops::norm::{batch_norm, BatchNormStats};
use crate::tensor::Tensor;
use anyhow::Result;

pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub weight: Value, // gamma
    pub bias: Value,   // beta
    pub stats: BatchNormStats,
    pub training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        BatchNorm2d {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            weight: Value::leaf(Tensor::ones(vec![num_features])),
            bias: Value::leaf(Tensor::zeros(vec![num_features])),
            stats: BatchNormStats {
                running_mean: vec![0.0; num_features],
                running_var: vec![1.0; num_features],
                momentum: 0.1,
            },
            training: true,
        }
    }

    pub fn forward(&mut self, x: &Value) -> Result<Value> {
        let xt = x.tensor();
        let (out, _, _) = batch_norm(
            &xt,
            &self.weight.tensor(),
            &self.bias.tensor(),
            Some(&mut self.stats),
            self.eps,
            self.training,
        )?;
        Ok(Value::leaf(out))
    }

    pub fn parameters(&self) -> Vec<Value> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct BatchNorm1d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub weight: Value,
    pub bias: Value,
    pub stats: BatchNormStats,
    pub training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        BatchNorm1d {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            weight: Value::leaf(Tensor::ones(vec![num_features])),
            bias: Value::leaf(Tensor::zeros(vec![num_features])),
            stats: BatchNormStats {
                running_mean: vec![0.0; num_features],
                running_var: vec![1.0; num_features],
                momentum: 0.1,
            },
            training: true,
        }
    }

    pub fn forward(&mut self, x: &Value) -> Result<Value> {
        let xt = x.tensor();
        let (out, _, _) = batch_norm(
            &xt,
            &self.weight.tensor(),
            &self.bias.tensor(),
            Some(&mut self.stats),
            self.eps,
            self.training,
        )?;
        Ok(Value::leaf(out))
    }

    pub fn parameters(&self) -> Vec<Value> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
