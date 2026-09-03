//! Stateful activation modules implementing Module.

use crate::autograd::Value;
use crate::nn::sequential::Module;
use anyhow::Result;

pub struct ReLU;
impl Module for ReLU {
    fn forward(&self, x: &Value) -> Result<Value> {
        let data: Vec<f32> = x.tensor().to_vec().iter().map(|&v| v.max(0.0)).collect();
        Ok(Value::leaf(crate::tensor::Tensor::from_vec_device(
            data,
            x.tensor().shape.clone(),
            x.tensor().device,
        )?))
    }
}

pub struct GELU;
impl Module for GELU {
    fn forward(&self, x: &Value) -> Result<Value> {
        x.gelu()
    }
}

pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, x: &Value) -> Result<Value> {
        let data: Vec<f32> = x
            .tensor()
            .to_vec()
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        Ok(Value::leaf(crate::tensor::Tensor::from_vec_device(
            data,
            x.tensor().shape.clone(),
            x.tensor().device,
        )?))
    }
}

pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, x: &Value) -> Result<Value> {
        let data: Vec<f32> = x.tensor().to_vec().iter().map(|&v| v.tanh()).collect();
        Ok(Value::leaf(crate::tensor::Tensor::from_vec_device(
            data,
            x.tensor().shape.clone(),
            x.tensor().device,
        )?))
    }
}

pub struct SiLU;
impl Module for SiLU {
    fn forward(&self, x: &Value) -> Result<Value> {
        let data: Vec<f32> = x
            .tensor()
            .to_vec()
            .iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect();
        Ok(Value::leaf(crate::tensor::Tensor::from_vec_device(
            data,
            x.tensor().shape.clone(),
            x.tensor().device,
        )?))
    }
}

pub struct LeakyReLU {
    pub negative_slope: f32,
}
impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        LeakyReLU { negative_slope }
    }
}
impl Module for LeakyReLU {
    fn forward(&self, x: &Value) -> Result<Value> {
        let ns = self.negative_slope;
        let data: Vec<f32> = x
            .tensor()
            .to_vec()
            .iter()
            .map(|&v| if v >= 0.0 { v } else { ns * v })
            .collect();
        Ok(Value::leaf(crate::tensor::Tensor::from_vec_device(
            data,
            x.tensor().shape.clone(),
            x.tensor().device,
        )?))
    }
}
