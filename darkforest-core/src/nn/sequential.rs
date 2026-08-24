//! Sequential container for cascading neural network modules.

use crate::autograd::Value;
use anyhow::Result;

pub trait Module: Send + Sync {
    fn forward(&self, x: &Value) -> Result<Value>;
    fn parameters(&self) -> Vec<Value> {
        vec![]
    }
}

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: vec![] }
    }

    pub fn add<M: Module + 'static>(&mut self, module: M) {
        self.layers.push(Box::new(module));
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        let mut cur = x.clone();
        for layer in &self.layers {
            cur = layer.forward(&cur)?;
        }
        Ok(cur)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = vec![];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p
    }
}
