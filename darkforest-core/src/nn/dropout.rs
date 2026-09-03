//! Dropout and Dropout2d regularization layers.

use crate::autograd::Value;
use crate::tensor::Tensor;
use anyhow::Result;
use rand::Rng;

pub struct Dropout {
    pub p: f32,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p, training: true }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        if !self.training || self.p <= 0.0 {
            return Ok(x.clone());
        }
        let scale = 1.0 / (1.0 - self.p);
        let mut rng = rand::thread_rng();
        let xt = x.tensor();
        let data: Vec<f32> = xt
            .to_vec()
            .iter()
            .map(|&v| {
                if rng.gen::<f32>() >= self.p {
                    v * scale
                } else {
                    0.0
                }
            })
            .collect();
        Ok(Value::leaf(Tensor::from_vec_device(
            data,
            xt.shape.clone(),
            xt.device,
        )?))
    }
}

pub struct Dropout2d {
    pub p: f32,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        Dropout2d { p, training: true }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        if !self.training || self.p <= 0.0 {
            return Ok(x.clone());
        }
        let xt = x.tensor();
        if xt.ndim() != 4 {
            return Ok(x.clone());
        }
        let (batch, c, h, w) = (xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]);
        let spatial = h * w;
        let scale = 1.0 / (1.0 - self.p);
        let mut rng = rand::thread_rng();

        let src = xt.to_vec();
        let mut out = vec![0.0f32; src.len()];

        for b in 0..batch {
            for ch in 0..c {
                let keep = rng.gen::<f32>() >= self.p;
                let offset = b * c * spatial + ch * spatial;
                for s in 0..spatial {
                    out[offset + s] = if keep { src[offset + s] * scale } else { 0.0 };
                }
            }
        }

        Ok(Value::leaf(Tensor::from_vec_device(
            out,
            xt.shape.clone(),
            xt.device,
        )?))
    }
}
