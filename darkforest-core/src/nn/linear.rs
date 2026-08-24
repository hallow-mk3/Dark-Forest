//! Linear layer: y = xW^T + b

use crate::autograd::Value;
use crate::tensor::Tensor;
use anyhow::Result;

pub struct Linear {
    pub weight: Value,       // shape [out_features, in_features]
    pub bias: Option<Value>, // shape [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Create with Kaiming uniform initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let std = (2.0 / in_features as f32).sqrt();
        let w = Tensor::randn(vec![out_features, in_features], std);
        let weight = Value::leaf(w);

        let b = if bias {
            let b_t = Tensor::zeros(vec![out_features]);
            Some(Value::leaf(b_t))
        } else {
            None
        };

        Linear {
            weight,
            bias: b,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        // Keep the model path on the stability-first CPU implementation for now.
        // The custom CUDA linear kernel can still be used in isolated tests, but the
        // full transformer graph presently fails when its backward path is invoked on
        // device-backed tensors. Falling back to the CPU matmul + re-upload routine is
        // the correct and safer path until the CUDA backend is fully stabilized.

        // x: [batch..., in_features]
        // weight: [out_features, in_features] -> transpose to [in_features, out_features]
        let w_t = {
            let wt = self.weight.tensor().transpose_last_two()?;
            Value::leaf(wt)
        };
        let out = x.matmul(&w_t)?;

        if let Some(ref b) = self.bias {
            out.add_bias(b)
        } else {
            Ok(out)
        }
    }

    pub fn to_device(&mut self, device: crate::tensor::Device) -> Result<()> {
        self.weight = self.weight.to_device(device.clone())?;
        if let Some(ref b) = self.bias {
            self.bias = Some(b.to_device(device)?);
        }
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}
