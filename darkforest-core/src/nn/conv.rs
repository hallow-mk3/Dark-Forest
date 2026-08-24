//! Conv1d and Conv2d neural network layers.

use crate::autograd::Value;
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};

pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub weight: Value,       // [out_channels, in_channels, kH, kW]
    pub bias: Option<Value>, // [out_channels]
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let (kh, kw) = kernel_size;
        let fan_in = in_channels * kh * kw;
        let std = (2.0 / fan_in as f32).sqrt();
        let w = Tensor::randn(vec![out_channels, in_channels, kh, kw], std);
        let weight = Value::leaf(w);

        let b = if bias {
            Some(Value::leaf(Tensor::zeros(vec![out_channels])))
        } else {
            None
        };

        Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias: b,
        }
    }

    /// Forward pass for Conv2d: x shape [batch, in_channels, H, W]
    pub fn forward(&self, x: &Value) -> Result<Value> {
        let xt = x.tensor();
        if xt.ndim() != 4 {
            return Err(anyhow!("Conv2d expects 4D tensor [N, C, H, W]"));
        }
        let (batch, in_c, h, w) = (xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]);
        if in_c != self.in_channels {
            return Err(anyhow!("Conv2d expected {} in_channels, got {}", self.in_channels, in_c));
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        let x_data = xt.to_vec();
        let w_data = self.weight.tensor().to_vec();
        let mut out_data = vec![0.0f32; batch * self.out_channels * out_h * out_w];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let ih_start = (oh * sh) as isize - ph as isize;
                        let iw_start = (ow * sw) as isize - pw as isize;

                        for ic in 0..in_c {
                            for f_h in 0..kh {
                                for f_w in 0..kw {
                                    let cur_h = ih_start + f_h as isize;
                                    let cur_w = iw_start + f_w as isize;

                                    if cur_h >= 0 && cur_h < h as isize && cur_w >= 0 && cur_w < w as isize {
                                        let in_idx = b * (in_c * h * w) + ic * (h * w) + (cur_h as usize) * w + (cur_w as usize);
                                        let w_idx = oc * (in_c * kh * kw) + ic * (kh * kw) + f_h * kw + f_w;
                                        sum += x_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        let out_idx = b * (self.out_channels * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }

        let mut out_val = Value::leaf(Tensor::from_vec_device(
            out_data,
            vec![batch, self.out_channels, out_h, out_w],
            xt.device.clone(),
        )?);

        if let Some(ref b) = self.bias {
            let b_data = b.tensor().to_vec();
            let mut v_data = out_val.tensor().to_vec();
            for nb in 0..batch {
                for oc in 0..self.out_channels {
                    let bias_val = b_data[oc];
                    for p in 0..(out_h * out_w) {
                        let idx = nb * (self.out_channels * out_h * out_w) + oc * (out_h * out_w) + p;
                        v_data[idx] += bias_val;
                    }
                }
            }
            out_val = Value::leaf(Tensor::from_vec_device(
                v_data,
                vec![batch, self.out_channels, out_h, out_w],
                xt.device,
            )?);
        }

        Ok(out_val)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}

pub struct Conv1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weight: Value,       // [out_channels, in_channels, kernel_size]
    pub bias: Option<Value>, // [out_channels]
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let std = (2.0 / fan_in as f32).sqrt();
        let w = Tensor::randn(vec![out_channels, in_channels, kernel_size], std);
        let weight = Value::leaf(w);

        let b = if bias {
            Some(Value::leaf(Tensor::zeros(vec![out_channels])))
        } else {
            None
        };

        Conv1d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias: b,
        }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        let xt = x.tensor();
        if xt.ndim() != 3 {
            return Err(anyhow!("Conv1d expects 3D tensor [N, C, L]"));
        }
        let (batch, in_c, l) = (xt.shape[0], xt.shape[1], xt.shape[2]);
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let out_l = (l + 2 * p - k) / s + 1;

        let x_data = xt.to_vec();
        let w_data = self.weight.tensor().to_vec();
        let mut out_data = vec![0.0f32; batch * self.out_channels * out_l];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for ol in 0..out_l {
                    let mut sum = 0.0f32;
                    let il_start = (ol * s) as isize - p as isize;
                    for ic in 0..in_c {
                        for f_k in 0..k {
                            let cur_l = il_start + f_k as isize;
                            if cur_l >= 0 && cur_l < l as isize {
                                let in_idx = b * (in_c * l) + ic * l + cur_l as usize;
                                let w_idx = oc * (in_c * k) + ic * k + f_k;
                                sum += x_data[in_idx] * w_data[w_idx];
                            }
                        }
                    }
                    out_data[b * (self.out_channels * out_l) + oc * out_l + ol] = sum;
                }
            }
        }

        let mut out_val = Value::leaf(Tensor::from_vec_device(
            out_data,
            vec![batch, self.out_channels, out_l],
            xt.device.clone(),
        )?);

        if let Some(ref b) = self.bias {
            let b_data = b.tensor().to_vec();
            let mut v_data = out_val.tensor().to_vec();
            for nb in 0..batch {
                for oc in 0..self.out_channels {
                    let bias_val = b_data[oc];
                    for ol in 0..out_l {
                        v_data[nb * (self.out_channels * out_l) + oc * out_l + ol] += bias_val;
                    }
                }
            }
            out_val = Value::leaf(Tensor::from_vec_device(
                v_data,
                vec![batch, self.out_channels, out_l],
                xt.device,
            )?);
        }

        Ok(out_val)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}
