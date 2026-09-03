//! Pooling layers: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d.

use crate::autograd::Value;
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};

pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        let xt = x.tensor();
        if xt.ndim() != 4 {
            return Err(anyhow!("MaxPool2d expects 4D tensor [N, C, H, W]"));
        }
        let (batch, c, h, w) = (xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        let x_data = xt.to_vec();
        let mut out_data = vec![f32::NEG_INFINITY; batch * c * out_h * out_w];

        for b in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        let ih_start = (oh * sh) as isize - ph as isize;
                        let iw_start = (ow * sw) as isize - pw as isize;

                        for f_h in 0..kh {
                            for f_w in 0..kw {
                                let cur_h = ih_start + f_h as isize;
                                let cur_w = iw_start + f_w as isize;
                                if cur_h >= 0
                                    && cur_h < h as isize
                                    && cur_w >= 0
                                    && cur_w < w as isize
                                {
                                    let in_idx = b * (c * h * w)
                                        + ch * (h * w)
                                        + (cur_h as usize) * w
                                        + (cur_w as usize);
                                    if x_data[in_idx] > max_val {
                                        max_val = x_data[in_idx];
                                    }
                                }
                            }
                        }
                        let out_idx =
                            b * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                        out_data[out_idx] = if max_val == f32::NEG_INFINITY {
                            0.0
                        } else {
                            max_val
                        };
                    }
                }
            }
        }

        Ok(Value::leaf(Tensor::from_vec_device(
            out_data,
            vec![batch, c, out_h, out_w],
            xt.device,
        )?))
    }
}

pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        AvgPool2d {
            kernel_size,
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        let xt = x.tensor();
        if xt.ndim() != 4 {
            return Err(anyhow!("AvgPool2d expects 4D tensor [N, C, H, W]"));
        }
        let (batch, c, h, w) = (xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        let x_data = xt.to_vec();
        let mut out_data = vec![0.0f32; batch * c * out_h * out_w];

        for b in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;
                        let ih_start = (oh * sh) as isize - ph as isize;
                        let iw_start = (ow * sw) as isize - pw as isize;

                        for f_h in 0..kh {
                            for f_w in 0..kw {
                                let cur_h = ih_start + f_h as isize;
                                let cur_w = iw_start + f_w as isize;
                                if cur_h >= 0
                                    && cur_h < h as isize
                                    && cur_w >= 0
                                    && cur_w < w as isize
                                {
                                    let in_idx = b * (c * h * w)
                                        + ch * (h * w)
                                        + (cur_h as usize) * w
                                        + (cur_w as usize);
                                    sum += x_data[in_idx];
                                }
                                count += 1.0;
                            }
                        }
                        let out_idx =
                            b * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                        out_data[out_idx] = sum / count.max(1.0);
                    }
                }
            }
        }

        Ok(Value::leaf(Tensor::from_vec_device(
            out_data,
            vec![batch, c, out_h, out_w],
            xt.device,
        )?))
    }
}

pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        AdaptiveAvgPool2d { output_size }
    }

    pub fn forward(&self, x: &Value) -> Result<Value> {
        let xt = x.tensor();
        if xt.ndim() != 4 {
            return Err(anyhow!("AdaptiveAvgPool2d expects 4D tensor [N, C, H, W]"));
        }
        let (batch, c, h, w) = (xt.shape[0], xt.shape[1], xt.shape[2], xt.shape[3]);
        let (out_h, out_w) = self.output_size;
        let x_data = xt.to_vec();
        let mut out_data = vec![0.0f32; batch * c * out_h * out_w];

        for b in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    let start_h = (oh * h) / out_h;
                    let end_h = ((oh + 1) * h + out_h - 1) / out_h;

                    for ow in 0..out_w {
                        let start_w = (ow * w) / out_w;
                        let end_w = ((ow + 1) * w + out_w - 1) / out_w;

                        let mut sum = 0.0f32;
                        let mut count = 0usize;

                        for ih in start_h..end_h {
                            for iw in start_w..end_w {
                                let in_idx = b * (c * h * w) + ch * (h * w) + ih * w + iw;
                                sum += x_data[in_idx];
                                count += 1;
                            }
                        }

                        let out_idx =
                            b * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                        out_data[out_idx] = sum / (count.max(1) as f32);
                    }
                }
            }
        }

        Ok(Value::leaf(Tensor::from_vec_device(
            out_data,
            vec![batch, c, out_h, out_w],
            xt.device,
        )?))
    }
}
