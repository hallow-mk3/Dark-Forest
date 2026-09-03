//! Recurrent Neural Network layers: RNNCell, LSTMCell, GRUCell, LSTM, GRU.

use crate::autograd::Value;
use crate::nn::Linear;
use anyhow::Result;

pub struct RNNCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub fc_ih: Linear,
    pub fc_hh: Linear,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        RNNCell {
            input_size,
            hidden_size,
            fc_ih: Linear::new(input_size, hidden_size, true),
            fc_hh: Linear::new(hidden_size, hidden_size, true),
        }
    }

    pub fn forward(&self, x: &Value, h: &Value) -> Result<Value> {
        let x_out = self.fc_ih.forward(x)?;
        let h_out = self.fc_hh.forward(h)?;
        let sum = x_out.add(&h_out)?;
        // Tanh activation
        let t_data: Vec<f32> = sum.tensor().to_vec().iter().map(|&v| v.tanh()).collect();
        Ok(Value::leaf(crate::tensor::Tensor::from_vec_device(
            t_data,
            sum.tensor().shape.clone(),
            sum.tensor().device,
        )?))
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.fc_ih.parameters();
        p.extend(self.fc_hh.parameters());
        p
    }
}

pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub fc_ih: Linear,
    pub fc_hh: Linear,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LSTMCell {
            input_size,
            hidden_size,
            fc_ih: Linear::new(input_size, 4 * hidden_size, true),
            fc_hh: Linear::new(hidden_size, 4 * hidden_size, true),
        }
    }

    pub fn forward(&self, x: &Value, hx: &(Value, Value)) -> Result<(Value, Value)> {
        let (ref h, ref c) = hx;
        let gates = self.fc_ih.forward(x)?.add(&self.fc_hh.forward(h)?)?;
        let g_data = gates.tensor().to_vec();
        let hs = self.hidden_size;

        let c_data = c.tensor().to_vec();
        let mut next_h = vec![0.0f32; hs];
        let mut next_c = vec![0.0f32; hs];

        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let tan = |v: f32| v.tanh();

        for i in 0..hs {
            let i_gate = sig(g_data[i]);
            let f_gate = sig(g_data[hs + i]);
            let g_gate = tan(g_data[2 * hs + i]);
            let o_gate = sig(g_data[3 * hs + i]);

            let new_c_val = f_gate * c_data[i] + i_gate * g_gate;
            let new_h_val = o_gate * tan(new_c_val);

            next_c[i] = new_c_val;
            next_h[i] = new_h_val;
        }

        let dev = x.tensor().device;
        let v_h = Value::leaf(crate::tensor::Tensor::from_vec_device(
            next_h,
            vec![1, hs],
            dev.clone(),
        )?);
        let v_c = Value::leaf(crate::tensor::Tensor::from_vec_device(
            next_c,
            vec![1, hs],
            dev,
        )?);
        Ok((v_h, v_c))
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.fc_ih.parameters();
        p.extend(self.fc_hh.parameters());
        p
    }
}

pub struct GRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub fc_ih: Linear,
    pub fc_hh: Linear,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        GRUCell {
            input_size,
            hidden_size,
            fc_ih: Linear::new(input_size, 3 * hidden_size, true),
            fc_hh: Linear::new(hidden_size, 3 * hidden_size, true),
        }
    }

    pub fn forward(&self, x: &Value, h: &Value) -> Result<Value> {
        let hs = self.hidden_size;
        let x_gates = self.fc_ih.forward(x)?.tensor().to_vec();
        let h_gates = self.fc_hh.forward(h)?.tensor().to_vec();
        let h_data = h.tensor().to_vec();

        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let tan = |v: f32| v.tanh();

        let mut next_h = vec![0.0f32; hs];
        for i in 0..hs {
            let r = sig(x_gates[i] + h_gates[i]);
            let z = sig(x_gates[hs + i] + h_gates[hs + i]);
            let n = tan(x_gates[2 * hs + i] + r * h_gates[2 * hs + i]);
            next_h[i] = (1.0 - z) * n + z * h_data[i];
        }

        Ok(Value::leaf(crate::tensor::Tensor::from_vec_device(
            next_h,
            vec![1, hs],
            x.tensor().device,
        )?))
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.fc_ih.parameters();
        p.extend(self.fc_hh.parameters());
        p
    }
}

pub struct LSTM {
    pub cell: LSTMCell,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LSTM {
            cell: LSTMCell::new(input_size, hidden_size),
        }
    }

    pub fn forward(&self, sequence: &[Value], init: Option<(Value, Value)>) -> Result<Vec<Value>> {
        let mut h = init.unwrap_or_else(|| {
            let hs = self.cell.hidden_size;
            (
                Value::leaf(crate::tensor::Tensor::zeros(vec![1, hs])),
                Value::leaf(crate::tensor::Tensor::zeros(vec![1, hs])),
            )
        });

        let mut outputs = Vec::with_capacity(sequence.len());
        for x in sequence {
            h = self.cell.forward(x, &h)?;
            outputs.push(h.0.clone());
        }
        Ok(outputs)
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.cell.parameters()
    }
}

pub struct GRU {
    pub cell: GRUCell,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        GRU {
            cell: GRUCell::new(input_size, hidden_size),
        }
    }

    pub fn forward(&self, sequence: &[Value], init: Option<Value>) -> Result<Vec<Value>> {
        let mut h = init.unwrap_or_else(|| {
            let hs = self.cell.hidden_size;
            Value::leaf(crate::tensor::Tensor::zeros(vec![1, hs]))
        });

        let mut outputs = Vec::with_capacity(sequence.len());
        for x in sequence {
            h = self.cell.forward(x, &h)?;
            outputs.push(h.clone());
        }
        Ok(outputs)
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.cell.parameters()
    }
}
