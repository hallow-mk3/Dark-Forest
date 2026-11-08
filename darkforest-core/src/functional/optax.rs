//! Functional Optimization & Gradient Transformations (JAX Optax equivalent)
//!
//! Features:
//! - Pure functional optimizer updates: `init`, `update`, `apply_updates`
//! - Composable gradient transformations: `clip_by_global_norm`, `add_decayed_weights`, `scale_by_adam`
//! - Full functional AdamW optimizer pipeline
//! - Referential transparency with zero hidden mutable state

use super::array::Array;
use anyhow::Result;

/// State for an Adam/AdamW optimizer holding first and second moments.
#[derive(Clone, Debug)]
pub struct AdamState {
    pub count: usize,
    pub m: Vec<Array>,
    pub v: Vec<Array>,
}

/// An AdamW functional optimizer.
#[derive(Clone, Debug)]
pub struct AdamWOpt {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_norm: Option<f32>,
}

impl AdamWOpt {
    pub fn new(lr: f32, b1: f32, b2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            b1,
            b2,
            eps,
            weight_decay,
            max_norm: None,
        }
    }

    pub fn with_clip(mut self, max_norm: f32) -> Self {
        self.max_norm = Some(max_norm);
        self
    }

    /// Initializes optimizer state for a given parameter list.
    pub fn init(&self, params: &[Array]) -> AdamState {
        let m = params.iter().map(|p| Array::zeros(p.shape().to_vec())).collect();
        let v = params.iter().map(|p| Array::zeros(p.shape().to_vec())).collect();
        AdamState { count: 0, m, v }
    }

    /// Computes pure parameter updates and the next optimizer state.
    pub fn update(
        &self,
        grads: &[Array],
        state: &AdamState,
        params: &[Array],
    ) -> Result<(Vec<Array>, AdamState)> {
        let next_count = state.count + 1;
        let t = next_count as f32;

        // Bias correction terms
        let bias_correction1 = 1.0 - self.b1.powi(t as i32);
        let bias_correction2 = 1.0 - self.b2.powi(t as i32);

        // Optional global norm clipping
        let clipped_grads = if let Some(max_n) = self.max_norm {
            clip_by_global_norm(grads, max_n)?
        } else {
            grads.to_vec()
        };

        let mut updates = Vec::with_capacity(params.len());
        let mut next_m = Vec::with_capacity(params.len());
        let mut next_v = Vec::with_capacity(params.len());

        for i in 0..params.len() {
            let g = &clipped_grads[i];
            let m_prev = &state.m[i];
            let v_prev = &state.v[i];
            let p = &params[i];

            let g_vec = g.to_vec();
            let m_vec = m_prev.to_vec();
            let v_vec = v_prev.to_vec();
            let p_vec = p.to_vec();

            let mut m_new_data = Vec::with_capacity(g_vec.len());
            let mut v_new_data = Vec::with_capacity(g_vec.len());
            let mut u_data = Vec::with_capacity(g_vec.len());

            for j in 0..g_vec.len() {
                let gj = g_vec[j];
                let mj = self.b1 * m_vec[j] + (1.0 - self.b1) * gj;
                let vj = self.b2 * v_vec[j] + (1.0 - self.b2) * (gj * gj);

                m_new_data.push(mj);
                v_new_data.push(vj);

                let m_hat = mj / bias_correction1;
                let v_hat = vj / bias_correction2;

                // Adam update step with decoupled weight decay:
                // u = -lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
                let step = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p_vec[j];
                u_data.push(-self.lr * step);
            }

            let shape = p.shape().to_vec();
            updates.push(Array::from_vec(u_data, shape.clone())?);
            next_m.push(Array::from_vec(m_new_data, shape.clone())?);
            next_v.push(Array::from_vec(v_new_data, shape)?);
        }

        let next_state = AdamState {
            count: next_count,
            m: next_m,
            v: next_v,
        };

        Ok((updates, next_state))
    }

    /// Purely applies updates to parameters: `new_params = params + updates`.
    pub fn apply_updates(&self, params: &[Array], updates: &[Array]) -> Result<Vec<Array>> {
        params
            .iter()
            .zip(updates.iter())
            .map(|(p, u)| p.add(u))
            .collect()
    }
}

/// Computes global L2 norm of gradients and clips them if exceeding `max_norm`.
pub fn clip_by_global_norm(grads: &[Array], max_norm: f32) -> Result<Vec<Array>> {
    let mut sum_sq = 0.0f64;
    for g in grads {
        for &val in &g.to_vec() {
            sum_sq += (val as f64) * (val as f64);
        }
    }

    let global_norm = sum_sq.sqrt() as f32;
    if global_norm <= max_norm || global_norm < 1e-12 {
        return Ok(grads.to_vec());
    }

    let scale = max_norm / global_norm;
    grads.iter().map(|g| g.scale(scale)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optax_adamw_step() {
        let opt = AdamWOpt::new(0.01, 0.9, 0.999, 1e-8, 0.01);
        let params = vec![Array::from_vec(vec![1.0, 2.0], vec![2]).unwrap()];
        let state = opt.init(&params);

        let grads = vec![Array::from_vec(vec![0.1, -0.2], vec![2]).unwrap()];
        let (updates, next_state) = opt.update(&grads, &state, &params).unwrap();
        let new_params = opt.apply_updates(&params, &updates).unwrap();

        assert_eq!(next_state.count, 1);
        let p_new = new_params[0].to_vec();
        // Since grads[0] > 0, update should decrease param[0]
        assert!(p_new[0] < 1.0);
        // Since grads[1] < 0, update should increase param[1]
        assert!(p_new[1] > 2.0);
    }
}

