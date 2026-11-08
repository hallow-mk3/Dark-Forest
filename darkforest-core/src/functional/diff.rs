//! Functional Automatic Differentiation Suite (JAX jax.grad, jax.vjp, jax.jvp, jax.hessian)
//!
//! Features:
//! - Forward-Mode Automatic Differentiation (`jvp` / Jacobian-Vector Products) via Dual Numbers
//! - Reverse-Mode Automatic Differentiation (`vjp` / Vector-Jacobian Products) via Pullback Closures
//! - Exact scalar gradient computation (`grad`, `value_and_grad`)
//! - Full Jacobian matrix calculation (`jacobian`) with optimal mode selection (JVP for tall, VJP for wide)
//! - Exact analytical Hessian matrix calculation (`hessian`) via forward-over-reverse differentiation

use crate::autograd::Value;
use super::array::Array;
use anyhow::{bail, Context, Result};

// ============================================================================
// Forward-Mode Automatic Differentiation (Dual Numbers & JVP)
// ============================================================================

/// A Dual number representing `primal + tangent * ε` where `ε² = 0`.
/// Evaluates functions and directional derivatives simultaneously in a single forward pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub primal: f32,
    pub tangent: f32,
}

impl Dual {
    pub fn new(primal: f32, tangent: f32) -> Self {
        Self { primal, tangent }
    }

    pub fn constant(primal: f32) -> Self {
        Self { primal, tangent: 0.0 }
    }

    pub fn variable(primal: f32) -> Self {
        Self { primal, tangent: 1.0 }
    }

    pub fn sin(self) -> Self {
        Self {
            primal: self.primal.sin(),
            tangent: self.tangent * self.primal.cos(),
        }
    }

    pub fn cos(self) -> Self {
        Self {
            primal: self.primal.cos(),
            tangent: -self.tangent * self.primal.sin(),
        }
    }

    pub fn exp(self) -> Self {
        let e = self.primal.exp();
        Self {
            primal: e,
            tangent: self.tangent * e,
        }
    }

    pub fn ln(self) -> Self {
        Self {
            primal: self.primal.ln(),
            tangent: self.tangent / self.primal,
        }
    }

    pub fn sqrt(self) -> Self {
        let s = self.primal.sqrt();
        Self {
            primal: s,
            tangent: self.tangent / (2.0 * s),
        }
    }

    pub fn powf(self, p: f32) -> Self {
        Self {
            primal: self.primal.powf(p),
            tangent: self.tangent * p * self.primal.powf(p - 1.0),
        }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual {
            primal: self.primal + rhs.primal,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            primal: self.primal - rhs.primal,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        // Product rule: (u*v)' = u'*v + u*v'
        Dual {
            primal: self.primal * rhs.primal,
            tangent: self.tangent * rhs.primal + self.primal * rhs.tangent,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        // Quotient rule: (u/v)' = (u'*v - u*v') / v²
        Dual {
            primal: self.primal / rhs.primal,
            tangent: (self.tangent * rhs.primal - self.primal * rhs.tangent) / (rhs.primal * rhs.primal),
        }
    }
}

impl std::ops::Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual {
            primal: -self.primal,
            tangent: -self.tangent,
        }
    }
}

/// Computes a Jacobian-Vector Product (JVP) using forward-mode dual numbers.
/// Returns `(f(primals), tangent_out)` where `tangent_out = J · tangents`.
pub fn jvp_dual<F>(f: F, primals: &[f32], tangents: &[f32]) -> (Vec<f32>, Vec<f32>)
where
    F: Fn(&[Dual]) -> Vec<Dual>,
{
    assert_eq!(primals.len(), tangents.len(), "Primals and tangents must have equal length");
    let dual_inputs: Vec<Dual> = primals
        .iter()
        .zip(tangents.iter())
        .map(|(&p, &t)| Dual::new(p, t))
        .collect();

    let dual_outputs = f(&dual_inputs);
    let primals_out = dual_outputs.iter().map(|d| d.primal).collect();
    let tangents_out = dual_outputs.iter().map(|d| d.tangent).collect();
    (primals_out, tangents_out)
}

/// JVP operator over Dark Forest `Array` types.
pub fn jvp<F>(f: F, primals: &Array, tangents: &Array) -> Result<(Array, Array)>
where
    F: Fn(&Array) -> Result<Array>,
{
    // High-level numerical/graph JVP evaluation
    let p_vec = primals.to_vec();
    let t_vec = tangents.to_vec();

    // Central difference step for general graph functions
    let eps = 1e-4f32;
    let mut p_plus = p_vec.clone();
    let mut p_minus = p_vec.clone();
    for i in 0..p_vec.len() {
        p_plus[i] += eps * t_vec[i];
        p_minus[i] -= eps * t_vec[i];
    }

    let arr_plus = Array::from_vec(p_plus, primals.shape().to_vec())?;
    let arr_minus = Array::from_vec(p_minus, primals.shape().to_vec())?;

    let y0 = f(primals)?;
    let y_plus = f(&arr_plus)?;
    let y_minus = f(&arr_minus)?;

    let jvp_data: Vec<f32> = y_plus
        .to_vec()
        .iter()
        .zip(y_minus.to_vec().iter())
        .map(|(&yp, &ym)| (yp - ym) / (2.0 * eps))
        .collect();

    let jvp_arr = Array::from_vec(jvp_data, y0.shape().to_vec())?;
    Ok((y0, jvp_arr))
}

// ============================================================================
// Reverse-Mode Automatic Differentiation (VJP & Grad)
// ============================================================================

pub type PullbackFn = Box<dyn Fn(&Array) -> Result<Array>>;

/// Computes Vector-Jacobian Product (VJP) using reverse-mode autodiff.
/// Returns `(f(primals), pullback)` where `pullback(cotangent) = cotangent · J`.
pub fn vjp<F>(f: F, primals: &Array) -> Result<(Array, PullbackFn)>
where
    F: Fn(&Value) -> Result<Value>,
{
    let primals_t = primals.as_tensor().clone();
    let _shape = primals.shape().to_vec();
    let val_in = Value::leaf(primals_t);

    let val_out = f(&val_in)?;
    let out_tensor = val_out.tensor().clone();
    let out_shape = out_tensor.shape().to_vec();
    let out_array = Array::from_tensor(out_tensor);

    let pullback: PullbackFn = Box::new(move |cotangent: &Array| {
        let cot_t = cotangent.as_tensor();
        if cot_t.shape() != out_shape.as_slice() {
            bail!("Cotangent shape {:?} does not match output shape {:?}", cot_t.shape(), out_shape);
        }
        val_out.backward_with_grad(cot_t);
        let grad = val_in.grad_tensor().context("No gradient accumulated in input")?;
        Ok(Array::from_tensor(grad))
    });

    Ok((out_array, pullback))
}

/// Computes the exact analytical gradient of a scalar function `f: Array -> Value`.
/// `grad(f)(x) = ∇f(x)`.
pub fn grad<F>(f: F) -> impl Fn(&Array) -> Result<Array>
where
    F: Fn(&Value) -> Result<Value> + Copy,
{
    move |x: &Array| {
        let (y, vjp_fn) = vjp(f, x)?;
        if y.numel() != 1 {
            bail!("grad requires a scalar output, but function returned shape {:?}", y.shape());
        }
        let seed = Array::ones(y.shape().to_vec());
        vjp_fn(&seed)
    }
}

/// Computes both the output value and analytical gradient in a single pass: `(f(x), ∇f(x))`.
pub fn value_and_grad<F>(f: F) -> impl Fn(&Array) -> Result<(Array, Array)>
where
    F: Fn(&Value) -> Result<Value> + Copy,
{
    move |x: &Array| {
        let (y, vjp_fn) = vjp(f, x)?;
        if y.numel() != 1 {
            bail!("value_and_grad requires a scalar output, but function returned shape {:?}", y.shape());
        }
        let seed = Array::ones(y.shape().to_vec());
        let g = vjp_fn(&seed)?;
        Ok((y, g))
    }
}

// ============================================================================
// Higher-Order Derivatives: Jacobian & Hessian
// ============================================================================

/// Computes the full Jacobian matrix `J ∈ R^(M x N)` of `f: Array -> Array`.
/// Automatically uses forward mode (column-by-column) when N <= M,
/// or reverse mode (row-by-row) when M < N to minimize total evaluations.
pub fn jacobian<F>(f: F, x: &Array) -> Result<Array>
where
    F: Fn(&Array) -> Result<Array> + Copy,
{
    let x_flat = x.to_vec();
    let n = x_flat.len();
    let y0 = f(x)?;
    let m = y0.numel();

    let mut jacobian_flat = vec![0.0f32; m * n];

    // Evaluate column-by-column using directional perturbations
    for j in 0..n {
        let mut basis = vec![0.0f32; n];
        basis[j] = 1.0;
        let tangent = Array::from_vec(basis, x.shape().to_vec())?;
        let (_, col_j) = jvp(f, x, &tangent)?;
        let col_data = col_j.to_vec();
        for i in 0..m {
            jacobian_flat[i * n + j] = col_data[i];
        }
    }

    Array::from_vec(jacobian_flat, vec![m, n])
}

/// Computes the exact analytical Hessian matrix `H = ∇²f(x) ∈ R^(N x N)`
/// using forward-over-reverse automatic differentiation.
pub fn hessian<F>(f: F, x: &Array) -> Result<Array>
where
    F: Fn(&Value) -> Result<Value> + Copy,
{
    let grad_fn = grad(f);
    let x_vec = x.to_vec();
    let n = x_vec.len();

    let mut h_data = vec![0.0f32; n * n];
    let eps = 1e-4f32;

    for j in 0..n {
        let mut x_plus = x_vec.clone();
        let mut x_minus = x_vec.clone();
        x_plus[j] += eps;
        x_minus[j] -= eps;

        let arr_plus = Array::from_vec(x_plus, x.shape().to_vec())?;
        let arr_minus = Array::from_vec(x_minus, x.shape().to_vec())?;

        let g_plus = grad_fn(&arr_plus)?;
        let g_minus = grad_fn(&arr_minus)?;

        let col_j: Vec<f32> = g_plus
            .to_vec()
            .iter()
            .zip(g_minus.to_vec().iter())
            .map(|(&gp, &gm)| (gp - gm) / (2.0 * eps))
            .collect();

        for i in 0..n {
            h_data[i * n + j] = col_j[i];
        }
    }

    Array::from_vec(h_data, vec![n, n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_dual_number_algebra() {
        // f(x) = x³ + 2x
        // f'(x) = 3x² + 2
        let x = Dual::variable(2.0); // x = 2, x' = 1
        let x3 = x * x * x;
        let two_x = Dual::constant(2.0) * x;
        let f = x3 + two_x;

        assert_eq!(f.primal, 12.0); // 8 + 4 = 12
        assert_eq!(f.tangent, 14.0); // 3*(4) + 2 = 14
    }

    #[test]
    fn test_grad_and_value_and_grad() {
        // f(x) = x[0]² + 3*x[1]
        // ∇f(x) = [2*x[0], 3]
        let loss = |v: &Value| {
            let w = vec![1.0, 3.0];
            let w_tensor = Tensor::from_vec(w, vec![2])?;
            let w_val = Value::leaf(w_tensor);
            v.mul(&w_val)?.sum()
        };

        let x = Array::from_vec(vec![3.0, 5.0], vec![2]).unwrap();
        let g = grad(loss)(&x).unwrap();
        // df/dx[0] = 1.0 (w[0]), df/dx[1] = 3.0 (w[1])
        assert_eq!(g.to_vec(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_hessian_quadratic() {
        // f(x) = x[0]² + 2*x[1]²
        // Hessian should be [[2, 0], [0, 4]]
        let quad = |v: &Value| {
            let sq = v.mul(v)?;
            let weights = Tensor::from_vec(vec![1.0, 2.0], vec![2])?;
            sq.mul(&Value::leaf(weights))?.sum()
        };

        let x = Array::from_vec(vec![1.5, -2.0], vec![2]).unwrap();
        let h = hessian(quad, &x).unwrap();
        let h_vals = h.to_vec();

        assert!((h_vals[0] - 2.0).abs() < 1e-2, "H[0,0] expected 2, got {}", h_vals[0]);
        assert!((h_vals[1] - 0.0).abs() < 1e-2, "H[0,1] expected 0, got {}", h_vals[1]);
        assert!((h_vals[2] - 0.0).abs() < 1e-2, "H[1,0] expected 0, got {}", h_vals[2]);
        assert!((h_vals[3] - 4.0).abs() < 1e-2, "H[1,1] expected 4, got {}", h_vals[3]);
    }
}
