//! Debugging, NaN Trapping & Profiling (JAX jax.debug & Nsight equivalent)
//!
//! Features:
//! - Immediate NaN / Inf trapping (`debug_check_nans`) with detailed telemetry
//! - Pure functional inspection callback (`debug_print`) for pipeline debugging
//! - Microsecond-resolution execution profiling (`ProfileTrace`)
//! - Statistical telemetry (mean, std, min, max, finite percentage)

use super::array::Array;
use anyhow::{bail, Result};
use std::time::Instant;

/// Diagnostic report generated when analyzing array values.
#[derive(Clone, Debug)]
pub struct ArrayTelemetry {
    pub shape: Vec<usize>,
    pub numel: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub nan_count: usize,
    pub inf_count: usize,
}

impl ArrayTelemetry {
    pub fn compute(array: &Array) -> Self {
        let data = array.to_vec();
        let numel = data.len();

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut finite_count = 0;

        for &x in &data {
            if x.is_nan() {
                nan_count += 1;
            } else if x.is_infinite() {
                inf_count += 1;
            } else {
                finite_count += 1;
                min = min.min(x);
                max = max.max(x);
                sum += x as f64;
            }
        }

        let mean = if finite_count > 0 {
            (sum / finite_count as f64) as f32
        } else {
            f32::NAN
        };

        Self {
            shape: array.shape().to_vec(),
            numel,
            min: if min.is_infinite() { f32::NAN } else { min },
            max: if max.is_infinite() { f32::NAN } else { max },
            mean,
            nan_count,
            inf_count,
        }
    }

    pub fn is_clean(&self) -> bool {
        self.nan_count == 0 && self.inf_count == 0
    }
}

/// Checks an array for any `NaN` or `Inf` values. If found, returns an error with full diagnostics.
pub fn debug_check_nans(array: &Array, context: &str) -> Result<()> {
    let telemetry = ArrayTelemetry::compute(array);
    if !telemetry.is_clean() {
        bail!(
            "NaN/Inf TRAP in [{}]: shape={:?}, total={}, NaNs={} ({:.2}%), Infs={} ({:.2}%), min={}, max={}, mean={}",
            context,
            telemetry.shape,
            telemetry.numel,
            telemetry.nan_count,
            (telemetry.nan_count as f64 / telemetry.numel as f64) * 100.0,
            telemetry.inf_count,
            (telemetry.inf_count as f64 / telemetry.numel as f64) * 100.0,
            telemetry.min,
            telemetry.max,
            telemetry.mean
        );
    }
    Ok(())
}

/// Prints a concise functional inspection of the array and returns it unchanged.
/// Allows transparent insertion into functional chains without altering logic.
pub fn debug_print(array: &Array, label: &str) -> Array {
    let telemetry = ArrayTelemetry::compute(array);
    let sample_data = array.to_vec();
    let preview_len = sample_data.len().min(6);
    let preview: Vec<String> = sample_data[..preview_len].iter().map(|v| format!("{:.4}", v)).collect();

    println!(
        "[DEBUG::{}] shape={:?} min={:.4} max={:.4} mean={:.4} sample=[{}{}]",
        label,
        telemetry.shape,
        telemetry.min,
        telemetry.max,
        telemetry.mean,
        preview.join(", "),
        if sample_data.len() > 6 { ", ..." } else { "" }
    );

    array.clone()
}

/// A microsecond-precision functional execution profiler.
pub struct ProfileTrace {
    label: String,
    start: Instant,
}

impl ProfileTrace {
    pub fn start(label: &str) -> Self {
        Self {
            label: label.to_string(),
            start: Instant::now(),
        }
    }

    /// Stops the timer and prints the elapsed microsecond/millisecond latency.
    pub fn stop(&self) -> f64 {
        let elapsed = self.start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let us = elapsed.as_micros();
        println!("[PROFILE::{}] elapsed: {:>8.3} ms ({} µs)", self.label, ms, us);
        ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_trap_detection() {
        let clean = Array::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(debug_check_nans(&clean, "clean_check").is_ok());

        let corrupted = Array::from_vec(vec![1.0, f32::NAN, 3.0], vec![3]).unwrap();
        let res = debug_check_nans(&corrupted, "corrupted_op");
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("NaN/Inf TRAP in [corrupted_op]"));
    }

    #[test]
    fn test_debug_print_transparency() {
        let x = Array::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
        let inspected = debug_print(&x, "inspection");
        assert_eq!(inspected.to_vec(), x.to_vec());
    }
}

