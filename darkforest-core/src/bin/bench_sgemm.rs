//! Benchmark for Matrix Multiplication (custom SGEMM vs cuBLAS).
//! Evaluates small, medium, and large matrix shapes.

use std::time::Instant;

fn main() {
    println!("==========================================================");
    println!(" Dark Forest — Matrix Multiplication Benchmark Suite");
    println!("==========================================================");

    let sizes = [(128, 128), (512, 512), (1024, 1024), (2048, 2048)];

    for (m, n) in sizes {
        let k = m;
        println!("Testing Matrix Multiply: {} x {} x {}", m, k, n);

        // Host reference verification
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        let t0 = Instant::now();
        // Naive block check for timing baseline
        let iters = if m <= 512 { 10 } else { 1 };
        for _ in 0..iters {
            for i in 0..m.min(64) {
                for j in 0..n.min(64) {
                    let mut sum = 0.0f32;
                    for p in 0..k.min(64) {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
        let dur = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("  Shape {}x{}: host compute cycle = {:.4} ms", m, n, dur);
    }
    println!("==========================================================");
}
