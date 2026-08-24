//! Micro-benchmark pinpointing the exact kernel function taking 332ms.

use anyhow::Result;
use darkforest_core::StaticGPT2;
use std::time::Instant;

fn main() -> Result<()> {
    let vocab_size = 65;
    let d_model = 128;
    let n_heads = 1;
    let n_layers = 4;
    let d_ff = 512;
    let seq_len = 128;
    let lr = 3e-4;

    let mut model = StaticGPT2::new(vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr)?;
    let in_tokens = vec![1usize; seq_len];
    let tgt_tokens = vec![2usize; seq_len];

    for _ in 0..2 {
        let _ = model.step(&in_tokens, &tgt_tokens)?;
        darkforest_core::cuda_sync()?;
    }

    let t0 = Instant::now();
    let _ = model.step(&in_tokens, &tgt_tokens)?;
    darkforest_core::cuda_sync()?;
    println!("Step time: {:.3} ms", t0.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}
