// Granular per-section timing by adding sync points between each stage
use anyhow::Result;
use darkforest_core::StaticGPT2;
use std::time::Instant;

macro_rules! timed {
    ($label:expr, $body:expr) => {{
        let t = Instant::now();
        $body;
        darkforest_core::cuda_sync()?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  {:40} {:7.3} ms", $label, ms);
        ms
    }};
}

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

    for _ in 0..3 {
        let _ = model.step(&in_tokens, &tgt_tokens)?;
        darkforest_core::cuda_sync()?;
    }

    println!("[Granular per-operation timing with sync after each op]");
    model.profile_step(&in_tokens, &tgt_tokens)?;

    Ok(())
}
