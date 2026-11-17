// Granular per-section timing by adding sync points between each stage
use anyhow::Result;
use darkforest_core::StaticGPT2;

fn main() -> Result<()> {
    let vocab_size = 50257;
    let d_model = 768;
    let n_heads = 12;
    let n_layers = 12;
    let d_ff = 3072;
    let seq_len = 1024;
    let lr = 1e-4;

    let mut model = StaticGPT2::new(vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr)?;
    let in_tokens = vec![1usize; seq_len];
    let tgt_tokens = vec![2usize; seq_len];

    for _ in 0..2 {
        let _ = model.step(&in_tokens, &tgt_tokens)?;
        darkforest_core::cuda_sync()?;
    }

    println!("[Granular per-operation timing with sync after each op]");
    model.profile_step(&in_tokens, &tgt_tokens)?;

    Ok(())
}
