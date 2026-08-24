//! Static-engine GPT-2 training binary.
//!
//! Uses `StaticGPT2` — zero Arc/Mutex/graph overhead per step.
//! Run: cargo run --release --bin train_static --features cuda

use anyhow::Result;
use darkforest_core::StaticGPT2;
use std::io::Write;
use std::time::Instant;

struct CharTokenizer {
    vocab: Vec<char>,
    index_by_char: std::collections::HashMap<char, usize>,
}

impl CharTokenizer {
    fn build(text: &str) -> Self {
        let chars: Vec<char> = text
            .chars()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let index_by_char = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharTokenizer { vocab: chars, index_by_char }
    }
    fn vocab_size(&self) -> usize { self.vocab.len() }
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.index_by_char.get(&c).copied())
            .collect()
    }
}

fn load_text() -> String {
    let path = "data/tinyshakespeare/input.txt";
    if std::path::Path::new(path).exists() {
        return std::fs::read_to_string(path).unwrap();
    }
    "Hello world! Dark Forest static engine. ".repeat(500)
}

fn main() -> Result<()> {
    std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8");

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut n_steps: usize = 100;
    let mut ctx_len: usize = 128;
    let mut d_model: usize = 128;
    let mut n_layers: usize = 4;
    let mut d_ff: usize = 512;
    let mut n_heads: usize = 1;
    let mut lr: f32 = 3e-4;
    let mut vocab_size_override: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { n_steps = args[i+1].parse()?; i += 2; }
            "--ctx-len" => { ctx_len = args[i+1].parse()?; i += 2; }
            "--d-model" => { d_model = args[i+1].parse()?; i += 2; }
            "--n-layers" => { n_layers = args[i+1].parse()?; i += 2; }
            "--d-ff" => { d_ff = args[i+1].parse()?; i += 2; }
            "--n-heads" => { n_heads = args[i+1].parse()?; i += 2; }
            "--lr" => { lr = args[i+1].parse()?; i += 2; }
            "--vocab-size" => { vocab_size_override = Some(args[i+1].parse()?); i += 2; }
            _ => { i += 1; }
        }
    }

    let text = load_text();
    let tok = CharTokenizer::build(&text);
    let tokens: Vec<usize> = tok.encode(&text);
    let vocab_size = vocab_size_override.unwrap_or_else(|| tok.vocab_size());

    println!("Dark Forest — Static Engine Training");
    println!("=====================================");
    println!("vocab={}, tokens={}, ctx={}", vocab_size, tokens.len(), ctx_len);

    println!("Configuration: d_model={}, n_layers={}, n_heads={}, d_ff={}, lr={}, vocab_size={}",
             d_model, n_layers, n_heads, d_ff, lr, vocab_size);

    let mut model = StaticGPT2::new(vocab_size, d_model, n_heads, n_layers, d_ff, ctx_len, lr)?;

    println!("Model ready. Running {} steps...\n", n_steps);

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n_tokens = tokens.len();
    let log_every = n_steps.min(50);

    let mut loss_log = std::fs::File::create("training_loss_static.csv")?;
    writeln!(loss_log, "step,loss")?;

    let mut first_loss = None;
    let mut final_loss = 0.0f32;
    let mut min_loss = f32::INFINITY;
    let mut step_times_ms = Vec::with_capacity(n_steps);

    for step in 1..=n_steps {
        let t0 = Instant::now();

        let start = rng.gen_range(0..n_tokens - ctx_len - 1);
        let input_tokens  = &tokens[start..start + ctx_len];
        let target_tokens = &tokens[start + 1..start + ctx_len + 1];

        let loss_val = model.step(input_tokens, target_tokens)?;
        darkforest_core::cuda_sync()?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        step_times_ms.push(elapsed_ms);

        first_loss.get_or_insert(loss_val);
        final_loss = loss_val;
        min_loss = min_loss.min(loss_val);

        if step % log_every == 0 || step == 1 {
            writeln!(loss_log, "{},{:.6}", step, loss_val)?;
            let tok_s = (ctx_len as f64 / elapsed_ms) * 1000.0;
            println!(
                "step {:5} | loss {:6.4} | step_time {:6.2}ms | {:.0} tok/s",
                step, loss_val, elapsed_ms, tok_s
            );
        }
    }

    // Skip first 5 steps (warmup) for median
    let mut warm = step_times_ms[5.min(step_times_ms.len() - 1)..].to_vec();
    warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = if warm.is_empty() { f64::NAN } else { warm[warm.len() / 2] };
    let mean_ms: f64 = if warm.is_empty() { f64::NAN } else { warm.iter().sum::<f64>() / warm.len() as f64 };

    println!("\n=======================================================");
    println!(" Dark Forest Static Engine — Final Benchmark");
    println!("=======================================================");
    println!(" Steps:         {}", n_steps);
    println!(" Median step:   {:.3} ms", median_ms);
    println!(" Mean step:     {:.3} ms", mean_ms);
    println!(" Loss: initial {:.4} | final {:.4} | min {:.4}",
             first_loss.unwrap_or(final_loss), final_loss, min_loss);
    println!("=======================================================");

    Ok(())
}
