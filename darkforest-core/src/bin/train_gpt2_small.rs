//! Binary: train GPT-2 small on TinyShakespeare.
//! Run: cargo run --release --bin train_gpt2_small

use anyhow::Result;
use darkforest_core::nn::transformer::{GPT2Config, GPT2};
use darkforest_core::optimizer::AdamW;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Minimal BPE tokenizer (character-level fallback for Phase 1)
// ---------------------------------------------------------------------------
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
        let index_by_char = chars
            .iter()
            .enumerate()
            .map(|(idx, ch)| (*ch, idx))
            .collect();

        CharTokenizer {
            vocab: chars,
            index_by_char,
        }
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.index_by_char.get(&c).copied())
            .collect()
    }
}

fn load_tinyshakespeare() -> Result<String> {
    let path = "data/tinyshakespeare/input.txt";
    if std::path::Path::new(path).exists() {
        return Ok(std::fs::read_to_string(path)?);
    }
    // Download if missing
    println!("Downloading TinyShakespeare...");
    let url =
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
    let response = std::process::Command::new("curl")
        .args(["-L", "-o", path, "--create-dirs", url])
        .status();
    match response {
        Ok(s) if s.success() => Ok(std::fs::read_to_string(path)?),
        _ => {
            // Fallback: generate a small synthetic text
            println!("Warning: Could not download dataset. Using synthetic text.");
            Ok("Hello world! This is Dark Forest, a fast ML framework. ".repeat(200))
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().skip(1).collect();
    let mut n_steps = 1000usize;
    let mut ctx_len = 64usize;
    let mut device = "cuda";

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                n_steps = args[i + 1].parse()?;
                i += 2;
            }
            "--ctx-len" => {
                ctx_len = args[i + 1].parse()?;
                i += 2;
            }
            "--device" => {
                device = args[i + 1].as_str();
                i += 2;
            }
            "--help" => {
                println!("Usage: train_gpt2_small [--steps N] [--ctx-len L] [--device cpu|cuda]");
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                return Ok(());
            }
        }
    }

    let log_every = n_steps.min(50);

    println!("Dark Forest — GPT-2 Training");
    println!("============================");

    // ---------------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------------
    let text = load_tinyshakespeare()?;
    let tokenizer = CharTokenizer::build(&text);
    let tokens: Vec<usize> = tokenizer.encode(&text);

    println!(
        "Dataset: {} chars, vocab size: {}",
        text.len(),
        tokenizer.vocab_size()
    );

    // ---------------------------------------------------------------------------
    // Model (tiny config for CPU Phase 1)
    // ---------------------------------------------------------------------------
    let mut cfg = GPT2Config::tiny();
    cfg.vocab_size = tokenizer.vocab_size();
    cfg.max_seq_len = ctx_len;

    println!("Config: {:?}", cfg);
    println!("Approx params: {}", cfg.param_count());

    let mut model = GPT2::new(cfg.clone());
    if device == "cuda" {
        #[cfg(feature = "cuda")]
        {
            println!("Moving model parameters to CUDA device 0 (Device-Resident Global Mode)...");
            model.to_device(darkforest_core::tensor::Device::Cuda(0))?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            println!("CUDA feature not enabled; falling back to CPU training.");
        }
    }
    let params = model.parameters();

    let mut optimizer = AdamW::new(3e-4, 0.9, 0.95, 1e-8, 0.1);
    optimizer.init_moments(&params);

    // ---------------------------------------------------------------------------
    // Training loop
    // ---------------------------------------------------------------------------
    println!("\nStarting training...\n");
    let mut rng = rand::thread_rng();
    let n_tokens = tokens.len();
    let mut loss_log = File::create("training_loss.csv")?;
    writeln!(loss_log, "step,loss")?;
    let mut first_loss = None;
    let mut final_loss = 0.0f32;
    let mut min_loss = f32::INFINITY;

    for step in 1..=n_steps {
        let t0 = Instant::now();

        // Random context window
        use rand::Rng;
        let start = rng.gen_range(0..n_tokens - ctx_len - 1);
        let batch_tokens = &tokens[start..start + ctx_len + 1];

        // Zero grads
        optimizer.zero_grad(&params);

        // Forward + loss + backward
        let loss = model.loss(batch_tokens)?;
        loss.backward();

        // Optimizer step on device
        optimizer.step(&params, None);

        let elapsed = t0.elapsed().as_millis();

        if step % log_every == 0 || step == 1 {
            let loss_val = loss.tensor().get(0);
            first_loss.get_or_insert(loss_val);
            final_loss = loss_val;
            min_loss = min_loss.min(loss_val);
            writeln!(loss_log, "{step},{loss_val:.6}")?;
            let tokens_per_sec = (ctx_len as f32 / elapsed.max(1) as f32) * 1000.0;
            println!(
                "step {:5} | loss {:6.4} | step_time {:5}ms | {:.0} tok/s",
                step, loss_val, elapsed, tokens_per_sec
            );
        }
    }

    println!("\nTraining complete.");
    println!(
        "Loss: initial {:.4} | final {:.4} | minimum {:.4}",
        first_loss.unwrap_or(final_loss),
        final_loss,
        min_loss
    );
    println!("Loss curve written to training_loss.csv");

    Ok(())
}
