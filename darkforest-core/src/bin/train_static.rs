//! Static-engine GPT-2 training binary.
//!
//! Uses `StaticGPT2` — zero Arc/Mutex/graph overhead per step.
//! Run: cargo run --release --bin train_static --features cuda

use anyhow::Result;
use darkforest_core::{Engine, EngineConfig, SamplerConfig, StaticLoRAConfig};
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
    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&idx| self.vocab.get(idx).copied().unwrap_or('?'))
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
    let mut checkpoint_save: Option<String> = None;
    let mut checkpoint_load: Option<String> = None;
    let mut max_grad_norm: Option<f32> = None;
    let mut eval_every: Option<usize> = None;
    let mut generate_tokens: Option<usize> = None;
    let mut generate_compare = false;
    let mut use_lora = false;
    let mut lora_rank: usize = 8;
    let mut lora_alpha: f32 = 16.0;
    let mut use_qlora = false;
    let mut lora_save: Option<String> = None;
    let mut merge_lora = false;
    let mut rep_penalty: f32 = 1.0;
    let mut stream_output = false;

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
            "--d-model" => {
                d_model = args[i + 1].parse()?;
                i += 2;
            }
            "--n-layers" => {
                n_layers = args[i + 1].parse()?;
                i += 2;
            }
            "--d-ff" => {
                d_ff = args[i + 1].parse()?;
                i += 2;
            }
            "--n-heads" => {
                n_heads = args[i + 1].parse()?;
                i += 2;
            }
            "--lr" => {
                lr = args[i + 1].parse()?;
                i += 2;
            }
            "--vocab-size" => {
                vocab_size_override = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--checkpoint-save" => {
                checkpoint_save = Some(args[i + 1].clone());
                i += 2;
            }
            "--checkpoint-load" => {
                checkpoint_load = Some(args[i + 1].clone());
                i += 2;
            }
            "--max-grad-norm" => {
                max_grad_norm = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--eval-every" => {
                eval_every = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--generate" => {
                generate_tokens = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--generate-compare" => {
                generate_tokens = Some(args[i + 1].parse()?);
                generate_compare = true;
                i += 2;
            }
            "--lora" => {
                use_lora = true;
                i += 1;
            }
            "--lora-rank" => {
                use_lora = true;
                lora_rank = args[i + 1].parse()?;
                i += 2;
            }
            "--lora-alpha" => {
                use_lora = true;
                lora_alpha = args[i + 1].parse()?;
                i += 2;
            }
            "--qlora" => {
                use_lora = true;
                use_qlora = true;
                i += 1;
            }
            "--save-lora" => {
                use_lora = true;
                lora_save = Some(args[i + 1].clone());
                i += 2;
            }
            "--merge-lora" => {
                merge_lora = true;
                i += 1;
            }
            "--rep-penalty" => {
                rep_penalty = args[i + 1].parse()?;
                i += 2;
            }
            "--stream" => {
                stream_output = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let text = load_text();
    let tok = CharTokenizer::build(&text);
    let tokens: Vec<usize> = tok.encode(&text);
    let vocab_size = vocab_size_override.unwrap_or_else(|| tok.vocab_size());

    println!("Dark Forest — Complete ML Engine Runtime");
    println!("=========================================");
    println!(
        "vocab={}, tokens={}, ctx={}",
        vocab_size,
        tokens.len(),
        ctx_len
    );

    let lora_cfg = if use_lora {
        Some(StaticLoRAConfig::new(lora_rank, lora_alpha).with_qlora(use_qlora))
    } else {
        None
    };

    let engine_cfg = EngineConfig {
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        seq_len: ctx_len,
        lr,
        max_grad_norm,
        weight_decay: 0.01,
        lora: lora_cfg,
    };

    let mut engine = Engine::new(engine_cfg)?;
    println!("Engine Summary: {}", engine.summary());

    if let Some(ref path) = checkpoint_load {
        println!("Loading initial weights and optimizer state from {}...", path);
        engine.load_checkpoint(std::path::Path::new(path))?;
        println!("Checkpoint loaded successfully. Engine step_count={}", engine.step_count());
    }

    if use_lora {
        if let Engine::LoRA(ref lora_model) = engine {
            println!("\n=======================================================");
            println!(" Parameter-Efficient Fine-Tuning Active (LoRA)");
            println!("=======================================================");
            println!("  Base Parameters:       {:>12}", lora_model.total_base_params());
            println!(
                "  Trainable LoRA Params: {:>12} ({:.3}%)",
                lora_model.trainable_lora_params(),
                lora_model.trainable_fraction()
            );
            println!("  LoRA Rank (r):         {:>12}", lora_rank);
            println!("  LoRA Alpha:            {:>12.1}", lora_alpha);
            if use_qlora {
                println!("  QLoRA (4-bit NF4):     ENABLED (7.11x base compression)");
            }
            println!("=======================================================\n");
        }
    }

    println!("Engine ready. Running {} steps...\n", n_steps);

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n_tokens = tokens.len();

    let mut loss_log = std::fs::File::create("training_loss_static.csv")?;
    writeln!(loss_log, "step,loss,rolling_time_ms,step_time_ms")?;

    let mut first_loss = None;
    let mut final_loss = 0.0f32;
    let mut min_loss = f32::INFINITY;
    let mut step_times_ms = Vec::with_capacity(n_steps);

    for step in 1..=n_steps {
        let t0 = Instant::now();

        let start = rng.gen_range(0..n_tokens - ctx_len - 1);
        let input_tokens = &tokens[start..start + ctx_len];
        let target_tokens = &tokens[start + 1..start + ctx_len + 1];

        let loss_val = engine.step(input_tokens, target_tokens)?;
        darkforest_core::cuda_sync()?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        step_times_ms.push(elapsed_ms);

        first_loss.get_or_insert(loss_val);
        final_loss = loss_val;
        min_loss = min_loss.min(loss_val);

        if step == 1 {
            println!("step {:5} | loss {:6.4} | step_time {:6.2}ms | {:.0} tok/s", step, loss_val, elapsed_ms, (ctx_len as f64 / elapsed_ms) * 1000.0);
            writeln!(loss_log, "{},{:.6},{:.3},{:.3}", step, loss_val, loss_val, elapsed_ms)?;
        } else if step % 25 == 0 || step == n_steps {
            let window_start = step.saturating_sub(25);
            let recent_times = &step_times_ms[window_start..step];
            let rolling_time_ms: f64 = recent_times.iter().sum::<f64>() / recent_times.len() as f64;
            let tok_s = (ctx_len as f64 / rolling_time_ms) * 1000.0;
            println!(
                "step {:5} | loss {:6.4} | rolling_step {:6.2}ms | instantaneous {:6.2}ms | {:.0} tok/s",
                step, loss_val, rolling_time_ms, elapsed_ms, tok_s
            );
            writeln!(loss_log, "{},{:.6},{:.3},{:.3}", step, loss_val, rolling_time_ms, elapsed_ms)?;
        }

        if let Some(eval_interval) = eval_every {
            if step % eval_interval == 0 {
                let eval_start = rng.gen_range(0..n_tokens - ctx_len - 1);
                let val_in = &tokens[eval_start..eval_start + ctx_len];
                let val_tgt = &tokens[eval_start + 1..eval_start + ctx_len + 1];
                let val_loss = engine.eval_step(val_in, val_tgt)?;
                println!("  [EVAL] step {:5} | validation loss: {:6.4}", step, val_loss);
            }
        }
    }

    // Skip first 5 steps (warmup) for median
    let mut warm = step_times_ms[5.min(step_times_ms.len() - 1)..].to_vec();
    warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = if warm.is_empty() {
        f64::NAN
    } else {
        warm[warm.len() / 2]
    };
    let mean_ms: f64 = if warm.is_empty() {
        f64::NAN
    } else {
        warm.iter().sum::<f64>() / warm.len() as f64
    };

    println!("\n=======================================================");
    println!(" Dark Forest Static Engine — Final Benchmark");
    println!("=======================================================");
    println!(" Steps:         {}", n_steps);
    println!(" Median step:   {:.3} ms", median_ms);
    println!(" Mean step:     {:.3} ms", mean_ms);
    println!(
        " Loss: initial {:.4} | final {:.4} | min {:.4}",
        first_loss.unwrap_or(final_loss),
        final_loss,
        min_loss
    );
    println!("=======================================================");

    if let Some(ref path) = checkpoint_save {
        println!("\nSaving trained model checkpoint to {}...", path);
        engine.save_checkpoint(std::path::Path::new(path))?;
        println!("Checkpoint saved successfully.");
    }

    if let Some(ref path) = lora_save {
        println!("\nSaving LoRA adapter checkpoint to {}...", path);
        engine.save_lora_checkpoint(std::path::Path::new(path))?;
        println!("LoRA checkpoint saved successfully.");
    }

    if merge_lora && engine.is_lora() {
        println!("\nMerging LoRA adapter weights into base model (W' = W + alpha/r * B@A)...");
        engine.merge_lora()?;
        println!("Merged LoRA adapters directly into base weights for zero-overhead deployment.");
    }

    if let Some(n_gen) = generate_tokens {
        let prompt = if std::path::Path::new("data/tinyshakespeare/input.txt").exists() {
            "ROMEO: "
        } else {
            "Dark Forest"
        };
        let prompt_tokens = tok.encode(prompt);
        let cfg = SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: rep_penalty,
            stop_tokens: Vec::new(),
            min_tokens: 0,
        };

        println!("\n=======================================================");
        println!(" Autoregressive Inference Benchmark");
        println!("=======================================================");
        println!(" Prompt:       {:?} ({} tokens)", prompt, prompt_tokens.len());
        println!(" New tokens:   {}", n_gen);
        println!(" Rep. penalty: {:.2}", rep_penalty);

        // ── Fast path: KV-cache O(S) decode ────────────────────────────────────
        println!("\n[1/2] Fast KV-Cache Decode (O(S) single-token decode steps)...");
        if stream_output {
            print!("  Output: ");
            std::io::stdout().flush()?;
            let tok_chars = tok.vocab.clone();
            let rep = engine.generate_stream(&prompt_tokens, n_gen, &cfg, |tok_idx| {
                let ch = tok_chars.get(tok_idx).copied().unwrap_or('?');
                print!("{}", ch);
                let _ = std::io::stdout().flush();
                true
            });
            println!();
            match rep {
                Ok(report) => {
                    println!(
                        "  Speed: {:.3} ms/token | {:.1} tok/s | total {:.2} ms",
                        report.ms_per_token, report.tokens_per_sec, report.total_time_ms
                    );
                }
                Err(e) => println!("  Streaming note: {e}"),
            }
        } else {
            let t0 = std::time::Instant::now();
            match engine.generate_fast(&prompt_tokens, n_gen, &cfg) {
                Ok(gen_tokens) => {
                    let elapsed = t0.elapsed();
                    let ms_per_token = elapsed.as_secs_f64() * 1000.0 / n_gen as f64;
                    let tokens_per_sec = n_gen as f64 / elapsed.as_secs_f64();
                    println!("  Output:\n  -------------------------------------------------------");
                    println!("  {}", tok.decode(&gen_tokens));
                    println!("  -------------------------------------------------------");
                    println!(
                        "  Speed: {:.3} ms/token | {:.1} tok/s | total {:.2?}",
                        ms_per_token, tokens_per_sec, elapsed
                    );

                    // ── Naive comparison (optional) ──────────────────────────────
                    if generate_compare {
                        println!("\n[2/2] Naive Full-Context Recompute (O(S²) re-run every token)...");
                        let t1 = std::time::Instant::now();
                        match engine.generate(&prompt_tokens, n_gen, &cfg) {
                            Ok(naive_tokens) => {
                                let elapsed_naive = t1.elapsed();
                                let ms_naive = elapsed_naive.as_secs_f64() * 1000.0 / n_gen as f64;
                                let tps_naive = n_gen as f64 / elapsed_naive.as_secs_f64();
                                println!("  Output:\n  -------------------------------------------------------");
                                println!("  {}", tok.decode(&naive_tokens));
                                println!("  -------------------------------------------------------");
                                println!(
                                    "  Speed: {:.3} ms/token | {:.1} tok/s | total {:.2?}",
                                    ms_naive, tps_naive, elapsed_naive
                                );

                                let speedup = ms_naive / ms_per_token;
                                println!("\n=======================================================");
                                println!(" KV-Cache Speedup Summary");
                                println!("=======================================================");
                                println!(" Fast (KV-cache):   {:.3} ms/token  ({:.1} tok/s)", ms_per_token, tokens_per_sec);
                                println!(" Naive (recompute): {:.3} ms/token  ({:.1} tok/s)", ms_naive, tps_naive);
                                println!(" Speedup:           {:.2}x faster", speedup);
                                println!("=======================================================");
                            }
                            Err(e) => println!("  Naive generation note: {e}"),
                        }
                    } else {
                        println!("\n  (tip: pass --generate-compare {n_gen} instead to also run the naive baseline)");
                    }
                }
                Err(e) => {
                    println!("  Fast generation note: {e}");
                }
            }
        }
        println!("=======================================================");
    }

    Ok(())
}

