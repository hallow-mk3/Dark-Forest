//! Dark Forest Standalone Engine CLI.
//!
//! Provides a direct, native command-line interface for:
//! - Interactive chat/REPL completion
//! - Autoregressive prompt generation with real-time streaming
//! - Model training and validation
//! - Parameter count and memory inspection
//!
//! Run:
//!   cargo run --release --bin darkforest_cli --features cuda -- generate --prompt "Once upon a time" --tokens 64

use anyhow::Result;
use darkforest_core::{Engine, EngineConfig, SamplerConfig};
use std::io::{self, Write};

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

fn load_dataset() -> String {
    let path = "data/tinyshakespeare/input.txt";
    if std::path::Path::new(path).exists() {
        return std::fs::read_to_string(path).unwrap();
    }
    "Dark Forest — High-Performance Rust & CUDA Machine Learning Runtime.\n".repeat(200)
}

fn print_help() {
    println!("Dark Forest CLI — High-Performance Rust & CUDA ML Engine");
    println!("Usage:");
    println!("  darkforest_cli generate [OPTIONS]     Generate text from a prompt");
    println!("  darkforest_cli chat     [OPTIONS]     Start interactive completion REPL");
    println!("  darkforest_cli info     [OPTIONS]     Print engine configuration and parameters");
    println!("  darkforest_cli help                   Show this help message\n");
    println!("  --config <PRESET>    Engine preset: 'gpt2', 'small', 'micro' (default: 'small')");
    println!("  --prompt <STR>       Input prompt text (default: 'Dark Forest')");
    println!("  --tokens <INT>       Maximum new tokens to generate (default: 64)");
    println!("  --temp <FLOAT>       Sampling temperature (default: 0.8)");
    println!("  --top-k <INT>        Top-K truncation cutoff (default: 40)");
    println!("  --top-p <FLOAT>      Nucleus Top-P filtering (default: 0.9)");
    println!("  --rep-penalty <FLOAT> Repetition penalty (default: 1.15)");
    println!("  --checkpoint <PATH>  Path to model checkpoint to load");
    println!("  --stream             Stream tokens to stdout in real-time");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "help" || args[0] == "--help" {
        print_help();
        return Ok(());
    }

    let command = args[0].as_str();

    let mut config_preset = "small".to_string();
    let mut prompt = "Dark Forest: ".to_string();
    let mut n_tokens: usize = 64;
    let mut temp: f32 = 0.8;
    let mut top_k: usize = 40;
    let mut top_p: f32 = 0.9;
    let mut rep_penalty: f32 = 1.15;
    let mut checkpoint_path: Option<String> = None;
    let mut stream = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                config_preset = args[i + 1].clone();
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            "--tokens" => {
                n_tokens = args[i + 1].parse()?;
                i += 2;
            }
            "--temp" => {
                temp = args[i + 1].parse()?;
                i += 2;
            }
            "--top-k" => {
                top_k = args[i + 1].parse()?;
                i += 2;
            }
            "--top-p" => {
                top_p = args[i + 1].parse()?;
                i += 2;
            }
            "--rep-penalty" => {
                rep_penalty = args[i + 1].parse()?;
                i += 2;
            }
            "--checkpoint" => {
                checkpoint_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--stream" => {
                stream = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let data_text = load_dataset();
    let tokenizer = CharTokenizer::build(&data_text);
    let vocab_size = tokenizer.vocab_size();

    let config = match config_preset.as_str() {
        "gpt2" | "gpt2_124m" => EngineConfig::gpt2_124m().with_vocab_size(vocab_size),
        "micro" => EngineConfig::micro(vocab_size, 128),
        _ => EngineConfig::small().with_vocab_size(vocab_size),
    };

    let mut engine = Engine::new(config)?;
    if let Some(ref path) = checkpoint_path {
        println!("Loading checkpoint from {}...", path);
        engine.load_checkpoint(std::path::Path::new(path))?;
    }

    let sampler_cfg = SamplerConfig {
        temperature: temp,
        top_k,
        top_p,
        repetition_penalty: rep_penalty,
        stop_tokens: Vec::new(),
        min_tokens: 0,
    };

    match command {
        "info" => {
            println!("=======================================================");
            println!(" Dark Forest Engine Configuration");
            println!("=======================================================");
            println!("  {}", engine.summary());
            println!("  Vocab Size:   {}", engine.vocab_size());
            println!("  Hidden Dim:   {}", engine.d_model());
            println!("  Layers:       {}", engine.n_layers());
            println!("  Heads:        {}", engine.n_heads());
            println!("  Context:      {} tokens", engine.seq_len());
            println!("=======================================================");
        }
        "generate" => {
            let prompt_tokens = tokenizer.encode(&prompt);
            println!("Prompt: {:?}", prompt);
            println!("Generating {} tokens (temp={temp}, top_k={top_k}, top_p={top_p}, rep={rep_penalty})...\n", n_tokens);
            print!("{prompt}");
            io::stdout().flush()?;

            let vocab_chars = tokenizer.vocab.clone();
            let report = engine.generate_stream(&prompt_tokens, n_tokens, &sampler_cfg, |tok| {
                if stream {
                    let ch = vocab_chars.get(tok).copied().unwrap_or('?');
                    print!("{}", ch);
                    let _ = io::stdout().flush();
                }
                true
            })?;
            println!("\n");
            println!("-------------------------------------------------------");
            println!(
                "Latency: {:.3} ms/token | Throughput: {:.1} tok/s | Total: {:.2} ms",
                report.ms_per_token, report.tokens_per_sec, report.total_time_ms
            );
            println!("-------------------------------------------------------");
        }
        "chat" => {
            println!("=======================================================");
            println!(" Dark Forest Interactive Completion REPL");
            println!(" Type 'quit' or 'exit' to terminate session.");
            println!("=======================================================\n");

            let stdin = io::stdin();
            let vocab_chars = tokenizer.vocab.clone();
            loop {
                print!("> ");
                io::stdout().flush()?;
                let mut line = String::new();
                if stdin.read_line(&mut line).is_err() || line.trim().is_empty() {
                    continue;
                }
                let input = line.trim();
                if input == "exit" || input == "quit" {
                    break;
                }
                let input_tokens = tokenizer.encode(input);
                if input_tokens.is_empty() {
                    println!("[Input could not be tokenized by current vocabulary]");
                    continue;
                }
                print!("{input}");
                io::stdout().flush()?;
                let _ = engine.generate_stream(&input_tokens, n_tokens, &sampler_cfg, |tok| {
                    let ch = vocab_chars.get(tok).copied().unwrap_or('?');
                    print!("{}", ch);
                    let _ = io::stdout().flush();
                    true
                });
                println!("\n");
            }
        }
        other => {
            eprintln!("Unknown command: '{}'. Run 'darkforest_cli help' for usage.", other);
            std::process::exit(1);
        }
    }

    Ok(())
}
