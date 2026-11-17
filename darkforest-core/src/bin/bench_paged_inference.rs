//! bench_paged_inference.rs — Paged KV Cache vs. Contiguous KV Cache Inference Benchmark
//!
//! Demonstrates end-to-end sequence length scaling using Dark Forest's paged memory system.
//! Unlike the flat KV cache (must pre-allocate `max_seq_len × n_layers × d_model`),
//! the paged system allocates memory incrementally in fixed-size pages.
//!
//! # What this measures
//!
//! - Paged KV cache decode step latency (ms/token) across token budgets
//! - Memory allocated vs memory ceiling comparison
//! - Sequence lengths reachable with paged vs. flat allocation on the same VRAM
//!
//! # Usage
//!
//! ```
//! cargo run --release --bin bench_paged_inference --features cuda
//! ```

use anyhow::Result;

#[cfg(feature = "cuda")]
use darkforest_core::engine::{
    paged_kv::{KVDtype, PagedKVCache},
    SamplerConfig, StaticGPT2,
};
#[cfg(feature = "cuda")]
use darkforest_cuda::get_memory_info;

fn main() -> Result<()> {
    println!("================================================================================");
    println!(" Dark Forest — Paged KV Cache vs. Flat KV Cache Inference Benchmark");
    println!("================================================================================\n");

    #[cfg(feature = "cuda")]
    run_paged_benchmark()?;

    #[cfg(not(feature = "cuda"))]
    println!(" [!] CUDA feature not enabled — rebuild with --features cuda\n");

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_paged_benchmark() -> Result<()> {
    let vocab_size = 256usize;
    let d_model = 768usize;
    let n_heads = 12usize;
    let n_layers = 12usize;
    let d_ff = 3072usize;
    let seq_len = 1024usize;
    let lr = 1e-4f32;

    let (free_init, total) = get_memory_info().unwrap_or((0, 0));
    let free_mb = free_init as f64 / (1024.0 * 1024.0);
    let total_mb = total as f64 / (1024.0 * 1024.0);
    println!(
        " GPU: {:.0} MB free / {:.0} MB total ({:.2} GB VRAM)",
        free_mb,
        total_mb,
        total_mb / 1024.0
    );
    println!(
        " Config: vocab={} d={} H={} L={} d_ff={} ctx={}",
        vocab_size, d_model, n_heads, n_layers, d_ff, seq_len
    );
    println!();

    // Build model
    println!(" Building StaticGPT2...");
    let mut model = StaticGPT2::new(vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr)?;

    let (free_after_model, _) = get_memory_info().unwrap_or((0, 0));
    let model_vram_mb = (free_init as f64 - free_after_model as f64) / (1024.0 * 1024.0);
    println!(" Model weights:  {:.1} MB VRAM", model_vram_mb);

    // -------------------------------------------------------------------------
    // Flat KV cache cost calculation (pre-allocated contiguous)
    // -------------------------------------------------------------------------
    let flat_kv_mb_per_ctx = |ctx: usize| -> f64 {
        // 2 (K+V) × n_layers × ctx × d_model × 4 bytes (f32)
        (2.0 * n_layers as f64 * ctx as f64 * d_model as f64 * 4.0) / (1024.0 * 1024.0)
    };

    // -------------------------------------------------------------------------
    // Paged KV cache: how many tokens can we fit in remaining VRAM?
    // -------------------------------------------------------------------------
    let page_size = 16usize;
    let bytes_per_page_layer =
        page_size * d_model * 4 * 2; // K + V, f32
    let bytes_per_page_total = bytes_per_page_layer * n_layers;
    let available_bytes = (free_after_model as f64 * 0.75) as usize; // leave 25% headroom
    let max_paged_blocks = available_bytes / bytes_per_page_total;
    let max_paged_tokens = max_paged_blocks * page_size;

    println!(
        " Flat KV @1024 ctx:  {:.1} MB  |  Flat KV @4096 ctx:  {:.1} MB  |  Flat KV @16K ctx: {:.1} MB",
        flat_kv_mb_per_ctx(1024),
        flat_kv_mb_per_ctx(4096),
        flat_kv_mb_per_ctx(16384),
    );
    println!(
        " Paged capacity:     {} blocks × {} tokens = {} tokens ({:.0}K context) in {:.1} MB",
        max_paged_blocks,
        page_size,
        max_paged_tokens,
        max_paged_tokens as f64 / 1024.0,
        (bytes_per_page_total * max_paged_blocks) as f64 / (1024.0 * 1024.0),
    );
    println!();

    // -------------------------------------------------------------------------
    // Build paged cache with capacity for available VRAM
    // -------------------------------------------------------------------------
    let bench_max_seq = max_paged_tokens.min(16384); // cap benchmark at 16K for time
    let paged_num_blocks = (bench_max_seq + page_size - 1) / page_size;

    println!(
        " Allocating paged cache: {} blocks × page_size {} = {} token capacity",
        paged_num_blocks, page_size, paged_num_blocks * page_size
    );
    let mut paged_cache = PagedKVCache::new(
        paged_num_blocks * page_size,
        page_size,
        n_layers,
        n_heads,
        d_model / n_heads,
        KVDtype::F32,
    )?;
    let (free_after_paged, _) = get_memory_info().unwrap_or((0, 0));
    let paged_vram_mb = (free_after_model as f64 - free_after_paged as f64) / (1024.0 * 1024.0);
    println!(" Paged cache VRAM:   {:.1} MB\n", paged_vram_mb);

    // -------------------------------------------------------------------------
    // Benchmark: decode latency at multiple sequence lengths
    // -------------------------------------------------------------------------
    let sampler = SamplerConfig::greedy();
    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let token_budgets: Vec<usize> = vec![16, 64, 256, 512, 1024, 2048, 4096];
    let token_budgets: Vec<usize> = token_budgets
        .into_iter()
        .filter(|&n| n + prompt.len() <= paged_num_blocks * page_size)
        .collect();

    println!(
        " {:<10} {:>14} {:>12} {:>14} {:>14}",
        "Tokens", "ms/token", "tok/s", "VRAM-used MB", "Pages used"
    );
    println!(" {}", "-".repeat(72));

    for &budget in &token_budgets {
        let (free_before, _) = get_memory_info().unwrap_or((0, 0));
        let t0 = std::time::Instant::now();

        let report =
            model.generate_paged(&prompt, budget, &mut paged_cache, &sampler)?;

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ms_per_tok = if report.new_tokens > 0 {
            elapsed_ms / report.new_tokens as f64
        } else {
            0.0
        };
        let tok_s = if elapsed_ms > 0.0 {
            report.new_tokens as f64 / elapsed_ms * 1000.0
        } else {
            0.0
        };

        let (free_after, _) = get_memory_info().unwrap_or((0, 0));
        let used_mb = (free_before as f64 - free_after as f64) / (1024.0 * 1024.0);
        let pages_used = paged_cache.sequence.block_table.len();

        println!(
            " {:<10} {:>14.3} {:>12.1} {:>14.1} {:>14}",
            report.new_tokens,
            ms_per_tok,
            tok_s,
            used_mb.max(0.0),
            pages_used,
        );
    }

    println!();
    println!("================================================================================");

    // -------------------------------------------------------------------------
    // BF16 capacity comparision 
    // -------------------------------------------------------------------------
    let bf16_capacity_tokens = max_paged_blocks * page_size * 2; // 2x due to half precision
    println!(
        " BF16 Paged capacity estimate: {} tokens ({:.0}K) — same VRAM as FP32 @{}K",
        bf16_capacity_tokens,
        bf16_capacity_tokens as f64 / 1024.0,
        paged_num_blocks * page_size / 1024,
    );
    println!("================================================================================\n");

    Ok(())
}
