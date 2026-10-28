from __future__ import annotations

import argparse
import sys
import time

import darkforest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="darkforest", description="Dark Forest ML Engine CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print the Dark Forest package version")
    subparsers.add_parser("status", help="Print runtime status and hardware detection")
    subparsers.add_parser("smoke-test", help="Run the autograd smoke test")
    subparsers.add_parser("static-smoke", help="Instantiate and test StaticGPT2 engine from Python")
    subparsers.add_parser("lora-smoke", help="Instantiate and test StaticLoRAGPT2 PEFT engine from Python")

    # Generate command
    gen_p = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_p.add_argument("--prompt", type=str, default="Dark Forest: ", help="Input text prompt")
    gen_p.add_argument("--tokens", type=int, default=32, help="Tokens to generate")
    gen_p.add_argument("--temp", type=float, default=0.8, help="Temperature (0.0 = greedy)")
    gen_p.add_argument("--top-k", type=int, default=40, help="Top-K cutoff")
    gen_p.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus filtering")
    gen_p.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty")

    # Train command
    train_p = subparsers.add_parser("train", help="Train a transformer model on sample text")
    train_p.add_argument("--steps", type=int, default=50, help="Training steps")
    train_p.add_argument("--ctx-len", type=int, default=64, help="Context length")
    train_p.add_argument("--d-model", type=int, default=128, help="Hidden dimension")
    train_p.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    train_p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA PEFT")
    train_p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    train_p.add_argument("--checkpoint-save", type=str, default=None, help="Save checkpoint path")

    # Bench command
    bench_p = subparsers.add_parser("bench", help="Run matrix multiplication and kernel benchmarks")
    bench_p.add_argument("--steps", type=int, default=20, help="Benchmark steps")
    bench_p.add_argument("--rows", type=int, default=1024, help="Matrix rows")
    bench_p.add_argument("--cols", type=int, default=1024, help="Matrix cols")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "version"):
        print(darkforest.version())
        return 0

    if args.command == "status":
        print(darkforest.status())
        return 0

    if args.command == "smoke-test":
        print(darkforest.smoke_test())
        return 0

    if args.command == "static-smoke":
        model = darkforest.StaticGPT2(
            vocab_size=65, d_model=64, n_heads=1, n_layers=2, d_ff=256, seq_len=32, lr=1e-3
        )
        print(f"Instantiated: {model}")
        print(
            f"Attributes: vocab={model.vocab_size}, d_model={model.d_model}, seq_len={model.seq_len}, lr={model.lr}"
        )
        return 0

    if args.command == "lora-smoke":
        model = darkforest.StaticLoRAGPT2(
            vocab_size=65, d_model=64, n_heads=1, n_layers=2, d_ff=256, seq_len=32, lr=1e-3, rank=4, alpha=8.0
        )
        print(f"Instantiated LoRA: {model}")
        print(
            f"Base params: {model.total_base_params}, Trainable LoRA params: {model.trainable_lora_params} ({model.trainable_fraction:.3f}%)"
        )
        return 0

    if args.command == "bench":
        print(f"Benchmarking matrix step: {args.rows}x{args.cols} for {args.steps} steps...")
        stats = darkforest.benchmark_matrix_step(args.steps, args.rows, args.cols)
        for k, v in stats.items():
            print(f"  {k:15}: {v:.3f}")
        return 0

    if args.command == "generate":
        model = darkforest.StaticGPT2(
            vocab_size=128, d_model=128, n_heads=2, n_layers=4, d_ff=512, seq_len=128, lr=1e-3
        )
        prompt_tokens = [ord(c) % 128 for c in args.prompt]
        t0 = time.perf_counter()
        gen = model.generate_fast(
            prompt_tokens,
            args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
        )
        elapsed = time.perf_counter() - t0
        out_text = "".join(chr(t) if 32 <= t < 127 else "?" for t in gen)
        print("Generated Output:")
        print("-------------------------------------------------------")
        print(out_text)
        print("-------------------------------------------------------")
        ms_per_tok = (elapsed * 1000.0) / max(args.tokens, 1)
        tok_s = args.tokens / max(elapsed, 1e-6)
        print(f"Latency: {ms_per_tok:.3f} ms/token | Throughput: {tok_s:.1f} tok/s | Total: {elapsed*1000:.2f} ms")
        return 0

    if args.command == "train":
        print(f"Initializing Dark Forest {'LoRA' if args.lora else 'Static'} Engine...")
        if args.lora:
            model = darkforest.StaticLoRAGPT2(
                vocab_size=128,
                d_model=args.d_model,
                n_heads=2,
                n_layers=args.n_layers,
                d_ff=args.d_model * 4,
                seq_len=args.ctx_len,
                lr=args.lr,
                rank=args.lora_rank,
                alpha=float(args.lora_rank * 2),
            )
            print(f"LoRA active: {model.trainable_lora_params} trainable params ({model.trainable_fraction:.3f}%)")
        else:
            model = darkforest.StaticGPT2(
                vocab_size=128,
                d_model=args.d_model,
                n_heads=2,
                n_layers=args.n_layers,
                d_ff=args.d_model * 4,
                seq_len=args.ctx_len,
                lr=args.lr,
            )

        print(f"Running {args.steps} training steps...")
        tokens = [((i * 7 + 13) % 127) + 1 for i in range(args.ctx_len + 1)]
        in_tok = tokens[:args.ctx_len]
        tgt_tok = tokens[1:args.ctx_len + 1]

        for s in range(1, args.steps + 1):
            loss = model.step(in_tok, tgt_tok)
            if s == 1 or s % 10 == 0 or s == args.steps:
                print(f"Step {s:5d}/{args.steps} | Loss: {loss:.4f}")

        if args.checkpoint_save:
            print(f"Saving checkpoint to {args.checkpoint_save}...")
            if hasattr(model, "save_checkpoint"):
                model.save_checkpoint(args.checkpoint_save)
            elif hasattr(model, "save_lora_checkpoint"):
                model.save_lora_checkpoint(args.checkpoint_save)
            print("Checkpoint saved.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
