# run_all.ps1 — Execute all Dark Forest and PyTorch benchmarks and regenerate committed JSON results.

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " Dark Forest Benchmark Suite — Full Reproducibility Run" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Environment telemetry
Write-Host "`n--- System & Toolchain Telemetry ---" -ForegroundColor Yellow
python --version
nvcc --version
cargo --version
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"

# 2. PyTorch SDPA Strong Baseline (with checkpointing, chunked CE, and without checkpointing)
Write-Host "`n--- Step 1: Running PyTorch SDPA Baseline Sweeps ---" -ForegroundColor Yellow
python benchmark/bench_sdpa_baseline.py

# 3. Dark Forest End-to-End Model Training Benchmark
Write-Host "`n--- Step 2: Running Dark Forest End-to-End Model Benchmark ---" -ForegroundColor Yellow
python benchmark/bench_darkforest_e2e.py

# 4. Attention Kernel OOM Horizon Benchmark (Kernel Isolation)
Write-Host "`n--- Step 3: Running Attention Kernel Horizon Benchmark ---" -ForegroundColor Yellow
cargo run --release --bin bench_sequence_limit --features cuda

# 5. Verification Test against README
Write-Host "`n--- Step 4: Validating README Tables against Generated JSON Results ---" -ForegroundColor Yellow
python scripts/test_readme_tables.py

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host " Benchmark Suite Complete. All JSON files regenerated and verified." -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green

