"""
latency_benchmark.py
---------------------
Step 3 from the plan. Benchmarks how long inference takes, correctly:

  - Discards the first N "cold start" runs (model/CUDA warmup, disk cache,
    etc. makes the first few runs slower and non-representative).
  - Reports mean +/- std over the remaining runs, not just a single number.
  - Reports per-megapixel time so results are comparable across different
    tile sizes.
  - Detects GPU (via torch, if installed) and reports which device was
    used -- a latency number without the hardware is not reproducible
    and not usable by anyone else on your team.

This file benchmarks a PLACEHOLDER inference function by default (so you
can confirm the harness itself works). Replace `dummy_inference_fn` with
a call into the actual model once the modeling team hands it over --
that is the ONLY line that needs to change.

Run:
    python latency_benchmark.py --n_warmup 5 --n_runs 50 --size 512
"""

import argparse
import time
import json
import numpy as np

from utils import ensure_dir


def detect_device():
    """Reports what hardware inference would run on. Torch is optional --
    if it's not installed yet, this just reports CPU and moves on."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "CPU (torch installed, but no CUDA GPU detected)"
    except ImportError:
        return "CPU (torch not installed -- install it to benchmark on GPU)"


def dummy_inference_fn(ir_tile):
    """
    PLACEHOLDER for the real model call. Currently just does a cheap
    numpy operation to simulate 'doing work' on a tile, so the benchmark
    harness has something real to time.

    TO USE WITH YOUR ACTUAL MODEL:
    Replace the body of this function with something like:
        with torch.no_grad():
            output = model(ir_tile_tensor)
        return output
    Everything else in this file (warmup handling, stats, per-megapixel
    normalization) stays exactly the same.
    """
    return np.fft.fft2(ir_tile.astype(np.float32)).real  # arbitrary stand-in workload


def benchmark(inference_fn, size=512, n_warmup=5, n_runs=50, seed=0):
    rng = np.random.default_rng(seed)
    tile = rng.integers(0, 255, (size, size), dtype=np.uint8)

    # --- Warmup: run and DISCARD these timings ---
    for _ in range(n_warmup):
        inference_fn(tile)

    # --- Timed runs ---
    times_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        inference_fn(tile)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    times_ms = np.array(times_ms)
    megapixels = (size * size) / 1_000_000

    return {
        "device": detect_device(),
        "tile_size": size,
        "n_warmup_discarded": n_warmup,
        "n_timed_runs": n_runs,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "ms_per_megapixel": float(np.mean(times_ms) / megapixels),
        "raw_times_ms": times_ms.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark harness (cold-start-aware).")
    parser.add_argument("--size", type=int, default=512, help="Simulated tile size (pixels, square)")
    parser.add_argument("--n_warmup", type=int, default=5, help="Warmup runs to discard")
    parser.add_argument("--n_runs", type=int, default=50, help="Timed runs to average over")
    parser.add_argument("--out", default="results/latency.json")
    args = parser.parse_args()

    results = benchmark(dummy_inference_fn, size=args.size, n_warmup=args.n_warmup, n_runs=args.n_runs)

    ensure_dir("results")
    # Keep raw times out of the console summary but keep them in the JSON for later analysis
    summary = {k: v for k, v in results.items() if k != "raw_times_ms"}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Device: {results['device']}")
    print(f"Tile size: {args.size}x{args.size} ({args.size*args.size/1_000_000:.2f} MP)")
    print(f"Mean latency: {results['mean_ms']:.2f} ms (+/- {results['std_ms']:.2f}), over {args.n_runs} runs after {args.n_warmup} warmup discards")
    print(f"Per-megapixel: {results['ms_per_megapixel']:.2f} ms/MP")
    print(f"\nNOTE: this timed a PLACEHOLDER function (see dummy_inference_fn in this file).")
    print(f"Swap in the real model call before reporting these numbers anywhere.")
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
