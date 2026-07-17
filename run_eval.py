"""
run_eval.py
-----------
Step 4 from the plan: this is the single entry point you actually run.
It runs metrics + hallucination audit + latency benchmark in one go, and
-- critically -- APPENDS a row to a persistent experiment_log.csv every
time you run it. That log is your entire "process tracking" deliverable:
one row per experiment, growing over the semester, ready to paste into
your final report or README with zero extra work.

Run:
    python run_eval.py --model_version v1 --notes "baseline CycleGAN"

    # once real data exists:
    python run_eval.py --gt_dir data/gt --gen_dir data/gen --ir_dir data/ir \\
        --model_version v2 --notes "added semantic loss term"
"""

import argparse
import csv
import os
from datetime import datetime

from utils import ensure_dir, generate_synthetic_dataset
from metrics import compute_psnr_ssim, compute_fid
from hallucination_audit import run_spectral_audit, generate_human_review_csv
from latency_benchmark import benchmark, dummy_inference_fn

LOG_PATH = "experiment_log.csv"
LOG_FIELDS = [
    "timestamp", "model_version", "n_test_images", "avg_psnr", "avg_ssim",
    "fid_score", "fid_warning", "avg_spectral_consistency", "n_flagged_hallucination",
    "mean_latency_ms", "ms_per_megapixel", "notes",
]


def append_to_log(row: dict):
    """Appends one row to experiment_log.csv, writing the header only once."""
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run the full SpectraVyom evaluation suite and log the result.")
    parser.add_argument("--gt_dir", default=None)
    parser.add_argument("--gen_dir", default=None)
    parser.add_argument("--ir_dir", default=None)
    parser.add_argument("--model_version", default="unlabeled", help="Tag this run, e.g. 'v1', 'v2-semantic-loss'")
    parser.add_argument("--notes", default="", help="Free-text note for the experiment log")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--latency_size", type=int, default=512)
    args = parser.parse_args()

    using_synthetic = args.gt_dir is None or args.gen_dir is None or args.ir_dir is None
    if using_synthetic:
        print("[INFO] Incomplete real-data paths given -- using synthetic smoke-test data for ALL steps.")
        print("       This run's numbers are NOT meaningful results -- do not paste them into your report.")
        synth_root = ensure_dir("synthetic_data")
        gt_dir, gen_dir, ir_dir = generate_synthetic_dataset(synth_root)
    else:
        gt_dir, gen_dir, ir_dir = args.gt_dir, args.gen_dir, args.ir_dir

    ensure_dir("results")

    print("\n=== 1/3: PSNR / SSIM / FID ===")
    psnr_ssim = compute_psnr_ssim(gt_dir, gen_dir)
    fid_result = {"fid_score": None, "warning": None} if args.skip_fid else compute_fid(gt_dir, gen_dir)
    print(f"PSNR: {psnr_ssim['avg_psnr']:.2f} dB | SSIM: {psnr_ssim['avg_ssim']:.4f}")
    if fid_result.get("fid_score") is not None:
        print(f"FID: {fid_result['fid_score']:.2f}")
    elif fid_result.get("error"):
        print(f"FID skipped: {fid_result['error']}")

    print("\n=== 2/3: Hallucination audit (spectral proxy) ===")
    audit = run_spectral_audit(ir_dir, gen_dir)
    print(f"Avg spectral consistency: {audit['avg_spectral_consistency']:.3f}")
    print(f"Flagged for human review: {len(audit['flagged_for_review'])} image(s)")
    generate_human_review_csv(gen_dir, "results/human_review_template.csv")

    print("\n=== 3/3: Latency benchmark ===")
    print("[NOTE] Timing placeholder function -- swap dummy_inference_fn for the real model call.")
    latency = benchmark(dummy_inference_fn, size=args.latency_size, n_warmup=5, n_runs=30)
    print(f"Mean latency: {latency['mean_ms']:.2f} ms (+/- {latency['std_ms']:.2f}) on {latency['device']}")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_version": args.model_version,
        "n_test_images": psnr_ssim["n_images"],
        "avg_psnr": round(psnr_ssim["avg_psnr"], 3),
        "avg_ssim": round(psnr_ssim["avg_ssim"], 4),
        "fid_score": round(fid_result["fid_score"], 3) if fid_result.get("fid_score") is not None else "N/A",
        "fid_warning": fid_result.get("warning") or "",
        "avg_spectral_consistency": round(audit["avg_spectral_consistency"], 3),
        "n_flagged_hallucination": len(audit["flagged_for_review"]),
        "mean_latency_ms": round(latency["mean_ms"], 2),
        "ms_per_megapixel": round(latency["ms_per_megapixel"], 2),
        "notes": ("[SYNTHETIC DATA -- not a real result] " if using_synthetic else "") + args.notes,
    }
    append_to_log(row)

    print(f"\n[DONE] Row appended to {LOG_PATH}. This file is your experiment tracker -- keep it in git.")


if __name__ == "__main__":
    main()
