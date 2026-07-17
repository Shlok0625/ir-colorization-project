"""
metrics.py
----------
Step 1 from the plan: metrics computation using LIBRARIES, not hand-rolled math.

Two tiers:
  1. PSNR / SSIM  -> scikit-image. Always available, no GPU needed.
  2. FID          -> clean-fid (optional heavy dependency, imported lazily
                     so this file doesn't crash for people who haven't
                     installed torch yet).

Run standalone:
    python metrics.py --gt_dir data/gt --gen_dir data/gen --out results/metrics.json

If --gt_dir/--gen_dir are omitted, it builds a synthetic dataset first so
you can confirm the script itself works before real data exists.
"""

import argparse
import json
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import load_image_pairs, generate_synthetic_dataset, ensure_dir


def compute_psnr_ssim(gt_dir, gen_dir):
    """
    Returns per-image and averaged PSNR/SSIM.

    PSNR (Peak Signal-to-Noise Ratio): higher = closer pixel-level match
    to ground truth. Sensitive to overall error magnitude, not perceptual
    quality -- a blurry-but-close image can score well.

    SSIM (Structural Similarity): 0-1, higher is better. Compares local
    luminance/contrast/structure instead of raw pixel error, so it
    correlates better with what a human would judge as "looks right".

    Why both: PSNR catches gross pixel error, SSIM catches structural
    distortion. Report both -- a panel will ask why you didn't if you skip one.
    """
    pairs = load_image_pairs(gt_dir, gen_dir)
    if not pairs:
        raise RuntimeError("No matching gt/gen image pairs found -- check your folder paths and filenames.")

    per_image = []
    for fname, gt, gen in pairs:
        psnr_val = peak_signal_noise_ratio(gt, gen, data_range=255)
        # channel_axis=2 tells skimage these are HxWxC color images, not grayscale
        ssim_val = structural_similarity(gt, gen, data_range=255, channel_axis=2)
        per_image.append({"file": fname, "psnr": float(psnr_val), "ssim": float(ssim_val)})

    avg_psnr = float(np.mean([p["psnr"] for p in per_image]))
    avg_ssim = float(np.mean([p["ssim"] for p in per_image]))

    return {
        "n_images": len(per_image),
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "per_image": per_image,
    }


def compute_fid(gt_dir, gen_dir):
    """
    FID (Frechet Inception Distance): lower = better, compares the
    DISTRIBUTION of generated images vs ground truth using deep features,
    not pixel-by-pixel. This is the metric that catches "technically sharp
    but statistically wrong" outputs that PSNR/SSIM miss.

    IMPORTANT CAVEAT you must put in your report: FID is unstable below
    ~2000 images. If your test set has 8 or 50 images, report the number
    but explicitly flag it as directional, not a reliable absolute score.

    Requires: pip install clean-fid torch torchvision
    Lazily imported so the rest of this script works without those installed.
    """
    try:
        from cleanfid import fid
    except ImportError:
        return {
            "error": "clean-fid not installed. Run: pip install clean-fid torch torchvision --break-system-packages",
            "fid_score": None,
        }

    score = fid.compute_fid(gt_dir, gen_dir, mode="clean")
    n_images = len(os.listdir(gt_dir))
    warning = None
    if n_images < 2000:
        warning = (f"Only {n_images} images -- FID is statistically unreliable below ~2000. "
                   "Report this number as directional only, not a final benchmark.")

    return {"fid_score": float(score), "n_images": n_images, "warning": warning}


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/FID between ground-truth and generated RGB images.")
    parser.add_argument("--gt_dir", default=None, help="Folder of ground-truth RGB images")
    parser.add_argument("--gen_dir", default=None, help="Folder of model-generated RGB images")
    parser.add_argument("--out", default="results/metrics.json", help="Where to save results JSON")
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID (useful if torch isn't installed yet)")
    args = parser.parse_args()

    if args.gt_dir is None or args.gen_dir is None:
        print("[INFO] No --gt_dir/--gen_dir given -- generating synthetic smoke-test data instead.")
        synth_root = ensure_dir("synthetic_data")
        gt_dir, gen_dir, _ = generate_synthetic_dataset(synth_root)
    else:
        gt_dir, gen_dir = args.gt_dir, args.gen_dir

    results = {"psnr_ssim": compute_psnr_ssim(gt_dir, gen_dir)}

    if not args.skip_fid:
        results["fid"] = compute_fid(gt_dir, gen_dir)
    else:
        results["fid"] = {"skipped": True}

    ensure_dir(os.path.dirname(args.out) or ".")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAvg PSNR: {results['psnr_ssim']['avg_psnr']:.2f} dB")
    print(f"Avg SSIM: {results['psnr_ssim']['avg_ssim']:.4f}")
    if "fid_score" in results["fid"] and results["fid"]["fid_score"] is not None:
        print(f"FID: {results['fid']['fid_score']:.2f}")
        if results["fid"].get("warning"):
            print(f"[WARN] {results['fid']['warning']}")
    elif results["fid"].get("error"):
        print(f"[SKIPPED] {results['fid']['error']}")

    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
