"""
hallucination_audit.py
-----------------------
Step 2 from the plan. There is no off-the-shelf "hallucination score"
library for IR->RGB translation -- this script builds the proxy audit
described earlier, in two parts:

  A. Spectral-consistency proxy (automated, no manual work):
     Compares a vegetation-ness signal derived from the IR input against
     a vegetation-ness signal derived from the model's RGB output. Large
     mismatches suggest the model painted "green vegetation" where the
     source data doesn't support it -- a concrete, checkable hallucination
     signature specific to remote sensing (a generic CV eval would miss this).

     NOTE: A real NDVI needs a near-infrared band + a red band from the
     actual satellite product. This script approximates using the single
     IR channel you have as a stand-in, and documents that assumption
     explicitly in the output -- swap in true NDVI the moment you have
     the actual Landsat band files.

  B. Structured human review rubric (semi-manual):
     Generates a CSV with one row per test image and empty scoring
     columns. 2-3 teammates independently fill in scores 1-5 for
     "fabricated detail" (invented buildings/roads/objects not
     supported by the IR input). This script also computes inter-rater
     agreement once the CSV is filled in, so you're not just reporting
     a mean score with no reliability check.

Run:
    python hallucination_audit.py --ir_dir data/ir --gen_dir data/gen --out results/
"""

import argparse
import csv
import os
import numpy as np
import cv2

from utils import load_image_pairs, generate_synthetic_dataset, ensure_dir


def vegetation_proxy_from_ir(ir_gray):
    """
    Crude proxy: in the ABSENCE of a real NIR band, treat brighter IR
    pixels as a stand-in for "warmer surface" (often less vegetation,
    since vegetation tends to run cooler in thermal IR during the day).
    This is a placeholder assumption -- replace with true NDVI
    (NIR-Red)/(NIR+Red) once you have actual Landsat band files.
    Returns a 0-1 map, higher = more "vegetation-like" by this proxy.
    """
    norm = ir_gray.astype(np.float32) / 255.0
    return 1.0 - norm  # cooler (darker IR) => higher vegetation-likeness, per assumption above


def vegetation_proxy_from_rgb(rgb):
    """
    Proxy 'greenness' from generated RGB: how much the green channel
    dominates red+blue. This is the ExG (Excess Green) index, a real
    and commonly used vegetation index for RGB-only imagery.
    """
    b, g, r = cv2.split(rgb.astype(np.float32))
    exg = 2 * g - r - b
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    return exg_norm


def spectral_consistency_score(ir_gray, gen_rgb):
    """
    Correlates the IR-derived proxy and RGB-derived greenness proxy.
    High correlation = model's colorization is consistent with what the
    IR input actually supports. Low/negative correlation = candidate
    hallucination (model invented vegetation/color not backed by input).
    Returns Pearson correlation coefficient, -1 to 1.
    """
    veg_ir = vegetation_proxy_from_ir(ir_gray).flatten()
    veg_rgb = vegetation_proxy_from_rgb(gen_rgb).flatten()
    if np.std(veg_ir) == 0 or np.std(veg_rgb) == 0:
        return 0.0
    corr = np.corrcoef(veg_ir, veg_rgb)[0, 1]
    return float(corr)


def run_spectral_audit(ir_dir, gen_dir):
    """Runs the automated proxy check across every ir/gen pair, flags low-consistency outliers."""
    pairs = load_image_pairs(ir_dir, gen_dir)  # reuses same matching logic; ir/ and gen/ filenames must align
    if not pairs:
        raise RuntimeError("No matching ir/gen pairs found -- check folder paths and filenames.")

    results = []
    for fname, ir_img, gen_img in pairs:
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if ir_img.ndim == 3 else ir_img
        score = spectral_consistency_score(ir_gray, gen_img)
        results.append({"file": fname, "spectral_consistency": score})

    scores = [r["spectral_consistency"] for r in results]
    avg = float(np.mean(scores))
    # Flag anything meaningfully below the batch average as worth a human look
    threshold = avg - 0.3
    flagged = [r["file"] for r in results if r["spectral_consistency"] < threshold]

    return {
        "avg_spectral_consistency": avg,
        "threshold_used": threshold,
        "flagged_for_review": flagged,
        "per_image": results,
        "assumption_note": ("Vegetation proxy derived from single IR channel brightness, NOT true NDVI. "
                            "Replace with real NIR/Red band NDVI once available from Landsat product."),
    }


def generate_human_review_csv(gen_dir, out_path, n_raters=3):
    """
    Creates a CSV template: one row per image, blank columns for each
    rater to fill in a 1-5 fabricated-detail score. Ship this to your
    2-3 teammates, collect it back filled in, then run
    `compute_inter_rater_agreement()` on the result.
    """
    files = sorted(f for f in os.listdir(gen_dir) if f.lower().endswith((".png", ".jpg", ".tif", ".tiff")))
    rater_cols = [f"rater{i+1}_score_1to5" for i in range(n_raters)]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", *rater_cols, "notes"])
        for fname in files:
            writer.writerow([fname, *([""] * n_raters), ""])

    print(f"[INFO] Human review template written to {out_path}.")
    print("       Rubric to send raters: score 1-5 how much INVENTED detail")
    print("       (buildings/roads/objects not supported by the IR input) appears.")
    print("       1 = no fabrication, 5 = heavily fabricated.")


def compute_inter_rater_agreement(filled_csv_path):
    """
    Once teammates fill in the CSV, run this to get:
      - mean score per rater (checks if one rater is systematically harsher)
      - pairwise correlation between raters (checks if they agree at all --
        low agreement means the rubric itself needs to be clearer, not that
        the images are ambiguous)
    """
    import csv as _csv
    rows = []
    with open(filled_csv_path) as f:
        reader = _csv.DictReader(f)
        rater_cols = [c for c in reader.fieldnames if c.startswith("rater")]
        for row in reader:
            rows.append(row)

    scores_by_rater = {c: [] for c in rater_cols}
    for row in rows:
        for c in rater_cols:
            val = row.get(c, "").strip()
            if val != "":
                scores_by_rater[c].append(float(val))

    means = {c: float(np.mean(v)) if v else None for c, v in scores_by_rater.items()}

    # Pairwise correlation between raters who have complete overlapping data
    pairwise_corr = {}
    cols = list(scores_by_rater.keys())
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = scores_by_rater[cols[i]], scores_by_rater[cols[j]]
            if len(a) == len(b) and len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
                pairwise_corr[f"{cols[i]}_vs_{cols[j]}"] = float(np.corrcoef(a, b)[0, 1])

    return {"mean_score_per_rater": means, "pairwise_correlation": pairwise_corr}


def main():
    parser = argparse.ArgumentParser(description="Hallucination audit: spectral proxy check + human review template.")
    parser.add_argument("--ir_dir", default=None)
    parser.add_argument("--gen_dir", default=None)
    parser.add_argument("--out", default="results/")
    parser.add_argument("--n_raters", type=int, default=3)
    args = parser.parse_args()

    if args.ir_dir is None or args.gen_dir is None:
        print("[INFO] No --ir_dir/--gen_dir given -- generating synthetic smoke-test data instead.")
        synth_root = ensure_dir("synthetic_data")
        _, gen_dir, ir_dir = generate_synthetic_dataset(synth_root)
    else:
        ir_dir, gen_dir = args.ir_dir, args.gen_dir

    ensure_dir(args.out)

    audit = run_spectral_audit(ir_dir, gen_dir)
    import json
    with open(os.path.join(args.out, "spectral_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\nAvg spectral consistency: {audit['avg_spectral_consistency']:.3f} (higher = more consistent, less likely hallucinated)")
    print(f"Flagged for human review: {len(audit['flagged_for_review'])} image(s)")

    csv_path = os.path.join(args.out, "human_review_template.csv")
    generate_human_review_csv(gen_dir, csv_path, n_raters=args.n_raters)


if __name__ == "__main__":
    main()
