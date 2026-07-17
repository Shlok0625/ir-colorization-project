# SpectraVyom — Evaluation, Documentation & Process Tracking

This folder covers: metrics computation, hallucination audit, latency
benchmarking, and experiment tracking for the SpectraVyom IR→RGB pipeline.

## 1. Setup

```bash
cd spectravyom_eval
pip install -r requirements.txt
# Optional, only needed for FID / GPU latency / detection mAP:
# pip install clean-fid torch torchvision torchmetrics
```

## 2. Folder convention (required before using real data)

```
data/
  gt/     ground-truth RGB images (real Landsat RGB bands)
  gen/    model-generated RGB images (your model's output)
  ir/     original low-res IR input tiles
```
Filenames must match across all three folders (`gt/tile_001.png` ==
`gen/tile_001.png` == `ir/tile_001.png`, same physical tile). The scripts
warn you if files are missing on one side — don't ignore that warning,
it means you're about to compare mismatched tiles.

**Until real data exists**, every script below runs on synthetic
dummy data automatically if you don't pass `--gt_dir/--gen_dir/--ir_dir`.
This confirms the code works, but the numbers are not real results —
each script prints a warning saying so. Do not paste synthetic numbers
into any report.

## 3. Running it

```bash
# One-shot: runs everything and logs the result
python run_eval.py --model_version v1 --notes "baseline CycleGAN"

# Individual pieces:
python metrics.py --gt_dir data/gt --gen_dir data/gen
python hallucination_audit.py --ir_dir data/ir --gen_dir data/gen
python latency_benchmark.py --size 512 --n_runs 50
```

Every `run_eval.py` call appends one row to `experiment_log.csv` —
commit this file to git. It is your entire process-tracking deliverable;
by the end of the project it's a ready-made results table.

## 4. Evaluation protocol (read this before trusting any number)

- **Test set must be frozen.** Same held-out images used across every
  model version compared. If the modeling team changes the test set
  between runs, numbers in `experiment_log.csv` are not comparable —
  check the `notes` column before comparing rows.
- **FID needs ~2000+ images to be statistically reliable.** Below that,
  treat the number as directional only. The script flags this
  automatically in its output.
- **The hallucination audit is a proxy, not ground truth.** The spectral
  consistency check uses IR brightness as a stand-in for a real NDVI
  (which needs true NIR + Red bands). Replace it with real NDVI once you
  have the actual multi-band Landsat files — this is flagged in
  `hallucination_audit.py`'s docstring and output.
- **Human review needs 2–3 independent raters**, not one person's
  opinion. Use `compute_inter_rater_agreement()` in
  `hallucination_audit.py` to check raters actually agree before
  reporting a mean fabrication score — low agreement means fix the
  rubric, not just average away the disagreement.
- **Latency numbers require the real model swapped in.** `latency_benchmark.py`
  times a placeholder function by default so the harness can be tested
  before the real model exists. Swap `dummy_inference_fn` for the real
  inference call — that is the only change needed.
- **Report hardware explicitly.** `latency_benchmark.py` auto-detects
  and logs CPU vs GPU; never report a bare millisecond number without it.

## 5. Known limitations to disclose in your final report

1. Vegetation proxy in the hallucination audit is derived from IR
   brightness, not true NDVI — approximation, not ground truth.
2. FID reliability depends on test set size — check before quoting it
   as a definitive score.
3. Detection mAP (if used) requires a pretrained/team-trained detector
   and `torchmetrics`/`pycocotools` — not included by default due to
   install weight; add when the modeling team has outputs ready.
