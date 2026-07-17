"""
utils.py
--------
Shared helpers used by every other script in this folder.

Why this file exists (read this first):
The modeling team hasn't handed you real IR->RGB outputs yet. Rather than
wait, this module generates FAKE-but-structured data so you can prove your
evaluation code is correct TODAY. When real data arrives, you only change
the folder paths passed to --gt_dir / --gen_dir on the command line. The
code does not change.

Folder convention every script in this project expects:
    some_folder/
        gt/       <- ground-truth RGB images (real Landsat RGB bands)
        gen/      <- model-generated RGB images (output of your IR->RGB model)
        ir/       <- original low-res IR input (used only by hallucination audit)
    Filenames must match across the three folders, e.g. gt/tile_001.png,
    gen/tile_001.png, ir/tile_001.png refer to the SAME satellite tile.
"""

import os
import numpy as np
import cv2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def generate_synthetic_dataset(root, n_images=8, size=256, seed=0):
    """
    Creates a tiny fake dataset with matching gt/gen/ir folders so the
    pipeline can be smoke-tested without waiting on the modeling team.

    The 'gen' images are the 'gt' images with noise + a color shift added,
    which mimics what an imperfect IR->RGB model output looks like
    (close to ground truth, but not identical) -- good enough to sanity
    check that metrics behave sensibly (i.e. not perfect, not garbage).
    """
    rng = np.random.default_rng(seed)
    gt_dir = ensure_dir(os.path.join(root, "gt"))
    gen_dir = ensure_dir(os.path.join(root, "gen"))
    ir_dir = ensure_dir(os.path.join(root, "ir"))

    for i in range(n_images):
        name = f"tile_{i:03d}.png"

        # Fake "ground truth" RGB: random smooth blobs so SSIM/PSNR have
        # actual structure to compare (pure random noise breaks SSIM).
        base = rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8)
        gt = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)

        # Fake "model output": gt + noise + slight hue shift, simulating
        # realistic imperfection instead of a trivial identical copy.
        noise = rng.normal(0, 10, gt.shape).astype(np.int16)
        gen = np.clip(gt.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Mild hue shift: blend 85% original channels with 15% rolled
        # channels, simulating a model that's mostly right but slightly
        # miscolors things -- rather than a total color swap.
        rolled = np.roll(gen, shift=1, axis=2)
        gen = np.clip(0.85 * gen.astype(np.float32) + 0.15 * rolled.astype(np.float32), 0, 255).astype(np.uint8)

        # Fake single-channel IR input: grayscale version of gt, downsampled
        # then upsampled to mimic real low-res IR characteristics.
        ir = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        ir_small = cv2.resize(ir, (size // 4, size // 4))
        ir = cv2.resize(ir_small, (size, size))

        cv2.imwrite(os.path.join(gt_dir, name), gt)
        cv2.imwrite(os.path.join(gen_dir, name), gen)
        cv2.imwrite(os.path.join(ir_dir, name), ir)

    return gt_dir, gen_dir, ir_dir


def load_image_pairs(gt_dir, gen_dir):
    """
    Loads matching gt/gen image pairs by filename. Returns a list of
    (filename, gt_array, gen_array). Skips files that don't have a match
    on both sides and prints a warning -- silent mismatches are how you
    end up comparing tile_003 against tile_030 without noticing.
    """
    gt_files = {f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))}
    gen_files = {f for f in os.listdir(gen_dir) if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))}
    common = sorted(gt_files & gen_files)

    missing_in_gen = gt_files - gen_files
    missing_in_gt = gen_files - gt_files
    if missing_in_gen:
        print(f"[WARN] {len(missing_in_gen)} files in gt/ have no match in gen/: {sorted(missing_in_gen)[:5]}...")
    if missing_in_gt:
        print(f"[WARN] {len(missing_in_gt)} files in gen/ have no match in gt/: {sorted(missing_in_gt)[:5]}...")

    pairs = []
    for fname in common:
        gt_img = cv2.imread(os.path.join(gt_dir, fname))
        gen_img = cv2.imread(os.path.join(gen_dir, fname))
        if gt_img is None or gen_img is None:
            print(f"[WARN] could not read {fname}, skipping")
            continue
        if gt_img.shape != gen_img.shape:
            gen_img = cv2.resize(gen_img, (gt_img.shape[1], gt_img.shape[0]))
        pairs.append((fname, gt_img, gen_img))

    return pairs
