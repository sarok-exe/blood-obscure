#!/usr/bin/env python3
"""Config-driven blood obscuring for anatomy school images.

Pure OpenCV color segmentation -- no ML, no training. Detection settings
live in a JSON config (see obscure_config.json). Supports three obscuring
methods: white fill, adaptive Gaussian blur, and block pixelation.

Usage:
    python obscure_blood.py --input DIR [--output DIR] [--config FILE]
                            [--method white|blur|pixelate] [--eval-masks DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# --------------------------------------------------------------------------- #
#  Loading / saving
# --------------------------------------------------------------------------- #

def load_image_bgr(path):
    """Load an image (any PIL-supported format incl. webp) as a BGR array."""
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_gt_mask(path, shape):
    """Load a binary ground-truth mask (white=blood) and match image size."""
    arr = np.asarray(Image.open(path).convert("L"))
    if arr.shape != tuple(shape):
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 127).astype(np.uint8) * 255


def save_image_bgr(bgr, path):
    """Save a BGR array via PIL (handles webp output)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


# --------------------------------------------------------------------------- #
#  Detection
# --------------------------------------------------------------------------- #

def detect_blood(bgr, config):
    """Detect blood pixels -> uint8 mask (0/255).

    Pipeline: HSV/LAB/redness layer matching -> region growing from
    grow_only candidates -> morphology -> connected-component filter.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    redness = r.astype(int) - np.maximum(g, b).astype(int)

    seed = np.zeros(bgr.shape[:2], dtype=np.uint8)
    candidates = np.zeros(bgr.shape[:2], dtype=np.uint8)

    for layer in config["layers"]:
        lo = np.array(layer["hsv_min"], dtype=np.uint8)
        hi = np.array(layer["hsv_max"], dtype=np.uint8)
        min_a = int(layer.get("min_a", 140))
        min_red = int(layer.get("min_redness", 40))

        m = cv2.inRange(hsv, lo, hi)
        m = cv2.bitwise_and(m, (lab[:, :, 1] > min_a).astype(np.uint8) * 255)
        m = cv2.bitwise_and(m, (redness > min_red).astype(np.uint8) * 255)
        if "max_b" in layer:
            m = cv2.bitwise_and(m, (lab[:, :, 2] < int(layer["max_b"])).astype(np.uint8) * 255)

        if layer.get("grow_only"):
            candidates = cv2.bitwise_or(candidates, m)
        else:
            seed = cv2.bitwise_or(seed, m)

    # Region growing: accept grow_only pixels adjacent to the seed.
    mask = seed.copy()
    grow_iter = int(config.get("grow_iterations", 80))
    kernel3 = np.ones((3, 3), np.uint8)
    for _ in range(grow_iter):
        new = cv2.dilate(mask, kernel3) & candidates & ~mask
        if new.sum() == 0:
            break
        mask |= new

    # Morphology cleanup (0 = skip).
    morph = config.get("morphology", {})
    open_k = int(morph.get("open_kernel", 0))
    close_k = int(morph.get("close_kernel", 0))
    dilate_k = int(morph.get("dilate", 0))
    if open_k > 0 and open_k % 2 == 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 0 and close_k % 2 == 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    if dilate_k > 0:
        mask = cv2.dilate(mask, np.ones((dilate_k, dilate_k), np.uint8))

    # Connected-component filter: drop skin speckles below min_area.
    min_area = int(config.get("min_area", 200))
    if min_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        mask = clean

    return mask


# --------------------------------------------------------------------------- #
#  Obscuring
# --------------------------------------------------------------------------- #

def obscure(bgr, mask, method, config):
    """Apply an obscuring method inside the mask. Returns a new BGR image."""
    if method == "white":
        result = bgr.copy()
        result[mask > 0] = (255, 255, 255)
        return result

    if method == "blur":
        # Adaptive kernel: 2x the largest blood blob dimension (odd, min 5).
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            max_dim = max(
                max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                for i in range(1, num_labels)
            )
            k = max(5, (max_dim * 2) | 1)
        else:
            k = 15
        blurred = cv2.GaussianBlur(bgr, (k, k), 0)
        if config.get("feather", False):
            # Soft edge: blend with a 2px-Gaussian-blurred mask.
            soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0).astype(np.float32) / 255.0
            soft = soft[..., None]
            return (soft * blurred + (1.0 - soft) * bgr).astype(np.uint8)
        result = bgr.copy()
        result[mask > 0] = blurred[mask > 0]
        return result

    if method == "pixelate":
        h, w = bgr.shape[:2]
        bs = max(1, int(config.get("pixelate_block", 12)))
        small = cv2.resize(bgr, (max(1, w // bs), max(1, h // bs)), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result = bgr.copy()
        result[mask > 0] = pixelated[mask > 0]
        return result

    raise ValueError(f"Unknown obscuring method: {method!r}")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def dice_iou(pred, gt):
    """Dice coefficient and IoU between two binary masks."""
    p = pred > 0
    g = gt > 0
    inter = int((p & g).sum())
    if inter == 0:
        if int(p.sum()) == 0 and int(g.sum()) == 0:
            return 1.0, 1.0
        return 0.0, 0.0
    dice = 2.0 * inter / (int(p.sum()) + int(g.sum()))
    iou = inter / int((p | g).sum())
    return dice, iou


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def collect_images(input_dir):
    return sorted(
        f for f in Path(input_dir).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def run_eval(images, mask_dir, config):
    rows = []
    for img_path in images:
        gt_path = Path(mask_dir) / f"{img_path.stem}_mask.png"
        if not gt_path.exists():
            print(f"[skip] {img_path.name}: no ground-truth mask")
            continue
        bgr = load_image_bgr(img_path)
        pred = detect_blood(bgr, config)
        gt = load_gt_mask(gt_path, pred.shape)
        dice, iou = dice_iou(pred, gt)
        rows.append((img_path.name, dice, iou))

    if not rows:
        print("No images with ground-truth masks found.")
        return

    print(f"\n{'image':<48} {'dice':>7} {'iou':>7}")
    print("-" * 66)
    for name, dice, iou in rows:
        print(f"{name:<48} {dice:7.3f} {iou:7.3f}")

    dices = np.array([r[1] for r in rows])
    ious = np.array([r[2] for r in rows])
    print("-" * 66)
    print(f"{'median':<48} {np.median(dices):7.3f} {np.median(ious):7.3f}")
    print(f"{'mean':<48} {dices.mean():7.3f} {ious.mean():7.3f}")
    print(f"({len(rows)} images evaluated)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Config-driven blood obscuring (no ML).")
    ap.add_argument("--input", required=True, help="Directory of images to process")
    ap.add_argument("--output", default=None, help="Directory for obscured output (default: stats only)")
    ap.add_argument("--config", default=None, help="Path to config JSON (default: obscure_config.json next to script)")
    ap.add_argument("--method", default="white", choices=["white", "blur", "pixelate"], help="Obscuring method")
    ap.add_argument("--eval-masks", default=None, metavar="DIR", help="Ground-truth mask dir -> evaluation mode")
    args = ap.parse_args(argv)

    config_path = Path(args.config) if args.config else Path(__file__).resolve().parent / "obscure_config.json"
    config = json.loads(config_path.read_text())

    images = collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return 1

    if args.eval_masks:
        run_eval(images, args.eval_masks, config)
        return 0

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        bgr = load_image_bgr(img_path)
        mask = detect_blood(bgr, config)
        n = int((mask > 0).sum())
        pct = 100.0 * n / mask.size
        print(f"{img_path.name}: {n} px ({pct:.2f}%)")
        if output_dir:
            result = obscure(bgr, mask, args.method, config)
            out_path = output_dir / f"{img_path.stem}_obscured{img_path.suffix}"
            save_image_bgr(result, out_path)
            print(f"  -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())