#!/usr/bin/env python3
"""
Evaluate blood segmentation: compare CNN predictions vs ground-truth masks
on held-out validation images (CNN vs old color-heuristic baseline), and run
inference on mystery test photos.

Usage:
    python evaluate_blood_cnn.py [--images DIR] [--masks DIR] [--test DIR]
"""

import argparse
import os
import sys
import pathlib
import glob
import numpy as np

from train_blood_cnn import (
    TORCH_AVAILABLE,
    torch,
    BloodUNet,
    load_image_bgr,
    load_mask,
    find_pairs,
    predict_patches_torch,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray):
    """Compute recall, precision, Dice, IoU given binary masks (0/1)."""
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    dice = (2 * tp) / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)

    return {
        "recall": float(recall),
        "precision": float(precision),
        "dice": float(dice),
        "iou": float(iou),
    }


def color_detector_baseline(image_bgr: np.ndarray):
    """
    Simple red-dominant heuristic baseline:
    Pixels where R > 100 AND R > 2*B AND R > 1.3*G → blood.
    """
    b, g, r = image_bgr[:, :, 0], image_bgr[:, :, 1], image_bgr[:, :, 2]
    mask = (r > 100) & (r > 2.0 * b) & (r > 1.3 * g)
    return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate blood segmentation model")
    parser.add_argument("--images", default="/home/sarok/Pictures/selfharm",
                        help="Directory with blood images")
    parser.add_argument("--masks", default="/home/sarok/Pictures/selfharm/masks",
                        help="Directory with mask PNGs")
    parser.add_argument("--test", default="/home/sarok/Pictures/test",
                        help="Directory with mystery test photos")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Part 1: Validation evaluation
    # ------------------------------------------------------------------
    val_stems = {
        "istockphoto-1412306826-612x612",
        "istockphoto-1145017396-612x612",
    }

    pairs = find_pairs(args.images, args.masks)
    val_pairs = [(i, m, s) for i, m, s in pairs if s in val_stems]

    print(f"\n{'='*60}")
    print("VALIDATION EVALUATION")
    print(f"{'='*60}")

    # Load model if available
    model = None
    if not TORCH_AVAILABLE:
        print("PyTorch not available — CNN metrics skipped.")
        print("Install with: pip install torch\n")
    else:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(project_dir, "models", "blood_cnn.pt")
        if not os.path.isfile(model_path):
            print(f"Model not found: {model_path}")
            print("Train first with: python train_blood_cnn.py\n")
        else:
            print(f"Loading model: {model_path}")
            model = BloodUNet(patch_size=96)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

    for img_path, msk_path, stem in val_pairs:
        print(f"\n--- {stem} ---")
        image = load_image_bgr(img_path)
        gt = load_mask(msk_path)
        gt_bin = (gt > 127).astype(np.uint8)

        # CNN prediction (if model available)
        if model is not None:
            prob = predict_patches_torch(model, image, patch_size=96, stride=48, batch_size=8)
            pred_bin = (prob > 0.5).astype(np.uint8)

            cnn_metrics = compute_metrics(pred_bin, gt_bin)
            print(f"  CNN  — Recall: {cnn_metrics['recall']:.3f}, "
                  f"Precision: {cnn_metrics['precision']:.3f}, "
                  f"Dice: {cnn_metrics['dice']:.3f}, "
                  f"IoU: {cnn_metrics['iou']:.3f}")

            cnn_blood = 100.0 * np.mean(pred_bin > 0)
            gt_blood = 100.0 * np.mean(gt_bin > 0)
            print(f"  Blood % — CNN: {cnn_blood:.1f}%, GT: {gt_blood:.1f}%")

        # Color baseline (always runs — no torch needed)
        baseline_mask = color_detector_baseline(image)
        base_metrics = compute_metrics(baseline_mask, gt_bin)
        print(f"  BASE — Recall: {base_metrics['recall']:.3f}, "
              f"Precision: {base_metrics['precision']:.3f}, "
              f"Dice: {base_metrics['dice']:.3f}, "
              f"IoU: {base_metrics['iou']:.3f}")

    # ------------------------------------------------------------------
    # Part 2: Mystery test photos
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("MYSTERY TEST PHOTOS (no ground truth)")
    print(f"{'='*60}")

    test_dir = args.test
    if not os.path.isdir(test_dir):
        print(f"Test directory not found: {test_dir}")
    else:
        test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
        if not test_files:
            print(f"No .jpg files found in {test_dir}")
        elif model is None:
            print("PyTorch/model not available — cannot run CNN inference.")
            print("Install with: pip install torch")
        else:
            for tf_path in test_files:
                image = load_image_bgr(tf_path)
                prob = predict_patches_torch(model, image, patch_size=96, stride=48, batch_size=8)
                blood_pct = 100.0 * np.mean(prob > 0.5)
                name = pathlib.Path(tf_path).name
                print(f"  {name}: predicted blood = {blood_pct:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()