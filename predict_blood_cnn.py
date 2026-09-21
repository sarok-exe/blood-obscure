#!/usr/bin/env python3
"""
Blood segmentation CNN inference (pure PyTorch).

Usage:
    python predict_blood_cnn.py image.png [--out mask.png] [--fast]
"""

import argparse
import os
import sys
import pathlib
import numpy as np

from train_blood_cnn import (
    TORCH_AVAILABLE,
    torch,
    HAS_CV2,
    HAS_PIL,
    Image,
    BloodUNet,
    load_image_bgr,
    predict_patches_torch,
    predict_whole_torch,
)


def main():
    parser = argparse.ArgumentParser(description="Blood segmentation inference")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--out", default=None, help="Output mask path (default: <input>_blood.png)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast draft mode (single 384×384 letterbox pass)")
    args = parser.parse_args()

    # Check torch
    if not TORCH_AVAILABLE:
        print("PyTorch not available — cannot run inference.")
        print("Install with: pip install torch")
        sys.exit(0)

    # Check model
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "models", "blood_cnn.pt")
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        print("Train first with: python train_blood_cnn.py")
        sys.exit(0)

    # Load model
    print(f"Loading model from {model_path} ...")
    model = BloodUNet(patch_size=96)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load image (BGR; PIL fallback for webp)
    image = load_image_bgr(args.image)
    print(f"Image: {args.image} — {image.shape[1]}×{image.shape[0]} (W×H)")

    # Predict
    if args.fast:
        print("Mode: fast (single 384×384 letterbox pass)")
        prob = predict_whole_torch(model, image)
    else:
        print("Mode: patch-based (96×96, stride 48)")
        prob = predict_patches_torch(model, image, patch_size=96, stride=48, batch_size=8)

    # Output path
    if args.out:
        out_path = args.out
    else:
        stem = pathlib.Path(args.image).stem
        out_path = os.path.join(os.path.dirname(args.image), f"{stem}_blood.png")

    # Save binary mask (255=blood, 0=bg)
    binary = (prob > 0.5).astype(np.uint8) * 255
    if HAS_CV2:
        import cv2
        cv2.imwrite(out_path, binary)
    else:
        Image.fromarray(binary).save(out_path)

    # Stats
    blood_pct = 100.0 * np.mean(prob > 0.5)
    print(f"Blood pixels: {blood_pct:.1f}%")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()