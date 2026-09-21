#!/usr/bin/env python3
"""Verify the red model covers EVERY shade of red.

1. Enumerates the full red HSV space (42 x 256 x 256 = 2,752,512 shades),
   converts each to BGR, and tests it against red_colors.json using the
   exact same layer logic as BloodDetector.build_mask.
2. Tests skin-tone reference shades against the red model — the honest
   tradeoff: skin IS reddish, so a pure red model covers skin too.
3. Runs the red model on the real anatomy photos and reports how much
   skin it would cover (why blood_colors.json is the blood detector).

Usage:
    python verify_red_model.py [image1 image2 ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from blood_obscure import BloodDetector

RED_HUES = list(range(0, 21)) + list(range(160, 181))


def build_shade_hsv(hues, s_range, v_range):
    """Vectorized enumeration of (h, s, v) shades -> uint8 HSV array."""
    hs = np.repeat(np.array(hues, dtype=np.uint8), len(s_range) * len(v_range))
    ss = np.tile(
        np.repeat(np.array(s_range, dtype=np.uint8), len(v_range)),
        len(hues),
    )
    vs = np.tile(np.array(v_range, dtype=np.uint8), len(hues) * len(s_range))
    return np.stack([hs, ss, vs], axis=1)


def layer_match(hsv, lab, redness, layer):
    """Exact copy of the per-layer logic in BloodDetector.build_mask."""
    lo = np.array(layer["hsv_min"], dtype=np.uint8)
    hi = np.array(layer["hsv_max"], dtype=np.uint8)
    min_a = int(layer.get("min_a", 140))
    min_red = int(layer.get("min_redness", 40))

    m = np.all((hsv >= lo) & (hsv <= hi), axis=1)
    m &= lab[:, 1] > min_a
    m &= redness > min_red
    if "max_b" in layer:
        m &= lab[:, 2] < int(layer["max_b"])
    return m


def test_shades(hsv_shades, layers):
    """Return boolean array: which shades match ANY layer."""
    bgr = cv2.cvtColor(hsv_shades.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    redness = bgr[:, 2].astype(int) - np.maximum(bgr[:, 0], bgr[:, 1]).astype(int)

    matched = np.zeros(len(hsv_shades), dtype=bool)
    for layer in layers:
        matched |= layer_match(hsv_shades, lab, redness, layer)
    return matched


def main() -> int:
    colors_path = Path(__file__).resolve().parent / "red_colors.json"
    with open(colors_path) as f:
        import json
        layers = json.load(f)["layers"]

    # --- 1. All red shades ---
    print("Enumerating all red shades (2,752,512)...")
    hsv = build_shade_hsv(RED_HUES, range(256), range(256))
    matched = test_shades(hsv, layers)
    total = len(hsv)
    covered = int(matched.sum())
    print(f"  red shades tested : {total:,}")
    print(f"  covered by model  : {covered:,} ({100.0 * covered / total:.4f}%)")
    print(f"  missed            : {total - covered:,}")
    if covered < total:
        miss = hsv[~matched][:10]
        print(f"  first missed HSV  : {[tuple(int(x) for x in m) for m in miss]}")

    # --- 2. Skin-tone reference shades ---
    print("\nSkin-tone reference shades (H 0-25, S 30-180, V 100-255)...")
    skin_hsv = build_shade_hsv(range(0, 26), range(30, 181), range(100, 256))
    skin_matched = test_shades(skin_hsv, layers)
    skin_total = len(skin_hsv)
    skin_covered = int(skin_matched.sum())
    print(f"  skin shades tested: {skin_total:,}")
    print(f"  red model covers  : {skin_covered:,} ({100.0 * skin_covered / skin_total:.1f}%)")
    print("  -> a pure red model covers skin too. This is why the blood")
    print("     model keeps min_a/min_redness + region growing.")

    # --- 3. Real images ---
    images = sys.argv[1:]
    if images:
        print("\nReal-image check (red model, white fill):")
        detector = BloodDetector(colors_path=colors_path)
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [skip] cannot read {img_path}")
                continue
            mask = detector.build_mask(img)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            b, g, r = cv2.split(img)
            redness = r.astype(int) - np.maximum(g, b).astype(int)

            white_px = int((mask > 0).sum())
            total_px = img.shape[0] * img.shape[1]
            red_px = int((redness > 40).sum())
            red_covered = int(((mask > 0) & (redness > 40)).sum())
            # bright-skin proxy: white pixels V>130, H 5-25
            bright_skin = (hsv_img[:, :, 2] > 130) & (hsv_img[:, :, 1] > 60) & (
                (hsv_img[:, :, 0] >= 5) & (hsv_img[:, :, 0] <= 25)
            )
            skin_covered = int(((mask > 0) & bright_skin).sum())
            skin_total = int(bright_skin.sum())

            name = Path(img_path).name
            print(f"  {name}:")
            print(f"    mask covers {white_px:,}/{total_px:,} px ({100.0 * white_px / total_px:.1f}%)")
            print(f"    red pixels covered: {red_covered:,}/{red_px:,} ({100.0 * red_covered / max(red_px, 1):.1f}%)")
            print(f"    bright-skin covered: {skin_covered:,}/{skin_total:,} ({100.0 * skin_covered / max(skin_total, 1):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())