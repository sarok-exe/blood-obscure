#!/usr/bin/env python3
"""Test every model variant on the test images, one folder per image.

For each image in the test folder a subfolder is created containing:
  <name>_mltree_white.png      - ML tree (th=0.5), white fill
  <name>_mltree_mask.png       - ML tree mask
  <name>_rule_white.png        - current blood rule model, white fill
  <name>_rule_mask.png         - current blood rule model mask
  <name>_hybrid_white.png      - ML tree + region growing, white fill
  <name>_hybrid_mask.png       - ML tree + region growing mask
  <name>_red_white.png         - exhaustive red model, white fill (reference)
  <name>_red_mask.png          - exhaustive red model mask
  <name>_metrics.txt           - per-model metrics

Usage:
    python test_models.py [input_dir] [output_dir]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from blood_obscure import BloodDetector
from evaluate_models import (
    RED_HUES,
    TRAIN_POS,
    TRAIN_NEG,
    build_shade_hsv,
    features,
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "red_classifier.joblib"


def train_or_load_tree():
    if MODEL_PATH.exists():
        import joblib
        return joblib.load(MODEL_PATH)
    pos = build_shade_hsv(RED_HUES, range(256), range(256))
    skin = build_shade_hsv(range(0, 26), range(30, 181), range(100, 256))
    nonred = build_shade_hsv(range(26, 160), range(0, 256, 4), range(0, 256, 4))
    gray = build_shade_hsv([0], [0], range(0, 256))
    neg = np.concatenate([skin, nonred, gray])
    rng = np.random.default_rng(42)
    pi = rng.choice(len(pos), TRAIN_POS, replace=False)
    ni = rng.choice(len(neg), TRAIN_NEG, replace=False)
    X = features(np.concatenate([pos[pi], neg[ni]]))
    y = np.concatenate([np.ones(TRAIN_POS), np.zeros(TRAIN_NEG)])
    clf = DecisionTreeClassifier(max_depth=12, random_state=42).fit(X, y)
    import joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return clf


def tree_mask(clf, bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    red = r.astype(np.float32) - np.maximum(g, b).astype(np.float32)
    X = np.column_stack([
        hsv[:, :, 0].ravel().astype(np.float32),
        hsv[:, :, 1].ravel().astype(np.float32),
        hsv[:, :, 2].ravel().astype(np.float32),
        lab[:, :, 0].ravel().astype(np.float32),
        lab[:, :, 1].ravel().astype(np.float32),
        lab[:, :, 2].ravel().astype(np.float32),
        red.ravel(),
    ])
    return clf.predict(X).reshape(bgr.shape[:2]).astype(np.uint8) * 255


def region_grow(base, cand):
    mask = base.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(80):
        new = cv2.dilate(mask, k3) & cand & ~mask
        if new.sum() == 0:
            break
        mask |= new
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def grow_candidates(bgr):
    """dark_clotted_red layer from blood_colors.json (grow_only)."""
    layers = json.load(open(Path(__file__).resolve().parent / "blood_colors.json"))["layers"]
    grow = [l for l in layers if l.get("grow_only")][0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    redness = r.astype(int) - np.maximum(g, b).astype(int)
    lo = np.array(grow["hsv_min"]); hi = np.array(grow["hsv_max"])
    cand = cv2.inRange(hsv, lo, hi)
    cand = cv2.bitwise_and(cand, (lab[:, :, 1] > grow["min_a"]).astype(np.uint8) * 255)
    cand = cv2.bitwise_and(cand, (redness > grow["min_redness"]).astype(np.uint8) * 255)
    cand = cv2.bitwise_and(cand, (lab[:, :, 2] < grow["max_b"]).astype(np.uint8) * 255)
    return cand


def white_fill(bgr, mask):
    out = bgr.copy()
    out[mask > 0] = (255, 255, 255)
    return out


def metrics(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(bgr)
    red_px = (r.astype(int) - np.maximum(g, b).astype(int)) > 40
    bs = (hsv[:, :, 2] > 130) & (hsv[:, :, 1] > 60) & (
        (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 25)
    )
    m = mask > 0
    total = bgr.shape[0] * bgr.shape[1]
    return {
        "mask_pct": round(100.0 * m.sum() / total, 2),
        "red_covered_pct": round(100.0 * (m & red_px).sum() / max(red_px.sum(), 1), 2),
        "bright_skin_covered_pct": round(100.0 * (m & bs).sum() / max(bs.sum(), 1), 2),
    }


def main() -> int:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Pictures/test"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Pictures/test/models_output"

    clf = train_or_load_tree()
    rule = BloodDetector()
    red = BloodDetector(colors_path=Path(__file__).resolve().parent / "red_colors.json")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in exts)
    if not images:
        print(f"No images found in {input_dir}")
        return 1

    for img_path in images:
        folder = output_dir / img_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[skip] {img_path.name}")
            continue

        print(f"\n=== {img_path.name} -> {folder} ===")

        # ML tree
        m_ml = tree_mask(clf, bgr)
        cv2.imwrite(str(folder / f"{img_path.stem}_mltree_mask.png"), m_ml)
        cv2.imwrite(str(folder / f"{img_path.stem}_mltree_white.png"), white_fill(bgr, m_ml))
        met_ml = metrics(bgr, m_ml)

        # Rule model
        m_rule = rule.build_mask(bgr)
        cv2.imwrite(str(folder / f"{img_path.stem}_rule_mask.png"), m_rule)
        cv2.imwrite(str(folder / f"{img_path.stem}_rule_white.png"), white_fill(bgr, m_rule))
        met_rule = metrics(bgr, m_rule)

        # Hybrid: ML tree + region growing
        m_hyb = region_grow(m_ml, grow_candidates(bgr))
        cv2.imwrite(str(folder / f"{img_path.stem}_hybrid_mask.png"), m_hyb)
        cv2.imwrite(str(folder / f"{img_path.stem}_hybrid_white.png"), white_fill(bgr, m_hyb))
        met_hyb = metrics(bgr, m_hyb)

        # Exhaustive red model (reference)
        m_red = red.build_mask(bgr)
        cv2.imwrite(str(folder / f"{img_path.stem}_red_mask.png"), m_red)
        cv2.imwrite(str(folder / f"{img_path.stem}_red_white.png"), white_fill(bgr, m_red))
        met_red = metrics(bgr, m_red)

        # Metrics file
        with open(folder / "metrics.txt", "w") as f:
            f.write(f"image: {img_path.name}\n")
            for label, met in [("mltree", met_ml), ("rule", met_rule),
                               ("hybrid", met_hyb), ("red", met_red)]:
                f.write(f"{label}: mask={met['mask_pct']}%  "
                        f"red_covered={met['red_covered_pct']}%  "
                        f"bright_skin_covered={met['bright_skin_covered_pct']}%\n")

        print(f"  mltree: mask={met_ml['mask_pct']}%  red={met_ml['red_covered_pct']}%  "
              f"skin={met_ml['bright_skin_covered_pct']}%")
        print(f"  rule  : mask={met_rule['mask_pct']}%  red={met_rule['red_covered_pct']}%  "
              f"skin={met_rule['bright_skin_covered_pct']}%")
        print(f"  hybrid: mask={met_hyb['mask_pct']}%  red={met_hyb['red_covered_pct']}%  "
              f"skin={met_hyb['bright_skin_covered_pct']}%")
        print(f"  red   : mask={met_red['mask_pct']}%  red={met_red['red_covered_pct']}%  "
              f"skin={met_red['bright_skin_covered_pct']}%")

    print(f"\nAll outputs in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())