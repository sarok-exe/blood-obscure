#!/usr/bin/env python3
"""Train the best classifier and test it on real anatomy photos.

Trains a DecisionTree (near-identical to RandomForest at 1/30th the cost)
on the red-shade + skin + non-red dataset, then applies it per-pixel to
real photos and compares against the current blood rule model.

Metrics on real photos:
  - red pixels covered (R-G>40)   : how much blood is caught
  - bright-skin covered (V>130,H 5-25,S>60): how much skin is wrongly hit

Usage:
    python train_best_model.py [image1 image2 ...]
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from evaluate_models import (
    RED_HUES,
    TRAIN_POS,
    TRAIN_NEG,
    build_shade_hsv,
    features,
)

OUT_MODEL = Path(__file__).resolve().parent / "models" / "red_classifier.joblib"


def train() -> DecisionTreeClassifier:
    print("Building training dataset...")
    pos = build_shade_hsv(RED_HUES, range(256), range(256))
    skin = build_shade_hsv(range(0, 26), range(30, 181), range(100, 256))
    nonred = build_shade_hsv(range(26, 160), range(0, 256, 4), range(0, 256, 4))
    gray = build_shade_hsv([0], [0], range(0, 256))
    neg = np.concatenate([skin, nonred, gray])

    rng = np.random.default_rng(42)
    pos_idx = rng.choice(len(pos), TRAIN_POS, replace=False)
    neg_idx = rng.choice(len(neg), TRAIN_NEG, replace=False)
    X = features(np.concatenate([pos[pos_idx], neg[neg_idx]]))
    y = np.concatenate([np.ones(TRAIN_POS), np.zeros(TRAIN_NEG)])

    print(f"  training DecisionTree on {len(X):,} samples...")
    t0 = time.time()
    clf = DecisionTreeClassifier(max_depth=12, random_state=42)
    clf.fit(X, y)
    print(f"  trained in {time.time()-t0:.1f}s, {clf.get_n_leaves():,} leaves")

    try:
        import joblib
        joblib.dump(clf, OUT_MODEL)
        print(f"  saved -> {OUT_MODEL}")
    except ImportError:
        print("  (joblib not available, model not saved)")

    return clf


def predict_image(clf, bgr):
    """Per-pixel classifier prediction -> binary mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    redness = r.astype(np.float32) - np.maximum(g, b).astype(np.float32)
    h, w = bgr.shape[:2]
    X = np.column_stack([
        hsv[:, :, 0].ravel().astype(np.float32),
        hsv[:, :, 1].ravel().astype(np.float32),
        hsv[:, :, 2].ravel().astype(np.float32),
        lab[:, :, 0].ravel().astype(np.float32),
        lab[:, :, 1].ravel().astype(np.float32),
        lab[:, :, 2].ravel().astype(np.float32),
        redness.ravel(),
    ])
    pred = clf.predict(X)
    return pred.reshape(h, w).astype(np.uint8) * 255


def main() -> int:
    clf = train()

    images = sys.argv[1:]
    if not images:
        print("\nNo images given — training only.")
        return 0

    from blood_obscure import BloodDetector
    blood = BloodDetector()  # current production rule model

    print("\nReal-photo comparison (ML tree vs current blood rule model):")
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] {img_path}")
            continue

        ml_mask = predict_image(clf, img)
        rule_mask = blood.build_mask(img)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(img)
        redness = r.astype(int) - np.maximum(g, b).astype(int)
        red_px = redness > 40
        bright_skin = (hsv_img[:, :, 2] > 130) & (hsv_img[:, :, 1] > 60) & (
            (hsv_img[:, :, 0] >= 5) & (hsv_img[:, :, 0] <= 25)
        )
        total = img.shape[0] * img.shape[1]

        name = Path(img_path).name
        print(f"  {name}:")
        for label, mask in [("ML tree", ml_mask), ("rule     ", rule_mask)]:
            m = mask > 0
            red_cov = int((m & red_px).sum())
            skin_cov = int((m & bright_skin).sum())
            print(f"    {label}: mask={100.0*m.sum()/total:6.1f}%  "
                  f"red covered={100.0*red_cov/max(red_px.sum(),1):5.1f}%  "
                  f"bright-skin covered={100.0*skin_cov/max(bright_skin.sum(),1):5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())