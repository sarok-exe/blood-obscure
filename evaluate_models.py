#!/usr/bin/env python3
"""Evaluate every candidate model on the full red-shade dataset.

Dataset (generated in-memory, vectorized):
  POSITIVE  - all red shades: H 0-20 + 160-180, S 0-255, V 0-255 (2,752,512)
  NEGATIVE  - skin tones: H 0-25, S 30-180, V 100-255 (the real confuser)
            - non-red colors: H 26-159 (all non-red hues) + grays

Models compared:
  1. red_colors.json    - pure hue rule model (exhaustive red)
  2. blood_colors.json  - production blood rule model (LAB/redness floors)
  3. DecisionTree       - sklearn, trained on color features
  4. RandomForest       - sklearn, trained on color features
  5. LogisticRegression - sklearn, trained on color features

Features per pixel: H, S, V, L, A, B, redness (R - max(G,B)).

Metrics: accuracy, red recall (coverage), skin false-positive rate,
non-red false-positive rate, F1.

Usage:
    python evaluate_models.py
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

RED_HUES = list(range(0, 21)) + list(range(160, 181))

TRAIN_POS = 400_000
TRAIN_NEG = 400_000
TEST_POS = 200_000
TEST_NEG = 200_000


def build_shade_hsv(hues, s_range, v_range):
    hs = np.repeat(np.array(hues, dtype=np.uint8), len(s_range) * len(v_range))
    ss = np.tile(
        np.repeat(np.array(s_range, dtype=np.uint8), len(v_range)), len(hues)
    )
    vs = np.tile(np.array(v_range, dtype=np.uint8), len(hues) * len(s_range))
    return np.stack([hs, ss, vs], axis=1)


def features(hsv_shades):
    """Feature matrix: HSV + LAB + redness, one row per shade."""
    bgr = cv2.cvtColor(hsv_shades.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    redness = bgr[:, 2].astype(np.float32) - np.maximum(
        bgr[:, 0], bgr[:, 1]
    ).astype(np.float32)
    return np.column_stack(
        [
            hsv_shades[:, 0].astype(np.float32),  # H
            hsv_shades[:, 1].astype(np.float32),  # S
            hsv_shades[:, 2].astype(np.float32),  # V
            lab[:, 0].astype(np.float32),         # L
            lab[:, 1].astype(np.float32),         # A
            lab[:, 2].astype(np.float32),         # B
            redness,                              # R - max(G,B)
        ]
    )


def rule_model_predict(hsv_shades, layers):
    """Rule-model prediction using the exact build_mask layer logic."""
    bgr = cv2.cvtColor(hsv_shades.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    redness = bgr[:, 2].astype(int) - np.maximum(bgr[:, 0], bgr[:, 1]).astype(int)

    pred = np.zeros(len(hsv_shades), dtype=bool)
    for layer in layers:
        lo = np.array(layer["hsv_min"], dtype=np.uint8)
        hi = np.array(layer["hsv_max"], dtype=np.uint8)
        min_a = int(layer.get("min_a", 140))
        min_red = int(layer.get("min_redness", 40))
        m = np.all((hsv_shades >= lo) & (hsv_shades <= hi), axis=1)
        m &= lab[:, 1] > min_a
        m &= redness > min_red
        if "max_b" in layer:
            m &= lab[:, 2] < int(layer["max_b"])
        pred |= m
    return pred


def report(name, y_true, y_pred, y_skin, y_nonred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    skin_fp = float(np.mean(y_pred[y_skin])) if y_skin.sum() else 0.0
    nonred_fp = float(np.mean(y_pred[y_nonred])) if y_nonred.sum() else 0.0
    print(f"  {name:<22} acc={acc:.4f}  red_recall={rec:.4f}  "
          f"skin_FP={skin_fp:.4f}  nonred_FP={nonred_fp:.4f}  F1={f1:.4f}")
    return {"name": name, "acc": acc, "recall": rec, "skin_fp": skin_fp,
            "nonred_fp": nonred_fp, "f1": f1}


def main() -> int:
    t0 = time.time()
    print("Building dataset...")

    # Positives: all red shades
    pos = build_shade_hsv(RED_HUES, range(256), range(256))
    # Negatives: skin tones + non-red hues + grays
    skin = build_shade_hsv(range(0, 26), range(30, 181), range(100, 256))
    nonred = build_shade_hsv(range(26, 160), range(0, 256, 4), range(0, 256, 4))
    gray = build_shade_hsv([0], [0], range(0, 256))
    neg = np.concatenate([skin, nonred, gray])

    print(f"  positives: {len(pos):,} red shades")
    print(f"  negatives: {len(neg):,} (skin {len(skin):,} + non-red {len(nonred):,} + gray {len(gray):,})")

    # Train/test split
    rng = np.random.default_rng(42)
    pos_idx = rng.choice(len(pos), TRAIN_POS + TEST_POS, replace=False)
    train_pos, test_pos = pos[pos_idx[:TRAIN_POS]], pos[pos_idx[TRAIN_POS:]]
    neg_idx = rng.choice(len(neg), TRAIN_NEG + TEST_NEG, replace=False)
    train_neg, test_neg = neg[neg_idx[:TRAIN_NEG]], neg[neg_idx[TRAIN_NEG:]]

    X_train = features(np.concatenate([train_pos, train_neg]))
    y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    X_test = features(np.concatenate([test_pos, test_neg]))
    y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    # Split test into skin vs non-red for the FP breakdown (exact tuple match)
    skin_set = set(map(tuple, skin.tolist()))
    is_skin = np.array([tuple(row) in skin_set for row in test_neg])
    test_skin = np.concatenate([np.zeros(len(test_pos), bool), is_skin])
    test_nonred = np.concatenate([np.zeros(len(test_pos), bool), ~is_skin])

    print(f"  train: {len(X_train):,}  test: {len(X_test):,}  "
          f"(built in {time.time()-t0:.1f}s)\n")

    results = []

    # --- Rule models ---
    print("Rule models:")
    for name, path in [("red_colors.json", "red_colors.json"),
                       ("blood_colors.json", "blood_colors.json")]:
        with open(path) as f:
            layers = json.load(f)["layers"]
        pred = rule_model_predict(np.concatenate([test_pos, test_neg]), layers)
        results.append(report(name, y_test, pred, test_skin, test_nonred))

    # --- ML models ---
    print("\nML models (training on", len(X_train), "samples):")
    models = [
        ("DecisionTree", DecisionTreeClassifier(max_depth=12, random_state=42)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=100, max_depth=14, n_jobs=-1, random_state=42)),
        ("LogisticRegression", LogisticRegression(max_iter=500, n_jobs=-1)),
    ]
    for name, model in models:
        tm = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append(report(name, y_test, pred, test_skin, test_nonred))
        print(f"    (trained in {time.time()-tm:.1f}s)")

    # --- Summary ---
    print("\nSummary (higher acc/F1 better; skin_FP is the critical number):")
    for r in sorted(results, key=lambda r: -r["f1"]):
        print(f"  {r['name']:<22} acc={r['acc']:.4f}  red_recall={r['recall']:.4f}  "
              f"skin_FP={r['skin_fp']:.4f}  nonred_FP={r['nonred_fp']:.4f}  F1={r['f1']:.4f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())