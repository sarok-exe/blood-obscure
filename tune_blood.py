#!/usr/bin/env python3
"""Parameter sweep harness for obscure_blood.detect_blood.

Sweeps the config dimensions that matter most (saturation floor, LAB A
threshold, redness threshold, morphology kernels, min component area) and
ranks the results by median Dice against ground-truth masks.

Usage:
    python tune_blood.py --images DIR --masks DIR --base-config FILE \
                         --out FILE [--quick]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from obscure_blood import detect_blood, dice_iou, load_gt_mask, load_image_bgr


def build_config(base, s_floor, min_a, min_red, open_k, close_k, min_area):
    """Deep-copy base config with one sweep combo applied."""
    cfg = json.loads(json.dumps(base))
    layers = []
    for layer in base["layers"]:
        l = dict(layer)
        l["hsv_min"] = list(layer["hsv_min"])
        l["hsv_min"][1] = s_floor
        l["min_a"] = min_a
        l["min_redness"] = min_red
        layers.append(l)
    cfg["layers"] = layers
    cfg["morphology"] = {"open_kernel": open_k, "close_kernel": close_k, "dilate": 0}
    cfg["min_area"] = min_area
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sweep detect_blood parameters against ground-truth masks.")
    ap.add_argument("--images", required=True, help="Directory of source images")
    ap.add_argument("--masks", required=True, help="Directory of ground-truth masks (<stem>_mask.png)")
    ap.add_argument("--base-config", required=True, help="Base config JSON to sweep around")
    ap.add_argument("--out", required=True, help="Output JSON with ranked results")
    ap.add_argument("--quick", action="store_true", help="Reduced grid for fast iteration")
    ap.add_argument("--grid", help="JSON file with custom grid lists "
                    "{\"s_floors\":[], \"min_as\":[], \"min_reds\":[], "
                    "\"open_ks\":[], \"close_ks\":[], \"min_areas\":[]}")
    args = ap.parse_args(argv)

    base = json.loads(Path(args.base_config).read_text())

    # Load images + ground truth once.
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    pairs = []
    for img_path in sorted(
        f for f in Path(args.images).iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ):
        gt_path = Path(args.masks) / f"{img_path.stem}_mask.png"
        if not gt_path.exists():
            print(f"[skip] {img_path.name}: no ground-truth mask")
            continue
        bgr = load_image_bgr(img_path)
        gt = load_gt_mask(gt_path, bgr.shape[:2])
        pairs.append((img_path.name, bgr, gt))

    if not pairs:
        print("No images with ground-truth masks found.")
        return 1

    if args.grid:
        g = json.loads(Path(args.grid).read_text())
        s_floors = g["s_floors"]
        min_as = g["min_as"]
        min_reds = g["min_reds"]
        open_ks = g["open_ks"]
        close_ks = g["close_ks"]
        min_areas = g["min_areas"]
    elif args.quick:
        s_floors = [40, 80]
        min_as = [140, 155]
        min_reds = [40, 60]
        open_ks = [0, 3]
        close_ks = [0, 5]
        min_areas = [200, 500]
    else:
        s_floors = [40, 60, 80]
        min_as = [120, 140, 155]
        min_reds = [20, 40, 60]
        open_ks = [0, 3, 5]
        close_ks = [0, 5, 9]
        min_areas = [50, 200, 500]

    total = len(s_floors) * len(min_as) * len(min_reds) * len(open_ks) * len(close_ks) * len(min_areas)
    print(f"Sweeping {total} combos over {len(pairs)} images...")

    results = []
    done = 0
    for sf in s_floors:
        for ma in min_as:
            for mr in min_reds:
                for ok in open_ks:
                    for ck in close_ks:
                        for marea in min_areas:
                            cfg = build_config(base, sf, ma, mr, ok, ck, marea)
                            dices, ious = [], []
                            for _name, bgr, gt in pairs:
                                pred = detect_blood(bgr, cfg)
                                d, i = dice_iou(pred, gt)
                                dices.append(d)
                                ious.append(i)
                            results.append({
                                "params": {
                                    "saturation_floor": sf,
                                    "min_a": ma,
                                    "min_redness": mr,
                                    "open_kernel": ok,
                                    "close_kernel": ck,
                                    "min_area": marea,
                                },
                                "median_dice": float(np.median(dices)),
                                "mean_iou": float(np.mean(ious)),
                            })
                            done += 1
                            if done % 50 == 0 or done == total:
                                print(f"  [{done}/{total}]")

    results.sort(key=lambda r: (-r["median_dice"], -r["mean_iou"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"sweep": "quick" if args.quick else "full", "results": results}, f, indent=2)

    print(f"\nTop 10 (of {len(results)}):")
    print(f"{'#':>3}  {'median_dice':>11} {'mean_iou':>8}  params")
    print("-" * 80)
    for i, r in enumerate(results[:10], 1):
        p = r["params"]
        print(f"{i:>3}  {r['median_dice']:11.3f} {r['mean_iou']:8.3f}  "
              f"s={p['saturation_floor']} a={p['min_a']} r={p['min_redness']} "
              f"open={p['open_kernel']} close={p['close_kernel']} area={p['min_area']}")
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())