#!/usr/bin/env python3
"""Generate ground-truth masks from manual edits (e.g. the test/ folder).

GT = pixels that are near-white in the manual edit AND not near-white in the
source. Manual-edit files are named <stem>e.<ext>; source files are <stem>.<ext>.

Usage:
    python make_manual_gt.py --source DIR --manual DIR --out DIR [--near-white 225]
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build GT masks from manual edits.")
    ap.add_argument("--source", required=True, help="Directory of source images")
    ap.add_argument("--manual", required=True, help="Directory of manual edits (<stem>e.<ext>)")
    ap.add_argument("--out", required=True, help="Output directory for GT masks")
    ap.add_argument("--near-white", type=int, default=225, help="Min-channel threshold (default 225)")
    args = ap.parse_args(argv)

    src_dir = Path(args.source)
    manual_dir = Path(args.manual)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for manual_path in sorted(manual_dir.iterdir()):
        if not manual_path.is_file():
            continue
        stem = manual_path.stem
        src_stem = stem[:-1] if stem.endswith("e") else stem
        cands = [p for p in src_dir.iterdir() if p.stem == src_stem and p.is_file()]
        if not cands:
            print(f"[skip] no source for {manual_path.name}")
            continue
        src = load(cands[0])
        manual = load(manual_path)
        if src.shape != manual.shape:
            print(f"[skip] shape mismatch {manual_path.name}")
            continue
        near_white = manual.min(axis=2) >= args.near_white
        src_not_white = src.min(axis=2) < args.near_white
        gt = (near_white & src_not_white).astype(np.uint8) * 255
        out_path = out_dir / f"{src_stem}_mask.png"
        Image.fromarray(gt).save(out_path)
        print(f"{src_stem}: {gt.mean() / 255 * 100:.1f}% coverage -> {out_path.name}")


if __name__ == "__main__":
    main()