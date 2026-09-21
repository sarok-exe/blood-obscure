#!/usr/bin/env python3
"""Generate 100x100 solid-color images covering every shade of red.

Each generated image is a single solid color: one shade of red fills the
entire 100x100 canvas. The script walks the red region of HSV color
space and writes one image per shade, so together the images cover every
possible red shade.

Red shades (OpenCV HSV scale: H 0-179, S 0-255, V 0-255):
    H in [0, 20]  and  [160, 180]   (red hue, including the wrap)
    S in [0, 255]                    (all saturations)
    V in [0, 255]                    (all brightnesses)

Total shades: 42 * 256 * 256 = 2,752,512.

Usage:
    python generate_red_shades.py --out ./red_shades            # all shades
    python generate_red_shades.py --out ./red_shades --limit 5000
    python generate_red_shades.py --out ./red_shades --step 4   # every 4th shade
    python generate_red_shades.py --out ./red_shades --size 100
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Red hue values in OpenCV HSV: 0..20 and the wrap 160..180.
RED_HUES = list(range(0, 21)) + list(range(160, 181))


def iter_red_shades(step: int = 1):
    """Yield (h, s, v) for every red shade, optionally sampling every Nth."""
    for h in RED_HUES:
        for s in range(0, 256, step):
            for v in range(0, 256, step):
                yield h, s, v


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="./red_shades",
        help="Output directory (default: ./red_shades)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Image size in pixels — square (default: 100)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Sample every Nth shade (default: 1 = all 2,752,512 shades)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many images (0 = unlimited)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    size = max(1, args.size)
    step = max(1, args.step)
    per_hue = ((255 // step) + 1) ** 2
    total = len(RED_HUES) * per_hue
    if args.limit:
        total = min(total, args.limit)

    if total > 100_000:
        # Solid-color PNGs compress to ~360 bytes each regardless of size.
        est_gb = total * 360 / 1e9
        print(
            f"Generating {total:,} images at {size}x{size} (~{est_gb:.1f} GB "
            f"on disk as PNG, ~{total * size * size * 3 / 1e9:.1f} GB raw). "
            f"ETA roughly {total / 3500:.0f}s at ~3500 images/s."
        )

    t0 = time.time()
    count = 0
    skipped = 0
    for h, s, v in iter_red_shades(step):
        if args.limit and count >= args.limit:
            break

        name = f"red_h{h:03d}_s{s:03d}_v{v:03d}.png"
        out_file = out / name
        if out_file.exists():
            skipped += 1
            count += 1
            continue

        # HSV -> BGR (OpenCV uses 0-180 hue)
        hsv = np.array([[[h, s, v]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        img = np.full((size, size, 3), bgr, dtype=np.uint8)

        cv2.imwrite(str(out_file), img)
        count += 1

        if count % 5000 == 0:
            elapsed = time.time() - t0
            rate = count / elapsed
            eta = (total - count) / rate if rate > 0 else 0
            print(f"  {count:,}/{total:,} images ({rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"Done: {count:,} red-shade images -> {out} in {elapsed:.1f}s"
          f" ({skipped:,} already existed, skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())