#!/usr/bin/env python3
"""CLI tool for obscuring blood in anatomy project images.

Usage:
    # Single image (inpaint mode, default)
    python cli.py image photo.jpg

    # Single image with blur fallback
    python cli.py image photo.jpg --method blur

    # Batch process a folder
    python cli.py batch ./input --output ./output

    # Batch with mask saving for inspection
    python cli.py batch ./input --output ./output --save-mask
"""

import argparse
import sys
from pathlib import Path

from blood_obscure import BloodDetector


def cmd_image(args: argparse.Namespace) -> int:
    """Process a single image."""
    detector = BloodDetector(
        blur_strength=args.blur_strength,
        inpaint_radius=args.radius,
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
        a_channel_threshold=args.a_threshold,
        dilate_pixels=args.dilate,
        blur_passes=args.blur_passes,
        block_size=args.block_size,
    )

    try:
        result, mask = detector.process_image(
            image_path=args.image,
            output_path=args.output,
            method=args.method,
            save_mask=args.save_mask,
            debug=args.debug,
            coords_path=args.coords,
        )
        print(f"Processed: {args.image}")
        if args.output:
            print(f"Saved to:  {args.output}")
        else:
            p = Path(args.image)
            suffix = "_red" if args.debug == "red" else "_black" if args.debug == "black" else "_clean"
            print(f"Saved to:  {p.parent / f'{p.stem}{suffix}{p.suffix}'}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Process all images in a directory."""
    detector = BloodDetector(
        blur_strength=args.blur_strength,
        inpaint_radius=args.radius,
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
        a_channel_threshold=args.a_threshold,
        dilate_pixels=args.dilate,
        blur_passes=args.blur_passes,
        block_size=args.block_size,
    )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        return 1

    print(f"Processing images from: {input_dir}")
    if args.output:
        print(f"Output directory: {args.output}")

    results = detector.process_batch(
        input_dir=input_dir,
        output_dir=args.output,
        method=args.method,
        save_mask=args.save_mask,
    )

    ok = sum(1 for _, s in results if s)
    fail = len(results) - ok
    print(f"\nDone: {ok} processed, {fail} failed")
    return 0 if fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Obscure blood in anatomy project images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["blur", "inpaint", "pixelate"],
        default="blur",
        help="Obscuring method: blur (default), inpaint (natural), "
             "or pixelate (square blur / mosaic)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=100,
        help="Square blur: size of each mosaic square in pixels. Must be "
             "larger than the blood regions so squares swallow the blood "
             "(default: 100)",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=10,
        help="Extra pixels added on each side of the blood region, even "
             "where there is no blood. E.g. a 200px blood area with "
             "--dilate 10 covers 220px (default: 10)",
    )
    parser.add_argument(
        "--blur-passes",
        type=int,
        default=2,
        help="Number of Gaussian blur passes applied (default: 2)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Inpaint radius in pixels (default: 5)",
    )
    parser.add_argument(
        "--blur-strength",
        type=int,
        default=201,
        help="Gaussian blur kernel size, must be odd. Must be much larger than "
             "the blood regions to fully dilute the red signal (default: 201)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0001,
        help="Minimum region area as fraction of image (default: 0.0001)",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=0.15,
        help="Max region area as fraction of image; larger components are "
             "treated as tissue false-positives (default: 0.15)",
    )
    parser.add_argument(
        "--a-threshold",
        type=int,
        default=140,
        help="LAB A-channel threshold for blood confirmation. Lower = more "
             "aggressive detection (may catch warm tissue), higher = stricter "
             "(default: 140)",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the detection mask for inspection",
    )
    parser.add_argument(
        "--debug",
        choices=["red", "black"],
        help="Experiment mode: 'red' paints every detected pixel red on a "
             "grayscale copy; 'black' paints every detected pixel black on "
             "the original (simple covering test)",
    )
    parser.add_argument(
        "--coords",
        help="Save all detected pixel coordinates (x, y) to this JSON file",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- Single image ---
    p_img = sub.add_parser("image", help="Process a single image")
    p_img.add_argument("image", help="Path to input image")
    p_img.add_argument("-o", "--output", help="Output path (default: auto)")
    p_img.set_defaults(func=cmd_image)

    # --- Batch ---
    p_batch = sub.add_parser("batch", help="Process all images in a directory")
    p_batch.add_argument("input_dir", help="Directory with input images")
    p_batch.add_argument("-o", "--output", help="Output directory (default: ./output)")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
