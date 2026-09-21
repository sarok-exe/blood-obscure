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
        use_sam=args.sam,
        colors_path=args.colors,
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
        use_sam=args.sam,
        colors_path=args.colors,
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


def cmd_detect(args: argparse.Namespace) -> int:
    """Stage 1: detect blood and write pixel coordinates to JSON."""
    detector = BloodDetector(
        blur_strength=args.blur_strength,
        inpaint_radius=args.radius,
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
        a_channel_threshold=args.a_threshold,
        dilate_pixels=args.dilate,
        blur_passes=args.blur_passes,
        block_size=args.block_size,
        use_sam=args.sam,
        colors_path=args.colors,
    )

    try:
        detector.detect_only(
            image_path=args.image,
            coords_path=args.coords,
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_edit(args: argparse.Namespace) -> int:
    """Stage 2: read coordinates JSON, paint those pixels."""
    detector = BloodDetector(
        blur_strength=args.blur_strength,
        inpaint_radius=args.radius,
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
        a_channel_threshold=args.a_threshold,
        dilate_pixels=args.dilate,
        blur_passes=args.blur_passes,
        block_size=args.block_size,
        use_sam=args.sam,
        colors_path=args.colors,
    )

    try:
        detector.edit_from_coords(
            image_path=args.image,
            coords_path=args.coords,
            output_path=args.output,
            method=args.method,
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    # --- Shared flags inherited by every subcommand ---
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--method",
        choices=["white", "pixelate", "blur", "inpaint"],
        default="white",
        help="Covering method: white (paint blood pixels white, default), "
             "pixelate (square mosaic), blur (Gaussian), inpaint (natural)",
    )
    common.add_argument(
        "--block-size",
        type=int,
        default=100,
        help="Square blur: size of each mosaic square in pixels (default: 100)",
    )
    common.add_argument(
        "--dilate",
        type=int,
        default=10,
        help="Extra pixels on each side of blood region (default: 10)",
    )
    common.add_argument(
        "--blur-passes",
        type=int,
        default=2,
        help="Number of Gaussian blur passes (default: 2)",
    )
    common.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Inpaint radius in pixels (default: 5)",
    )
    common.add_argument(
        "--blur-strength",
        type=int,
        default=201,
        help="Gaussian blur kernel size, must be odd (default: 201)",
    )
    common.add_argument(
        "--min-area",
        type=float,
        default=0.0001,
        help="Min region area as fraction of image (default: 0.0001)",
    )
    common.add_argument(
        "--max-area",
        type=float,
        default=1.0,
        help="Max region area as fraction of image (default: 1.0 = no limit)",
    )
    common.add_argument(
        "--a-threshold",
        type=int,
        default=140,
        help="LAB A-channel threshold for blood (default: 140)",
    )
    common.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the detection mask for inspection",
    )
    common.add_argument(
        "--debug",
        choices=["red", "black"],
        help="Experiment mode: 'red' or 'black' paint detection overlay",
    )
    common.add_argument(
        "--coords",
        help="Save detected pixel coordinates (x, y) to this JSON file",
    )
    common.add_argument(
        "--sam",
        action="store_true",
        help="Refine the mask with MobileSAM (ONNX, CPU). Experimental — "
             "slower and currently under-segments/over-segments vs color-only.",
    )
    common.add_argument(
        "--colors",
        help="Path to color layers JSON file (default: blood_colors.json). "
             "Use red_colors.json for the exhaustive red model.",
    )

    parser = argparse.ArgumentParser(
        description="Obscure blood in anatomy project images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- Single image ---
    p_img = sub.add_parser("image", parents=[common], help="Process a single image")
    p_img.add_argument("image", help="Path to input image")
    p_img.add_argument("-o", "--output", help="Output path (default: auto)")
    p_img.set_defaults(func=cmd_image)

    # --- Batch ---
    p_batch = sub.add_parser("batch", parents=[common],
                             help="Process all images in a directory")
    p_batch.add_argument("input_dir", help="Directory with input images")
    p_batch.add_argument("-o", "--output", help="Output directory (default: ./output)")
    p_batch.set_defaults(func=cmd_batch)

    # --- Detect (stage 1) ---
    p_detect = sub.add_parser(
        "detect", parents=[common],
        help="Stage 1: detect blood, write pixel coordinates to JSON",
    )
    p_detect.add_argument("image", help="Path to input image")
    p_detect.add_argument(
        "coords",
        help="Output JSON file with detected pixel coordinates (x, y)",
    )
    p_detect.set_defaults(func=cmd_detect)

    # --- Edit (stage 2) ---
    p_edit = sub.add_parser(
        "edit", parents=[common],
        help="Stage 2: read coordinates JSON, paint those pixels",
    )
    p_edit.add_argument("image", help="Path to original input image")
    p_edit.add_argument(
        "coords",
        help="JSON file with pixel coordinates (from detect)",
    )
    p_edit.add_argument("-o", "--output", help="Output path (default: auto)")
    p_edit.set_defaults(func=cmd_edit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
