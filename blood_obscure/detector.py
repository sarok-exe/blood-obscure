"""Core blood detection and obscuring engine."""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class BloodDetector:
    """Detects blood/red regions in anatomy images and obscures them.

    Uses HSV color space with tiered thresholds for fresh and dried blood,
    morphological cleanup, and connected-component filtering.

    Supports two obscuring modes:
    - inpaint (default): reconstructs region from surrounding pixels (natural look)
    - blur: Gaussian blur over detected regions (fallback, faster)
    """

    def __init__(
        self,
        blur_strength: int = 201,
        inpaint_radius: int = 3,
        min_area_ratio: float = 0.0001,
        max_area_ratio: float = 0.15,
        a_channel_threshold: int = 140,
        dilate_pixels: int = 5,
        blur_passes: int = 2,
        block_size: int = 100,
        margin_mm: float = 2.0,
        dpi: int = 300,
    ):
        self.blur_strength = blur_strength
        self.inpaint_radius = inpaint_radius
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.a_channel_threshold = a_channel_threshold
        self.dilate_pixels = dilate_pixels
        self.blur_passes = blur_passes
        self.block_size = block_size
        self.margin_mm = margin_mm
        self.dpi = dpi

    @property
    def effective_dilation(self) -> int:
        """Coverage margin in pixels.

        If dilate_pixels is explicitly set (> 0) it wins; otherwise the
        margin is converted from millimeters using the image DPI:
        px = mm / 25.4 * dpi.
        """
        if self.dilate_pixels > 0:
            return self.dilate_pixels
        return max(1, round(self.margin_mm / 25.4 * self.dpi))

    # ------------------------------------------------------------------ #
    #  HSV blood ranges (OpenCV scale: H 0-180, S 0-255, V 0-255)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fresh_blood_range() -> Tuple[np.ndarray, np.ndarray]:
        """Bright, oxygenated red — fresh blood (H 0-10)."""
        return np.array([0, 60, 50]), np.array([10, 255, 255])

    @staticmethod
    def _fresh_blood_range2() -> Tuple[np.ndarray, np.ndarray]:
        """Wrap-around range for red (H 160-180)."""
        return np.array([160, 60, 50]), np.array([180, 255, 255])

    @staticmethod
    def _dried_blood_range() -> Tuple[np.ndarray, np.ndarray]:
        """Dried/coagulated blood — orange-shifted hue, must be saturated.

        Narrow H band (8-15) and high S floor (100) keep warm tissue out;
        the A-channel filter (applied later) rejects the rest.
        """
        return np.array([8, 100, 40]), np.array([15, 255, 255])

    # ------------------------------------------------------------------ #
    #  Mask pipeline
    # ------------------------------------------------------------------ #

    def _a_channel_mask(self, bgr: np.ndarray) -> np.ndarray:
        """LAB A-channel confirmation: blood is strongly red (A high),
        warm tissue is only mildly red. Rejects tissue false positives."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return (lab[:, :, 1] > self.a_channel_threshold).astype(np.uint8) * 255

    def build_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Build a clean binary mask of blood regions.

        Pipeline: HSV threshold (tiered) -> LAB A-channel confirmation
                  -> morphological cleanup -> component area filter.
        """
        # 1. HSV thresholding — dual-range for red + narrow dried-blood tier
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo1, hi1 = self._fresh_blood_range()
        lo2, hi2 = self._fresh_blood_range2()
        lo3, hi3 = self._dried_blood_range()

        mask = (
            cv2.inRange(hsv, lo1, hi1)
            | cv2.inRange(hsv, lo2, hi2)
            | cv2.inRange(hsv, lo3, hi3)
        )

        # 2. LAB A-channel confirmation — rejects warm tissue (muscle, skin)
        mask = cv2.bitwise_and(mask, self._a_channel_mask(bgr))

        # 3. Morphological cleanup — CLOSE then OPEN with elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Connected-component filter — drop tiny noise AND giant tissue blobs.
        #    A component covering > max_area_ratio of the frame is almost
        #    certainly warm tissue, not blood.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = int(bgr.shape[0] * bgr.shape[1] * self.min_area_ratio)
        max_area = int(bgr.shape[0] * bgr.shape[1] * self.max_area_ratio)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):  # skip background
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
                clean[labels == i] = 255

        return clean

    # ------------------------------------------------------------------ #
    #  Obscuring methods
    # ------------------------------------------------------------------ #

    def _inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Obscure blood via inpainting — reconstructs from surrounding pixels.

        Uses a soft-edged blend: the inpainted region is feathered into the
        original via a blurred mask, eliminating hard seams/halos at the
        boundary.
        """
        # Dilate mask slightly to cover edge halos
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        inpainted = cv2.inpaint(bgr, dilated, self.inpaint_radius, cv2.INPAINT_TELEA)

        # Soft edge: feather the mask so the transition is gradual
        soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0).astype(np.float32) / 255.0
        soft = soft[..., None]
        return (soft * inpainted + (1.0 - soft) * bgr).astype(np.uint8)

    def _blur(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Obscure blood via strong Gaussian blur.

        Each blood pixel is expanded to a (2*effective_dilation + 1)
        square before blurring. The kernel must be much larger than the
        blood regions so the red signal is diluted into surrounding tissue
        instead of smeared into a pink haze. Multiple passes strengthen
        the effect.
        """
        dilation = self.effective_dilation
        if dilation > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilation)
        blurred = bgr.copy()
        for _ in range(max(1, self.blur_passes)):
            blurred = cv2.GaussianBlur(
                blurred, (self.blur_strength, self.blur_strength), 0
            )
        return np.where(mask[..., None] > 0, blurred, bgr)

    def _pixelate(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Obscure blood via square blur (mosaic/pixelation).

        The image is divided into a grid of squares, each block_size x
        block_size pixels. Every square is filled with the average color
        of its pixels, producing large gathered squares. Blocks that
        touch the (dilated) blood mask are replaced — the mosaic is
        allowed to spill outside the blood region.
        """
        dilation = self.effective_dilation
        if dilation > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilation)

        h, w = bgr.shape[:2]
        bs = max(1, self.block_size)

        # Downscale then upscale with nearest neighbor = blocky squares
        small_h, small_w = max(1, h // bs), max(1, w // bs)
        small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return np.where(mask[..., None] > 0, pixelated, bgr)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def process_image(
        self,
        image_path: str | Path,
        output_path: Optional[str | Path] = None,
        method: str = "inpaint",
        save_mask: bool = False,
        debug: Optional[str] = None,
        coords_path: Optional[str | Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single image: detect blood and obscure it.

        Args:
            image_path: Path to the input image.
            output_path: Where to save the result. If None, saves alongside input
                         with '_clean' suffix.
            method: 'inpaint' (natural) or 'blur' (fast fallback).
            save_mask: If True, also saves the detection mask.
            debug: Experiment mode for inspecting detection:
                   'red'   - paint every detected pixel bright red on a
                             grayscale copy (visualize what was found)
                   'black' - paint every detected pixel black on the original
                             (simple covering experiment)
            coords_path: If set, saves all detected pixel coordinates as
                         (x, y) pairs to a JSON file.

        Returns:
            Tuple of (processed_image, mask).
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        mask = self.build_mask(img)

        if debug == "red":
            # Grayscale copy so the red overlay pops clearly
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result[mask > 0] = (0, 0, 255)  # BGR: pure red
        elif debug == "black":
            result = img.copy()
            result[mask > 0] = (0, 0, 0)  # BGR: black
        elif method == "inpaint":
            result = self._inpaint(img, mask)
        elif method == "pixelate":
            result = self._pixelate(img, mask)
        else:
            result = self._blur(img, mask)

        # Export detected pixel coordinates (x, y) as JSON
        if coords_path is not None:
            ys, xs = np.where(mask > 0)
            coords = [{"x": int(x), "y": int(y)} for x, y in zip(xs, ys)]
            coords_path = Path(coords_path)
            coords_path.parent.mkdir(parents=True, exist_ok=True)
            with open(coords_path, "w") as f:
                json.dump({"count": len(coords), "pixels": coords}, f, indent=1)
            print(f"Detected {len(coords)} blood pixels -> {coords_path}")

        # Determine output path
        if output_path is None:
            p = Path(image_path)
            suffix = "_red" if debug == "red" else "_black" if debug == "black" else "_clean"
            output_path = p.parent / f"{p.stem}{suffix}{p.suffix}"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), result)

        if save_mask:
            mask_path = output_path.parent / f"{output_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)

        return result, mask

    def process_batch(
        self,
        input_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        method: str = "inpaint",
        save_mask: bool = False,
    ) -> list[Tuple[Path, bool]]:
        """Process all images in a directory.

        Returns:
            List of (file_path, success) tuples.
        """
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / "output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        results = []
        for img_path in sorted(images):
            try:
                out_path = output_dir / f"{img_path.stem}_clean{img_path.suffix}"
                self.process_image(img_path, out_path, method=method, save_mask=save_mask)
                results.append((img_path, True))
                print(f"  [OK] {img_path.name} -> {out_path.name}")
            except Exception as e:
                results.append((img_path, False))
                print(f"  [FAIL] {img_path.name}: {e}")

        return results
