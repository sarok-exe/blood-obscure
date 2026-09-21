"""Core blood detection and obscuring engine."""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .refiner import MobileSAMRefiner


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
        max_area_ratio: float = 1.0,
        a_channel_threshold: int = 140,
        dilate_pixels: int = 0,
        blur_passes: int = 2,
        block_size: int = 100,
        colors_path: Optional[str | Path] = None,
        use_sam: bool = False,
    ):
        self.blur_strength = blur_strength
        self.inpaint_radius = inpaint_radius
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.a_channel_threshold = a_channel_threshold
        self.dilate_pixels = dilate_pixels
        self.blur_passes = blur_passes
        self.block_size = block_size
        self.colors_path = colors_path or (
            Path(__file__).resolve().parent.parent / "blood_colors.json"
        )
        self.layers = self._load_color_layers()
        self.use_sam = use_sam
        self.refiner = MobileSAMRefiner()
        self.classifier = self._load_classifier()

    def _load_color_layers(self) -> list[dict]:
        """Load red color layers from the color file.

        Each layer: {"name", "hsv_min", "hsv_max", "min_a", "min_redness"}.
        Falls back to the original hardcoded layers if the file is missing.
        """
        try:
            with open(self.colors_path) as f:
                data = json.load(f)
            layers = data["layers"]
            if not layers:
                raise ValueError("color file has no layers")
            return layers
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            return [
                {
                    "name": "fresh_bright_red",
                    "hsv_min": [0, 60, 50],
                    "hsv_max": [10, 255, 255],
                    "min_a": self.a_channel_threshold,
                    "min_redness": 40,
                },
                {
                    "name": "fresh_bright_red_wrapped",
                    "hsv_min": [160, 60, 50],
                    "hsv_max": [180, 255, 255],
                    "min_a": self.a_channel_threshold,
                    "min_redness": 40,
                },
                {
                    "name": "dried_red",
                    "hsv_min": [8, 100, 40],
                    "hsv_max": [15, 255, 255],
                    "min_a": self.a_channel_threshold,
                    "min_redness": 40,
                },
            ]

    def _load_classifier(self):
        """Load the trained red classifier (DecisionTree) if available.

        The classifier is the skin-safe seed for the ML detection path.
        Returns None if the model file is missing — the detector then
        falls back to the color-layer path.
        """
        try:
            import joblib

            path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "red_classifier.joblib"
            )
            if path.exists():
                return joblib.load(str(path))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    #  Mask pipeline
    # ------------------------------------------------------------------ #

    def _face_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Face detection via OpenCV Haar cascade — dilated face boxes.

        Faces are the most common skin false-positive. Returns an empty
        mask if the cascade is unavailable or no faces are found.

        Each detected box is kept only if it actually contains skin-tone
        pixels (bright, moderately saturated, reddish hue). The cascade
        fires false positives on anatomy photos (tissue folds, dark
        regions); those boxes contain no skin and must NOT be excluded —
        they can cover real blood.
        """
        h, w = bgr.shape[:2]
        face_mask = np.zeros((h, w), dtype=np.uint8)
        try:
            cascade_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "haarcascade_frontalface_default.xml"
            )
            if not cascade_path.exists():
                return face_mask
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(str(cascade_path))
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            skin = (V > 130) & (S > 60) & (H >= 5) & (H <= 25)
            for (x, y, fw, fh) in faces:
                x1 = max(0, x - int(0.1 * fw))
                y1 = max(0, y - int(0.15 * fh))
                x2 = min(w, x + fw + int(0.1 * fw))
                y2 = min(h, y + fh + int(0.2 * fh))
                box = skin[y1:y2, x1:x2]
                if box.sum() < 0.01 * box.size:
                    continue  # no skin in box — cascade false positive
                cv2.rectangle(face_mask, (x1, y1), (x2, y2), 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            face_mask = cv2.dilate(face_mask, kernel)
        except Exception:
            pass  # cascade missing or failed — no face exclusion
        return face_mask

    def build_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Build a clean binary mask of blood regions.

        Two detection paths:

        ML path (default, when models/red_classifier.joblib exists):
        - Seed = trained DecisionTree classifier (near-zero skin FP).
        - Region growing into dark_clotted candidates (dark blood edges)
          PLUS a bright layer gated by a per-image saturation floor
          estimated from the image's own skin tones. The floor sits above
          ~95% of the image's skin saturation, so bright/lighted skin is
          never covered while medium-bright blood is recovered.

        Layer path (fallback, no classifier file):
        - Every pixel is matched against the red color layers in the
          color file (blood_colors.json). A pixel is blood if it falls in
          ANY layer's HSV range AND passes that layer's LAB A-channel and
          R-G redness thresholds. Layers marked "grow_only" are accepted
          only when adjacent to already-detected blood (region growing).

        Then: face exclusion, morphological cleanup (layer path only —
        the ML path is already skin-tight and morphology re-adds skin),
        component filter.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        b, g, r = cv2.split(bgr)
        redness = r.astype(int) - np.maximum(g, b).astype(int)

        if self.classifier is not None:
            mask = self._build_mask_ml(bgr, hsv, lab, redness)
        else:
            mask = self._build_mask_layers(hsv, lab, redness)

        # 2. Exclude faces — skin false positives. Blood is never on a face.
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(self._face_mask(bgr)))

        # 3. Morphological cleanup — CLOSE then OPEN with elliptical kernel.
        #    Layer path only: the ML path's mask is already skin-tight and
        #    CLOSE/OPEN expands it back into adjacent skin.
        if self.classifier is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Connected-component filter — drop tiny noise AND giant tissue blobs.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = int(bgr.shape[0] * bgr.shape[1] * self.min_area_ratio)
        max_area = int(bgr.shape[0] * bgr.shape[1] * self.max_area_ratio)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):  # skip background
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
                clean[labels == i] = 255

        # 5. MobileSAM refinement — precise segmentation of each candidate
        #    box. Catches dark blood edges color thresholds miss, excludes
        #    skin that color wrongly includes. Falls back to the coarse
        #    mask if the ONNX models are unavailable.
        if self.use_sam and self.refiner.available:
            clean = self.refiner.refine(bgr, clean)
            # SAM could theoretically re-add face skin near a blood box —
            # re-apply face exclusion to the refined mask.
            clean = cv2.bitwise_and(clean, cv2.bitwise_not(self._face_mask(bgr)))

        return clean

    def _build_mask_layers(
        self, hsv: np.ndarray, lab: np.ndarray, redness: np.ndarray
    ) -> np.ndarray:
        """Color-layer seed + region growing (fallback path)."""
        # 1. Match every pixel against every red layer in the color file
        seed = np.zeros(hsv.shape[:2], dtype=np.uint8)
        grow_candidates = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for layer in self.layers:
            lo = np.array(layer["hsv_min"], dtype=np.uint8)
            hi = np.array(layer["hsv_max"], dtype=np.uint8)
            min_a = int(layer.get("min_a", self.a_channel_threshold))
            min_red = int(layer.get("min_redness", 40))

            layer_mask = cv2.inRange(hsv, lo, hi)
            layer_mask = cv2.bitwise_and(
                layer_mask, (lab[:, :, 1] > min_a).astype(np.uint8) * 255
            )
            layer_mask = cv2.bitwise_and(
                layer_mask, (redness > min_red).astype(np.uint8) * 255
            )
            if "max_b" in layer:
                layer_mask = cv2.bitwise_and(
                    layer_mask,
                    (lab[:, :, 2] < int(layer["max_b"])).astype(np.uint8) * 255,
                )
            if layer.get("grow_only"):
                grow_candidates = cv2.bitwise_or(grow_candidates, layer_mask)
            else:
                seed = cv2.bitwise_or(seed, layer_mask)

        # 1b. Region growing: accept grow-only pixels adjacent to seed.
        #     Iteratively dilate the seed and keep only candidate pixels
        #     that touch the growing region — dark blood edges get caught,
        #     isolated skin never does.
        mask = seed.copy()
        kernel3 = np.ones((3, 3), np.uint8)
        for _ in range(80):
            new = cv2.dilate(mask, kernel3) & grow_candidates & ~mask
            if new.sum() == 0:
                break
            mask |= new
        return mask

    def _build_mask_ml(
        self, bgr: np.ndarray, hsv: np.ndarray, lab: np.ndarray, redness: np.ndarray
    ) -> np.ndarray:
        """ML-tree seed + adaptive dual-layer region growing.

        Seed: the trained DecisionTree classifier — near-zero skin false
        positives. Grow candidates: the existing dark_clotted layer (dark
        blood edges) PLUS a bright layer gated by a per-image saturation
        floor estimated from the image's own skin tones. The floor sits
        above ~95% of the image's skin saturation, so bright/lighted skin
        is never covered while medium-bright blood is recovered.
        """
        b, g, r = cv2.split(bgr)
        red = r.astype(np.float32) - np.maximum(g, b).astype(np.float32)
        X = np.column_stack(
            [
                hsv[:, :, 0].ravel().astype(np.float32),
                hsv[:, :, 1].ravel().astype(np.float32),
                hsv[:, :, 2].ravel().astype(np.float32),
                lab[:, :, 0].ravel().astype(np.float32),
                lab[:, :, 1].ravel().astype(np.float32),
                lab[:, :, 2].ravel().astype(np.float32),
                red.ravel(),
            ]
        )
        tree_mask = self.classifier.predict(X).reshape(bgr.shape[:2]) > 0
        seed = tree_mask.astype(np.uint8) * 255

        # Per-image saturation floor: skin = red-hue, bright, not blood.
        # Floor sits above ~95% of the image's own skin saturation.
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red_hue = ((H >= 0) & (H <= 20)) | ((H >= 160) & (H <= 180))
        skin_est = red_hue & (V > 130) & ~tree_mask
        if skin_est.sum() > 50:
            floor = int(np.percentile(S[skin_est], 95)) + 15
        else:
            floor = 150
        floor = max(floor, 140)

        # Grow candidates: dark_clotted (existing grow_only layers) + bright layer
        grow_candidates = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for layer in self.layers:
            if not layer.get("grow_only"):
                continue
            lo = np.array(layer["hsv_min"], dtype=np.uint8)
            hi = np.array(layer["hsv_max"], dtype=np.uint8)
            min_a = int(layer.get("min_a", self.a_channel_threshold))
            min_red = int(layer.get("min_redness", 40))
            layer_mask = cv2.inRange(hsv, lo, hi)
            layer_mask = cv2.bitwise_and(
                layer_mask, (lab[:, :, 1] > min_a).astype(np.uint8) * 255
            )
            layer_mask = cv2.bitwise_and(
                layer_mask, (redness > min_red).astype(np.uint8) * 255
            )
            if "max_b" in layer:
                layer_mask = cv2.bitwise_and(
                    layer_mask,
                    (lab[:, :, 2] < int(layer["max_b"])).astype(np.uint8) * 255,
                )
            grow_candidates = cv2.bitwise_or(grow_candidates, layer_mask)

        # Bright layer: medium-bright blood the tree misses (V 115-146),
        # gated by the per-image saturation floor to keep skin out.
        bright = (
            red_hue
            & (V > 115)
            & (S >= floor)
            & (lab[:, :, 1] > 128)
            & (lab[:, :, 2] < 152)
        ).astype(np.uint8) * 255
        grow_candidates = cv2.bitwise_or(grow_candidates, bright)

        # Region growing: accept grow candidates adjacent to seed.
        mask = seed.copy()
        kernel3 = np.ones((3, 3), np.uint8)
        for _ in range(80):
            new = cv2.dilate(mask, kernel3) & grow_candidates & ~mask
            if new.sum() == 0:
                break
            mask |= new
        return mask

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

        The kernel is adaptive: sized to 2x the largest blood blob so the
        red signal is diluted into surrounding tissue even for large blood
        regions. Only the blood pixels are replaced — nothing outside the
        mask is touched.
        """
        if self.dilate_pixels > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_pixels)

        # Adaptive kernel: 2x the largest blood blob dimension, so even the
        # center of a big blood region samples surrounding tissue.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels > 1:
            max_dim = max(
                max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                for i in range(1, num_labels)
            )
            k = int(max_dim * 2) | 1  # odd kernel size
            k = min(k, 999)
        else:
            k = self.blur_strength

        blurred = bgr.copy()
        for _ in range(max(1, self.blur_passes)):
            blurred = cv2.GaussianBlur(blurred, (k, k), 0)
        return np.where(mask[..., None] > 0, blurred, bgr)

    def _pixelate(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Obscure blood via square blur (mosaic/pixelation).

        The image is divided into a grid of squares, each block_size x
        block_size pixels. Every square is filled with the average color
        of its pixels, producing large gathered squares. Blocks that
        touch the (dilated) blood mask are replaced — the mosaic is
        allowed to spill outside the blood region.
        """
        if self.dilate_pixels > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_pixels)

        h, w = bgr.shape[:2]
        bs = max(1, self.block_size)

        # Downscale then upscale with nearest neighbor = blocky squares
        small_h, small_w = max(1, h // bs), max(1, w // bs)
        small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return np.where(mask[..., None] > 0, pixelated, bgr)

    def _white(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Obscure blood by painting every detected pixel pure white.

        Only the masked pixels are changed — everything outside the mask
        is untouched. This is the strongest possible covering: the blood
        color is fully replaced, not diluted.
        """
        if self.dilate_pixels > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_pixels)

        result = bgr.copy()
        result[mask > 0] = (255, 255, 255)  # BGR: white
        return result

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
        elif method == "white":
            result = self._white(img, mask)
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

    # ------------------------------------------------------------------ #
    #  Two-stage pipeline: detect → JSON → edit
    # ------------------------------------------------------------------ #

    def detect_only(
        self,
        image_path: str | Path,
        coords_path: str | Path,
    ) -> np.ndarray:
        """Stage 1 — detect blood and write every pixel's (x, y) to JSON.

        The image is NOT modified. This isolates the detection model from
        editing so that missed blood or false positives can be diagnosed
        independently.

        JSON schema::

            {
              "image": "photo.jpg",
              "width": 1920,
              "height": 1080,
              "count": 12345,
              "pixels": [{"x": 1, "y": 2}, ...]
            }

        Returns the detection mask.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        mask = self.build_mask(img)
        ys, xs = np.where(mask > 0)
        coords = [{"x": int(x), "y": int(y)} for x, y in zip(xs, ys)]

        coords_path = Path(coords_path)
        coords_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image": str(image_path),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "count": len(coords),
            "pixels": coords,
        }
        with open(coords_path, "w") as f:
            json.dump(data, f, indent=1)
        print(f"Stage 1: {len(coords)} blood pixels detected -> {coords_path}")
        return mask

    def edit_from_coords(
        self,
        image_path: str | Path,
        coords_path: str | Path,
        output_path: Optional[str | Path] = None,
        method: str = "white",
    ) -> np.ndarray:
        """Stage 2 — read coordinates JSON and paint those pixels.

        No detection happens here. This is a pure editing pass: load the
        image, read every (x, y) from the JSON, and cover them with the
        chosen method (default: white fill).

        If the output still shows blood, the problem is in the JSON
        (i.e. the detection stage missed it), not in this editor.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        with open(coords_path) as f:
            data = json.load(f)

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        valid = 0
        for p in data["pixels"]:
            x, y = int(p["x"]), int(p["y"])
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = 255
                valid += 1

        if method == "white":
            result = self._white(img, mask)
        elif method == "pixelate":
            result = self._pixelate(img, mask)
        elif method == "inpaint":
            result = self._inpaint(img, mask)
        else:
            result = self._blur(img, mask)

        if output_path is None:
            p = Path(image_path)
            output_path = p.parent / f"{p.stem}_clean{p.suffix}"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        print(f"Stage 2: painted {valid} pixels ({method}) -> {output_path}")
        return result

    # ------------------------------------------------------------------ #
    #  Batch processing
    # ------------------------------------------------------------------ #

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
