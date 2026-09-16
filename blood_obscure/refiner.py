"""MobileSAM-based mask refinement for blood detection.

The color-threshold detector finds candidate blood regions but cannot
separate dark/clotted blood from skin tones (they overlap in color
space). MobileSAM (ViT-Tiny, 9.66M params) refines each candidate box
into a precise segmentation mask, catching dark blood edges that color
thresholds miss while excluding skin.

Runs fully on CPU via onnxruntime — no torch. Two ONNX models:
- mobile_sam_image_encoder.onnx  (28 MB, image -> embedding)
- sam_mask_decoder_single.onnx   (16 MB, embedding + box -> mask)

Both from https://huggingface.co/Acly/MobileSAM (MIT license).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ENCODER_PATH = MODELS_DIR / "mobile_sam_image_encoder.onnx"
DECODER_PATH = MODELS_DIR / "sam_mask_decoder_single.onnx"

LONG_SIDE = 1024


def _get_preprocess_shape(oldh: int, oldw: int, long_side_length: int = LONG_SIDE) -> Tuple[int, int]:
    """SAM ResizeLongestSide: scale so the longest side == long_side_length."""
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    return int(newh + 0.5), int(neww + 0.5)


def _apply_coords(coords: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    """Scale coordinates from original-image space to 1024-space."""
    old_h, old_w = original_size
    new_h, new_w = _get_preprocess_shape(old_h, old_w)
    coords = coords.astype(float).copy()
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


class MobileSAMRefiner:
    """Refines a coarse blood mask using MobileSAM (ONNX, CPU, no torch).

    The coarse color-threshold mask provides candidate boxes; MobileSAM
    segments each box precisely, catching dark blood edges that color
    thresholds miss and excluding skin that color wrongly includes.
    """

    def __init__(
        self,
        encoder_path: Optional[str | Path] = None,
        decoder_path: Optional[str | Path] = None,
    ):
        self.encoder_path = Path(encoder_path) if encoder_path else ENCODER_PATH
        self.decoder_path = Path(decoder_path) if decoder_path else DECODER_PATH
        self._encoder = None
        self._decoder = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """True if both ONNX models exist on disk."""
        if self._available is None:
            self._available = self.encoder_path.exists() and self.decoder_path.exists()
        return self._available

    def _load(self) -> None:
        if self._encoder is None:
            import onnxruntime

            self._encoder = onnxruntime.InferenceSession(
                str(self.encoder_path), providers=["CPUExecutionProvider"]
            )
            self._decoder = onnxruntime.InferenceSession(
                str(self.decoder_path), providers=["CPUExecutionProvider"]
            )

    def _embed(self, bgr: np.ndarray) -> np.ndarray:
        """Image -> image embedding (1, 256, 64, 64).

        The ONNX graph bakes in normalization (mean/std) and padding to
        1024x1024, but NOT resizing — we resize longest side to 1024.
        Input must be 0-255 HWC float32.
        """
        h, w = bgr.shape[:2]
        scale = LONG_SIDE * 1.0 / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        resized = cv2.resize(bgr, (new_w, new_h))
        return self._encoder.run(None, {"input_image": resized.astype(np.float32)})[0]

    def _decode(self, embedding: np.ndarray, box: list, orig_size: Tuple[int, int]) -> np.ndarray:
        """Box prompt (XYXY, original pixels) -> binary mask at original resolution."""
        h, w = orig_size
        onnx_coord = np.array(box, dtype=np.float32).reshape(2, 2)[None, :, :]
        onnx_label = np.array([2, 3], dtype=np.float32)[None, :]  # box TL, BR
        onnx_coord = _apply_coords(onnx_coord, (h, w)).astype(np.float32)

        ort_inputs = {
            "image_embeddings": embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array([h, w], dtype=np.float32),
        }
        masks, _, _ = self._decoder.run(None, ort_inputs)
        return masks[0, 0] > 0.0  # SAM mask threshold is 0.0, not 0.5

    def refine(
        self,
        bgr: np.ndarray,
        coarse_mask: np.ndarray,
        min_box_area: int = 100,
        pad: int = 8,
    ) -> np.ndarray:
        """Refine a coarse mask with MobileSAM.

        Each connected component of the coarse mask becomes a box prompt;
        SAM's segmentation of that box is added to the result. Falls back
        to the coarse mask unchanged if the ONNX models are unavailable.

        Args:
            bgr: Original BGR image.
            coarse_mask: Binary mask from the color-threshold detector.
            min_box_area: Ignore components smaller than this (noise).
            pad: Expand each box by this many pixels (SAM works better
                 with some context around the target).

        Returns:
            Refined binary mask (uint8, 0/255), same size as input.
        """
        if not self.available:
            return coarse_mask
        self._load()

        h, w = bgr.shape[:2]
        embedding = self._embed(bgr)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            coarse_mask, connectivity=8
        )
        refined = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, num_labels):  # skip background
            x, y, bw, bh, area = stats[i]
            if area < min_box_area:
                continue
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            sam_mask = self._decode(embedding, [x1, y1, x2, y2], (h, w))
            refined[sam_mask] = 255

        return refined