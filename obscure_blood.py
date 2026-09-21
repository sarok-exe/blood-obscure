#!/usr/bin/env python3
"""Config-driven blood obscuring for anatomy school images.

Pure OpenCV color segmentation -- no ML, no training. Detection settings
live in a JSON config (see obscure_config.json). Supports three obscuring
methods: white fill, adaptive Gaussian blur, and block pixelation.

Usage:
    python obscure_blood.py --input DIR [--output DIR] [--config FILE]
                            [--method white|blur|pixelate] [--eval-masks DIR]
                            [--model MODEL.onnx] [--model-only]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# --------------------------------------------------------------------------- #
#  Loading / saving
# --------------------------------------------------------------------------- #

def load_image_bgr(path):
    """Load an image (any PIL-supported format incl. webp) as a BGR array."""
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_gt_mask(path, shape):
    """Load a binary ground-truth mask (white=blood) and match image size."""
    arr = np.asarray(Image.open(path).convert("L"))
    if arr.shape != tuple(shape):
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 127).astype(np.uint8) * 255


def save_image_bgr(bgr, path):
    """Save a BGR array via PIL (handles webp output)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


# --------------------------------------------------------------------------- #
#  Detection
# --------------------------------------------------------------------------- #

def detect_blood(bgr, config):
    """Detect blood pixels -> uint8 mask (0/255).

    Pipeline: HSV/LAB/redness layer matching -> region growing from
    grow_only candidates -> morphology -> connected-component filter.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    redness = r.astype(int) - np.maximum(g, b).astype(int)
    # R / max(G,B) ratio, floored at 1 to avoid div-by-zero. Near-black blood
    # has a high ratio (reddish); dark shadows have a low ratio (neutral).
    ratio = r.astype(np.float32) / np.maximum(np.maximum(g, b), 1).astype(np.float32)

    # Adaptive saturation floor: estimate the image's own skin saturation and
    # raise the seed layers' S minimum above it, so bright skin is never
    # covered while medium-bright blood is still recovered.
    seed_s_floor = None
    if config.get("adaptive_saturation", False):
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red_hue = (H <= 20) | (H >= 160)
        skin_est = red_hue & (V > 130) & (S < 150)  # bright red-hue, moderate sat
        if skin_est.sum() > 50:
            seed_s_floor = int(np.percentile(S[skin_est], 95)) + 15
        else:
            seed_s_floor = 40
        seed_s_floor = max(seed_s_floor, 40)

    seed = np.zeros(bgr.shape[:2], dtype=np.uint8)
    candidates = np.zeros(bgr.shape[:2], dtype=np.uint8)

    for layer in config["layers"]:
        lo = np.array(layer["hsv_min"], dtype=np.uint8)
        hi = np.array(layer["hsv_max"], dtype=np.uint8)
        min_a = int(layer.get("min_a", 140))
        min_red = int(layer.get("min_redness", 40))

        # Seed layers use the adaptive S floor (raised, never lowered);
        # grow-only layers keep their configured S floors.
        if seed_s_floor is not None and not layer.get("grow_only"):
            lo = lo.copy()
            lo[1] = max(int(lo[1]), seed_s_floor)

        m = cv2.inRange(hsv, lo, hi)
        m = cv2.bitwise_and(m, (lab[:, :, 1] > min_a).astype(np.uint8) * 255)
        m = cv2.bitwise_and(m, (redness > min_red).astype(np.uint8) * 255)
        if "min_redness_ratio" in layer:
            m = cv2.bitwise_and(m, (ratio > float(layer["min_redness_ratio"])).astype(np.uint8) * 255)
        if "max_b" in layer:
            m = cv2.bitwise_and(m, (lab[:, :, 2] < int(layer["max_b"])).astype(np.uint8) * 255)

        if layer.get("grow_only"):
            candidates = cv2.bitwise_or(candidates, m)
        else:
            seed = cv2.bitwise_or(seed, m)

    # Region growing: accept grow_only pixels adjacent to the seed.
    mask = seed.copy()
    grow_iter = int(config.get("grow_iterations", 80))
    kernel3 = np.ones((3, 3), np.uint8)
    for _ in range(grow_iter):
        new = cv2.dilate(mask, kernel3) & candidates & ~mask
        if new.sum() == 0:
            break
        mask |= new

    # Morphology cleanup (0 = skip).
    morph = config.get("morphology", {})
    open_k = int(morph.get("open_kernel", 0))
    close_k = int(morph.get("close_kernel", 0))
    dilate_k = int(morph.get("dilate", 0))
    if open_k > 0 and open_k % 2 == 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 0 and close_k % 2 == 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    if dilate_k > 0:
        mask = cv2.dilate(mask, np.ones((dilate_k, dilate_k), np.uint8))

    # Connected-component filter: drop skin speckles below min_area.
    min_area = int(config.get("min_area", 200))
    if min_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        mask = clean

    return mask


# --------------------------------------------------------------------------- #
#  Neural-network model (optional ONNX)
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model_session(model_path):
    """Load an ONNX blood model -> (session, input_h, input_w) or (None, None, None)."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape  # ['batch', 3, H, W]
        h, w = int(shape[2]), int(shape[3])
        print(f"[model] loaded {model_path} (input {w}x{h})")
        return sess, h, w
    except Exception as e:
        print(f"[warn] failed to load model {model_path}: {e}")
        return None, None, None


def model_mask(session, h, w, bgr):
    """Run the ONNX model on a BGR image -> uint8 0/255 mask at original size."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    x = arr.transpose(2, 0, 1)[None].astype(np.float32)
    logits = session.run(None, {"input": x})[0][0, 0]
    prob = 1.0 / (1.0 + np.exp(-logits))
    m = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


def combined_mask(bgr, config, session=None, model_h=None, model_w=None, model_only=False):
    """Color mask, optionally unioned with the NN model mask (or model alone)."""
    color = detect_blood(bgr, config)
    if session is None:
        return color
    try:
        mm = model_mask(session, model_h, model_w, bgr)
    except Exception as e:
        print(f"[warn] model inference failed: {e}; using color-only mask")
        return color
    if model_only:
        return mm
    return cv2.bitwise_or(color, mm)


# --------------------------------------------------------------------------- #
#  Obscuring
# --------------------------------------------------------------------------- #

def _feather_mask(mask, sigma):
    """Blur the mask into a soft 0..1 alpha so an effect fades at its edges.

    Computed at reduced resolution (the result is a smooth gradient anyway),
    which keeps it cheap for per-frame video use.
    """
    h, w = mask.shape[:2]
    s = 4
    small = cv2.resize(mask, (max(1, w // s), max(1, h // s)),
                       interpolation=cv2.INTER_AREA)
    soft = cv2.GaussianBlur(small, (0, 0), sigmaX=max(0.1, float(sigma) / s))
    soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
    return (soft.astype(np.float32) / 255.0)[..., None]


def _blend(base, fill, alpha):
    """alpha*base + (1-alpha)*fill, back to uint8."""
    out = base.astype(np.float32)
    out *= (1.0 - alpha)
    out += fill.astype(np.float32) * alpha
    return out.astype(np.uint8)


def _fog_fill(bgr, mask, scale, dilate=0, smooth=0.0):
    """Fill the masked region with colours sampled from around it.

    The mask is dilated before inpainting so the fill is pulled from *outside*
    the red blood boundary - otherwise the boundary's own red colour is
    diffused straight back in. Works at reduced resolution for speed (the
    smoothing is applied there too), then is upscaled into a soft fog.
    """
    h, w = bgr.shape[:2]
    s = max(2, int(scale))
    sw, sh = max(1, w // s), max(1, h // s)
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    ds = max(0, int(dilate) // s) if dilate > 0 else 0
    if ds > 0:
        m = cv2.dilate(m, np.ones((2 * ds + 1, 2 * ds + 1), np.uint8))
    mb = m > 0
    if mb.any():
        filled = cv2.inpaint(small, m, 3, cv2.INPAINT_TELEA)
        # Colour-match the fill to the ring just outside the region so the fog
        # sits at the same brightness/tint as its surroundings (avoids the
        # brighter-than-surroundings patch that inpainting alone leaves).
        ring = (cv2.dilate(m, np.ones((2 * ds + 3, 2 * ds + 3), np.uint8)) > 0) & ~mb
        if ring.any():
            shift = small[ring].mean(0) - filled[mb].mean(0)
            filled = np.clip(filled.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        small = filled
    if smooth > 0:
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(0.1, float(smooth) / s))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _coverage_mask(bgr, mask, config):
    """Union the detection mask with a raw redness mask.

    ``detect_blood`` deliberately drops small components and applies morphology
    so it never covers skin, but that also leaves patches of genuinely red
    (blood) pixels outside the mask. For coverage we add those back: strongly
    red, saturated pixels are unioned in, lightly cleaned so single-pixel noise
    cannot seed a fog blob, then merged with the detection mask.
    """
    fcfg = config.get("fog", {})
    red = int(fcfg.get("safety_redness", 60))
    sat = int(fcfg.get("safety_saturation", 40))
    if red <= 0 or not (mask > 0).any():
        return mask
    b, g, r = cv2.split(bgr.astype(int))
    d = r - np.maximum(g, b)
    s = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    safety = ((d > red) & (s > sat)).astype(np.uint8) * 255
    safety = cv2.morphologyEx(safety, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    combined = cv2.bitwise_or(mask, safety)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _expand_mask(mask, factor, min_px=1, downscale=4):
    """Grow each blob outward by ``factor`` times its own radius.

    A blob's local thickness (distance to the nearest background pixel) gives
    its radius, so expanding every blob by its own radius doubles its diameter
    - small specks grow a little, large pools grow a lot. A minimum expansion
    (``min_px``) keeps tiny detections covered. Computed at reduced resolution
    because the result is a smooth region anyway.
    """
    mb = (mask > 0).astype(np.uint8)
    if factor <= 0 or not mb.any():
        return (mb * 255).astype(np.uint8)
    h, w = mb.shape[:2]
    s = max(1, int(downscale))
    small = cv2.resize(mb, (max(1, w // s), max(1, h // s)),
                       interpolation=cv2.INTER_NEAREST) if s > 1 else mb
    dist_in = cv2.distanceTransform(small, cv2.DIST_L2, 3)
    max_r = float(dist_in.max())
    k = int(2 * max_r + 1) | 1
    # Propagate each blob's radius outwards so an outside pixel knows the
    # radius of the blob nearest to it.
    radius_near = cv2.dilate(dist_in, np.ones((k, k), np.float32)) if k > 1 else dist_in
    dist_out = cv2.distanceTransform(1 - small, cv2.DIST_L2, 3)
    grow = np.maximum(factor * radius_near, float(min_px) / s)
    expanded = ((dist_out <= grow) | (small > 0)).astype(np.uint8) * 255
    if s > 1:
        expanded = cv2.resize(expanded, (w, h), interpolation=cv2.INTER_LINEAR)
        expanded = (expanded >= 127).astype(np.uint8) * 255
    return expanded


def obscure(bgr, mask, method, config):
    """Apply an obscuring method inside the mask. Returns a new BGR image."""
    if method == "white":
        result = bgr.copy()
        result[mask > 0] = (255, 255, 255)
        return result

    if method == "fog":
        # Two-pass soft fog that blends with the surrounding colours:
        #   pass 1 - replace the blood with colours pulled from just outside it
        #   pass 2 - extend the covered area outwards and fog again, wider/softer
        fcfg = config.get("fog", {})
        base = int(fcfg.get("base", config.get("pixelate_block", 30)))
        extend = int(fcfg.get("extend", max(1, base // 2)))
        feather = float(fcfg.get("feather", 0.5))
        expand = float(fcfg.get("expand", 1.0))

        # Cover all detected + strongly-red pixels, then grow every blob by
        # `expand` x its own radius so a 50px blob is covered by ~2x that.
        base_mask = _coverage_mask(bgr, mask, config)
        cover = _expand_mask(base_mask, expand, min_px=max(1, base // 2))

        fog1 = _fog_fill(bgr, cover, scale=max(4, base // 4),
                         dilate=max(1, base // 4), smooth=base * 0.3)
        out = _blend(bgr, fog1, _feather_mask(cover, sigma=max(1.0, base * feather)))

        wide = cv2.dilate(cover, np.ones((max(1, extend),) * 2, np.uint8))
        fog2 = _fog_fill(out, cover, scale=max(4, base // 3),
                         dilate=max(1, base // 3), smooth=base * 0.5)
        return _blend(out, fog2, _feather_mask(wide, sigma=max(1.0, base * feather * 1.5)))

    if method == "blur":
        # Adaptive kernel: 2x the largest blood blob dimension (odd, min 5).
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            max_dim = max(
                max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                for i in range(1, num_labels)
            )
            k = max(5, (max_dim * 2) | 1)
        else:
            k = 15
        blurred = cv2.GaussianBlur(bgr, (k, k), 0)
        if config.get("feather", False):
            # Soft edge: blend with a 2px-Gaussian-blurred mask.
            soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0).astype(np.float32) / 255.0
            soft = soft[..., None]
            return (soft * blurred + (1.0 - soft) * bgr).astype(np.uint8)
        result = bgr.copy()
        result[mask > 0] = blurred[mask > 0]
        return result

    if method == "pixelate":
        h, w = bgr.shape[:2]
        bs = max(1, int(config.get("pixelate_block", 12)))
        small = cv2.resize(bgr, (max(1, w // bs), max(1, h // bs)), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result = bgr.copy()
        result[mask > 0] = pixelated[mask > 0]
        return result

    raise ValueError(f"Unknown obscuring method: {method!r}")


# --------------------------------------------------------------------------- #
#  Anti-detection post-processing
# --------------------------------------------------------------------------- #

def apply_anti_detect(img, config, target_size=None):
    """Global post-processing that breaks reverse-image-search matching while
    keeping the image recognizable: light grain, slight resize, optional flip.

    Reads the "anti_detect" section of the config:
      noise_sigma      Gaussian noise std-dev on 0-255 (0 = off)
      resize_factor    output scale (1.0 = off, <1 shrinks, >1 enlarges)
      flip_horizontal  mirror the image (default off)

    target_size overrides resize_factor (used by the video pipeline so every
    frame lands on the exact same even dimensions).
    """
    ad = config.get("anti_detect", {})
    out = img
    sigma = float(ad.get("noise_sigma", 0.0))
    if sigma > 0:
        noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if target_size is not None:
        out = cv2.resize(out, target_size, interpolation=cv2.INTER_AREA)
    else:
        factor = float(ad.get("resize_factor", 1.0))
        if 0.0 < factor != 1.0:
            h, w = out.shape[:2]
            out = cv2.resize(out, (max(1, int(round(w * factor))),
                                   max(1, int(round(h * factor)))),
                             interpolation=cv2.INTER_AREA)
    if ad.get("flip_horizontal", False):
        out = cv2.flip(out, 1)
    return out


# --------------------------------------------------------------------------- #
#  Video processing (ffmpeg pipes)
# --------------------------------------------------------------------------- #

def probe_video(path):
    """Return (width, height, fps_rational, audio_codec) via ffprobe.

    ffprobe reports the *coded* dimensions, but ffmpeg auto-rotates frames on
    decode according to the display matrix. For a 90/270 degree rotation the
    decoded (display) dimensions are the coded ones swapped, so account for
    that here to match what the raw pipe actually emits.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = abs(int(sd["rotation"])) % 360
    if rotation in (90, 270):
        w, h = h, w
    fps = v.get("r_frame_rate", "25/1")  # keep the exact rational (e.g. 30000/1001)
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    return w, h, fps, (audio["codec_name"] if audio else None)


def process_video(input_path, output_path, config, method=None):
    """Process a video frame-by-frame with the blood obscuring pipeline.

    Decodes raw BGR frames through an ffmpeg pipe, runs detect_blood +
    obscure + apply_anti_detect on every frame, and re-encodes with libx264
    while keeping the original audio. The "anti_detect" config controls the
    output size / grain / flip; the output size is forced to even dimensions
    (required by libx264's yuv420p).

    The decode pipe scales every frame to the exact output size, so the frame
    geometry can never drift (rotation metadata, odd coded sizes, sample aspect
    ratios, ...) - otherwise the reshape below would scramble the image.
    """
    if method is None:
        method = config.get("obscure_method", "fog")
    w, h, fps, audio_codec = probe_video(input_path)
    factor = float(config.get("anti_detect", {}).get("resize_factor", 1.0))
    ow = max(2, int(round(w * factor)) // 2 * 2)
    oh = max(2, int(round(h * factor)) // 2 * 2)
    frame_bytes = ow * oh * 3
    temporal_alpha = float(config.get("fog", {}).get("temporal_alpha", 0.0))

    decode = ["ffmpeg", "-v", "error", "-i", str(input_path),
              "-vf", f"scale={ow}:{oh}",
              "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-"]
    enc_base = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{ow}x{oh}", "-r", fps, "-i", "-",
                "-i", str(input_path),
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-y", str(output_path)]

    def run(audio_args):
        dec = subprocess.Popen(decode, stdout=subprocess.PIPE, bufsize=10**7)
        enc = subprocess.Popen(enc_base + audio_args,
                               stdin=subprocess.PIPE, bufsize=10**7)
        prev = None  # temporal state (EMA of the detection mask)
        try:
            while True:
                raw = dec.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                bgr = np.frombuffer(raw, np.uint8).reshape(oh, ow, 3)
                mask = detect_blood(bgr, config)
                if temporal_alpha > 0:
                    # Smooth how the coverage grows/shrinks/moves between
                    # frames: an exponential moving average of the mask's
                    # signed distance field, then a zero threshold. Boundaries
                    # therefore drift continuously instead of popping pixel by
                    # pixel when a detection appears or disappears. Distances
                    # are clamped because cv2.distanceTransform returns ~3.4e38
                    # for an all-foreground input, which would otherwise poison
                    # the average for hundreds of frames.
                    mb = (mask > 0).astype(np.uint8)
                    if mb.any():
                        maxd = float(max(mb.shape))
                        phi = (np.minimum(cv2.distanceTransform(mb, cv2.DIST_L2, 3), maxd)
                               - np.minimum(cv2.distanceTransform(1 - mb, cv2.DIST_L2, 3), maxd))
                        prev = phi if prev is None else (
                            temporal_alpha * phi + (1.0 - temporal_alpha) * prev)
                    elif prev is not None:
                        prev = (1.0 - temporal_alpha) * prev  # fade coverage out
                    if prev is not None:
                        mask = (prev >= 0).astype(np.uint8) * 255
                detected = int((mask > 0).sum()) > 0
                result = obscure(bgr, mask, method, config) if detected else bgr
                result = apply_anti_detect(result, config, target_size=(ow, oh))
                enc.stdin.write(result.tobytes())
        finally:
            dec.stdout.close()
            enc.stdin.close()
            dec.wait()
            enc.wait()
        return enc.returncode

    # Copy the audio stream when possible; fall back to re-encoding it as AAC.
    rc = run(["-c:a", "copy"])
    if rc != 0:
        rc = run(["-c:a", "aac"])
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed (rc={rc})")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def dice_iou(pred, gt):
    """Dice coefficient and IoU between two binary masks."""
    p = pred > 0
    g = gt > 0
    inter = int((p & g).sum())
    if inter == 0:
        if int(p.sum()) == 0 and int(g.sum()) == 0:
            return 1.0, 1.0
        return 0.0, 0.0
    dice = 2.0 * inter / (int(p.sum()) + int(g.sum()))
    iou = inter / int((p | g).sum())
    return dice, iou


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def collect_images(input_dir):
    return sorted(
        f for f in Path(input_dir).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def run_eval(images, mask_dir, config, session=None, model_h=None, model_w=None, model_only=False):
    rows = []
    for img_path in images:
        gt_path = Path(mask_dir) / f"{img_path.stem}_mask.png"
        if not gt_path.exists():
            print(f"[skip] {img_path.name}: no ground-truth mask")
            continue
        bgr = load_image_bgr(img_path)
        pred = combined_mask(bgr, config, session, model_h, model_w, model_only)
        gt = load_gt_mask(gt_path, pred.shape)
        dice, iou = dice_iou(pred, gt)
        rows.append((img_path.name, dice, iou))

    if not rows:
        print("No images with ground-truth masks found.")
        return

    print(f"\n{'image':<48} {'dice':>7} {'iou':>7}")
    print("-" * 66)
    for name, dice, iou in rows:
        print(f"{name:<48} {dice:7.3f} {iou:7.3f}")

    dices = np.array([r[1] for r in rows])
    ious = np.array([r[2] for r in rows])
    print("-" * 66)
    print(f"{'median':<48} {np.median(dices):7.3f} {np.median(ious):7.3f}")
    print(f"{'mean':<48} {dices.mean():7.3f} {ious.mean():7.3f}")
    print(f"({len(rows)} images evaluated)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Config-driven blood obscuring (no ML).")
    ap.add_argument("--input", required=True, help="Directory of images to process")
    ap.add_argument("--output", default=None, help="Directory for obscured output (default: stats only)")
    ap.add_argument("--config", default=None, help="Path to config JSON (default: obscure_config.json next to script)")
    ap.add_argument("--method", default="pixelate", choices=["white", "blur", "pixelate"], help="Obscuring method")
    ap.add_argument("--eval-masks", default=None, metavar="DIR", help="Ground-truth mask dir -> evaluation mode")
    ap.add_argument("--model", default=None, metavar="MODEL.onnx", help="Optional ONNX blood model to union with color detection")
    ap.add_argument("--model-only", action="store_true", help="Use only the ONNX model mask (no color detection)")
    args = ap.parse_args(argv)

    config_path = Path(args.config) if args.config else Path(__file__).resolve().parent / "obscure_config.json"
    config = json.loads(config_path.read_text())

    session = model_h = model_w = None
    if args.model:
        session, model_h, model_w = load_model_session(args.model)

    images = collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return 1

    if args.eval_masks:
        run_eval(images, args.eval_masks, config, session, model_h, model_w, args.model_only)
        return 0

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        bgr = load_image_bgr(img_path)
        mask = combined_mask(bgr, config, session, model_h, model_w, args.model_only)
        n = int((mask > 0).sum())
        pct = 100.0 * n / mask.size
        print(f"{img_path.name}: {n} px ({pct:.2f}%)")
        if output_dir:
            result = obscure(bgr, mask, args.method, config)
            out_path = output_dir / f"{img_path.stem}_obscured{img_path.suffix}"
            save_image_bgr(result, out_path)
            print(f"  -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())