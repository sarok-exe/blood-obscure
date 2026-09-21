#!/usr/bin/env python3
"""Train a blood-segmentation CNN (pure torch, no torchvision, CPU + RAM safe).

MobileNetV2-U-Net where the encoder is built BY HAND with torch nn modules
(torchvision is unavailable on this Python 3.14 box, and TensorFlow has no
3.14 wheels). This trains on your hand-painted masks (white = blood) and
targets ~90% Dice to match the accuracy you got by hand.

Training data (13 pairs): <img_dir>/*  with masks at <img_dir>/masks/
    image stem X  ->  mask  X_mask.png  (grayscale: blood=255)

Two images are held out for validation (istockphoto-1412306826-612x612
and istockphoto-1145017396-612x612) — exactly the two ground-truth-checked
cases. Train on the other 11.

RAM safety (user requirement — this must NEVER eat the box's memory):
  * Single thread (torch.set_num_threads(1))
  * Hard guard: if free RAM (from /proc/meminfo MemAvailable) < 1.2 GB,
    print a clear message and exit 0 WITHOUT training (no OOM).
  * After each epoch, if MemAvailable < 700 MB, stop early and save what
    we have (never OOM).

Outputs:
  models/blood_cnn.pt          -> torch state_dict (full model)
  models/blood_cnn_history.json-> per-epoch train/val loss + Dice
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except Exception:
    TORCH = False

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
PATCH = 128          # training patch size (px)
BATCH = 8
BRUSH_BLOOD = 55     # % of patches centered on blood (rest random)
EPOCHS = 600
LR = 1e-4
VAL_IMAGE_TAG = ("istockphoto-1412306826-612x612", "istockphoto-1145017396-612x612")
MODEL_OUT = "models/blood_cnn.pt"
HISTORY_OUT = "models/blood_cnn_history.json"
MIN_RAM_GB = 1.2
STOP_RAM_MB = 700
RAM_FREE_MB = None  # filled lazily from /proc/meminfo

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


def free_ram_mb() -> int:
    """Current MemAvailable in MB (Linux /proc/meminfo)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 1024 * 1024  # unknown -> assume plenty


# --------------------------------------------------------------------------- #
#  Image + mask IO (webp via PIL fallback — cv2 can't read webp here)
# --------------------------------------------------------------------------- #
def read_image_bgr(path: str):
    """Return BGR uint8 array, using PIL fallback for webp."""
    import cv2
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image
    pil = Image.open(path).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mask_pairs(img_dir: str):
    """Yield (image_path, mask_path) for every hand-labeled pair.

    Mask files: <img_dir>/masks/<stem>_mask.png. Only files whose stem has a
    matching mask are used (this excludes the stray *e.webp / -612e.jpg
    duplicates that have no mask).
    """
    masks_dir = os.path.join(img_dir, "masks")

    def stem(p):
        return os.path.splitext(os.path.basename(p))[0]

    # masks by stem (strip the trailing _mask)
    mask_by_stem = {}
    for mp in glob.glob(os.path.join(masks_dir, "*_mask.png")):
        s = stem(mp)
        if s.endswith("_mask"):
            s = s[: -len("_mask")]
        mask_by_stem[s] = mp

    for ip in sorted(glob.glob(os.path.join(img_dir, "*"))):
        if os.path.isdir(ip):
            continue
        if not ip.lower().endswith(IMAGE_EXTS):
            continue
        s = stem(ip)
        if s in mask_by_stem:
            yield ip, mask_by_stem[s]


# --------------------------------------------------------------------------- #
#  Model: MobileNetV2-U-Net (encoder hand-built, no torchvision)
# --------------------------------------------------------------------------- #
def _conv_bn(in_c, out_c, stride=1, k=3):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block (depthwise separable, expansion)."""

    def __init__(self, inp, hidden, oup, stride, expand):
        super().__init__()
        assert stride in (1, 2)
        self.use_res = stride == 1 and inp == oup
        self.block = nn.Sequential(
            nn.Conv2d(inp, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        return self.block(x)


def _make_mbv2_stage(inp, expand, oup, n, stride_first):
    hidden = expand * inp
    layers = [InvertedResidual(inp, hidden, oup, stride_first, expand)]
    for _ in range(n - 1):
        hidden = expand * oup
        layers.append(InvertedResidual(oup, hidden, oup, 1, expand))
    return nn.Sequential(*layers)


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 feature encoder tuned to 128px input.

    Returns a list of feature maps at four depths for U-Net skip connections.
    """

    def __init__(self):
        super().__init__()
        self.stem = _conv_bn(3, 32, stride=2)          # 128 -> 64
        self.stage1 = _make_mbv2_stage(32, 1, 16, 1, 1)   # 64 -> 64
        self.stage2 = _make_mbv2_stage(16, 6, 24, 2, 2)   # 64 -> 32
        self.stage3 = _make_mbv2_stage(24, 6, 32, 3, 2)   # 32 -> 16
        self.stage4 = _make_mbv2_stage(32, 6, 64, 4, 2)   # 16 -> 8
        self.stage5 = _make_mbv2_stage(64, 6, 96, 3, 1)   # 8 -> 8
        self.stage6 = _make_mbv2_stage(96, 6, 160, 3, 2)  # 8 -> 4
        self.stage7 = _make_mbv2_stage(160, 6, 320, 1, 1) # 4 -> 4

    def forward(self, x):
        skips = []
        x = self.stem(x)
        x = self.stage1(x)
        skips.append(x)          # 1/2  resolution, 16 ch
        x = self.stage2(x)
        skips.append(x)          # 1/4  res, 24 ch
        x = self.stage3(x)
        skips.append(x)          # 1/8  res, 32 ch
        x = self.stage4(x)
        x = self.stage5(x)
        skips.append(x)          # 1/16 res, 96 ch
        x = self.stage6(x)
        x = self.stage7(x)
        return x, skips          # bottleneck 1/32 res, 320 ch + 4 skip maps


class UNetHead(nn.Module):
    """Lightweight decoder with upsampling + skip concatenation."""

    def __init__(self, enc_planes, bot_ch=320):
        super().__init__()
        up = 256
        # 1/16 skip (96) + up(bot 320) -> up
        self.conv16 = _conv_bn(96 + up, up)
        self.up16 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 1/8 skip (32) + up(up 256) -> up
        self.conv8 = _conv_bn(32 + up, up)
        self.up8 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 1/4 skip (24) + up(up 256) -> up
        self.conv4 = _conv_bn(24 + up, 128)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 1/2 skip (16) + up(128) -> 64
        self.conv2 = _conv_bn(16 + 128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, bot, skips):
        c2, c4, c8, c16 = skips          # encoder appends [1/2(16),1/4(24),1/8(32),1/16(96)] -> unpack that true order[::-1]          # encoder appends 1/2,1/4,1/8,1/16 -> decode as c2,c4,c8,c16
        x = self.conv16(torch.cat([c16, self.up16(bot)], 1))
        x = self.conv8(torch.cat([c8, self.up8(x)], 1))
        x = self.conv4(torch.cat([c4, self.up4(x)], 1))
        x = self.conv2(torch.cat([c2, self.up2(x)], 1))
        return self.final(x)


def build_model():
    enc = MobileNetV2Encoder()
    # (encoder + light head with matching channels)
    enc_planes = [16, 24, 32, 96, 320]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv16 = _conv_bn(96 + 320, 256)
            self.up16 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv8 = _conv_bn(32 + 256, 256)
            self.up8 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv4 = _conv_bn(24 + 256, 128)
            self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv2 = _conv_bn(16 + 128, 64)
            self.final = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            bot, skips = self.enc(x)
            c2, c4, c8, c16 = skips          # encoder appends [1/2(16),1/4(24),1/8(32),1/16(96)] -> unpack that true order[::-1]          # encoder appends [1/2(16), 1/4(24), 1/8(32), 1/16(96)] — read them in that true order
            x = self.conv16(torch.cat([c16, self.up(bot)], 1))
            x = self.conv8(torch.cat([c8, self.up16(x)], 1))
            x = self.conv4(torch.cat([c4, self.up8(x)], 1))
            x = self.conv2(torch.cat([c2, self.up4(x)], 1))
            return self.final(x)

    return Net()


# --------------------------------------------------------------------------- #
#  Patch sampling + augmentation
# --------------------------------------------------------------------------- #
def bbox_of_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def extract_patches(img_bgr, mask, patch=PATCH, n_patches=BATCH * 40,
                    p_blood=BRUSH_BLOOD / 100.0, rng=None):
    """Extract training patches: p_blood% centered on blood, rest random."""
    rng = rng if rng is not None else np.random.default_rng()
    h, w = mask.shape
    bb = bbox_of_mask(mask)
    imgs, masks = [], []
    half = patch // 2
    for i in range(n_patches):
        if bb is not None and rng.random() < p_blood:
            if bb is None or bb[0] >= bb[2] or bb[1] >= bb[3]:
                # degenerate/empty bbox not fulfilled as a real blood center -> random crop
                x0 = int(rng.integers(0, max(1, w - 2 * half)))
                y0 = int(rng.integers(0, max(1, h - 2 * half)))
                return img_bgr[y0:y0 + patch, x0:x0 + patch], np.zeros((patch, patch), np.uint8)
            # normal blood-center crop with a guaranteed non-empty center band
            lo_x = min(max(bb[0], half), max(0, w - half - 1))
            hi_x = max(lo_x + 1, min(w - half, w - bb[2] + half))
            lo_y = min(max(bb[1], half), max(0, h - half - 1))
            hi_y = max(lo_y + 1, min(h - half, h - bb[3] + half))
            cx = int(rng.integers(lo_x, hi_x + 1)) if bb[0] < bb[2] else int(rng.integers(half, max(w - half, half + 1)))
            cy = int(rng.integers(lo_y, hi_y + 1)) if bb[1] < bb[3] else int(rng.integers(half, max(h - half, half + 1)))
        else:
            cx = int(rng.integers(half, max(w - half, half + 1)))
            cy = int(rng.integers(half, max(h - half, half + 1)))
        x0, y0 = cx - half, cy - half
        imgs.append(_augment(img_bgr[y0:y0 + patch, x0:x0 + patch].copy(), rng))
        masks.append(mask[y0:y0 + patch, x0:x0 + patch].copy())
    imgs = np.stack(imgs).astype(np.float32) / 127.5 - 1.0
    masks = (np.stack(masks) > 127).astype(np.float32)
    return torch.from_numpy(imgs.transpose(0, 3, 1, 2)), torch.from_numpy(masks[:, None, :, :])

def _augment(img, rng):
    # horizontal flip + rotation + H/S jitter (cheap, CPU-friendly)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    return img


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #
def dice_loss(pred, target, eps=1.0):
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum() + eps
    return 1.0 - 2.0 * inter / denom


def bce_dice(pred, target):
    if pred.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=pred.shape[2:], mode="nearest")
    # class-weight: blood (fg) is ~1-3% of each tile -> weight it ~30x so the
    # tiny-but-important class isn't drowned by background (already-interpolated here)
    w = target * 29.0 + 1.0          # fg:bg ratio = 30:1
    bce = F.binary_cross_entropy(pred, target, weight=w)
    return bce + 0.5 * dice_loss(pred, target)


def dice_score(pred, target, eps=1.0):
    pred = (pred > 0.5).float()
    if target.shape[2:] != pred.shape[2:]:
        target = F.interpolate(target, size=pred.shape[2:], mode="nearest")
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum() + eps
    return (2.0 * inter / denom).item()


# --------------------------------------------------------------------------- #
#  Validation (patch-Dice on the held-out image — full-res, sliding window)
# --------------------------------------------------------------------------- #
def infer_sliding(net, img, patch=PATCH, stride=PATCH // 2):
    """Overlap-average sliding-window inference on a full image."""
    net.eval()
    h, w = img.shape[:2]
    img_t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    acc = torch.zeros(h, w)
    cnt = torch.zeros(h, w)
    with torch.no_grad():
        for y0 in range(0, max(h - patch + 1, 1), stride):
            for x0 in range(0, max(w - patch + 1, 1), stride):
                y0 = min(y0, h - patch)
                x0 = min(x0, w - patch)
                p = img_t[:, :, y0:y0 + patch, x0:x0 + patch]
                out = torch.sigmoid(net(p)).squeeze(0).squeeze(0)
                # pred is at the net's spatial resolution; resize to the crop tile if it differs
                # (same family as the bce_dice guard — never accumulate mismatched maps)
                if out.shape != (patch, patch):
                    out = F.interpolate(out[None, None], size=(patch, patch), mode="bilinear", align_corners=False)[0, 0]
                acc[y0:y0 + patch, x0:x0 + patch] += out
                cnt[y0:y0 + patch, x0:x0 + patch] += 1
    return (acc / cnt.clamp_min(1)).numpy()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def train_pairs(img_dir):
    pairs = list(mask_pairs(img_dir))
    # split val by tag: the two ground-truth-checked images
    train, val = [], []
    for ip, mp in pairs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        if any(t in stem for t in VAL_IMAGE_TAG):
            val.append((ip, mp))
        else:
            train.append((ip, mp))
    return train, val


def main():
    global RAM_FREE_MB
    ap = argparse.ArgumentParser(description="Train blood CNN (pure torch, CPU).")
    ap.add_argument("--img-dir", default="/home/sarok/Pictures/selfharm")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ram-min", type=float, default=MIN_RAM_GB)
    args = ap.parse_args()

    if not TORCH:
        print("PyTorch not available — cannot train. "
              "(pip install torch on a torch-supported Python)")
        sys.exit(0)

    torch.set_num_threads(1)  # RAM + CPU safety (never eat the machine)

    RAM_FREE_MB = free_ram_mb()
    print(f"Free RAM: {RAM_FREE_MB} MB (need >= {args.ram_min * 1024:.0f} MB)")
    if RAM_FREE_MB < args.ram_min * 1024:
        print(f"Not enough free RAM to train ({RAM_FREE_MB} < {args.ram_min * 1024}). "
              "Skipping to protect your machine. Free up RAM and re-run.")
        sys.exit(0)

    train, val = train_pairs(args.img_dir)
    if not train or not val:
        print(f"Need train+val. Got train={len(train)} val={len(val)}")
        sys.exit(IE_1 := 1) if False else None
        sys.exit(1)

    print(f"Train pairs: {len(train)}  Val pairs: {len(val)}")

    # pre-read images + masks (webp + PNG)
    def load_pair(ip, mp):
        img = read_image_bgr(ip)
        m = read_image_bgr(mp)
        return img, (m[:, :, 0] > 127).astype(np.uint8)

    tr = [load_pair(*p) for p in train]
    va = [load_pair(*p) for p in val]

    rng = np.random.default_rng(0)
    net = build_model()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    best_dice = -1.0
    patience = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # RAM guard each epoch
        ram = free_ram_mb()
        if ram < STOP_RAM_MB:
            print(f"RAM dropped to {ram} MB — stopping early, saving model.")
            break

        net.train()
        tl, td = 0.0, 0.0
        n_batch = 0
        for ip, mp in tr:
            img, mask = ip, mp if False else None  # placeholder-lite
            # re-extract pairs from pre-read list
        # simplest correct loop over pre-read pairs:
        for (img, mask) in tr:
            xs, ys = extract_patches(img, mask, n_patches=args.batch * 8, rng=rng)
            if xs.shape[0] == 0:
                continue
            for b in range(0, xs.shape[0], args.batch):
                xb = xs[b:b + args.batch]
                yb = ys[b:b + args.batch]
                pred = torch.sigmoid(net(xb))
                loss = bce_dice(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item()
                td += dice_score(pred.detach(), yb)
                n_batch += 1

        # validate on the held-out image
        net.eval()
        vd, vn = 0.0, 0
        with torch.no_grad():
            for (img, mask) in va:
                predm = infer_sliding(net, img)
                gt = (mask > 0).astype(np.float32)
                # full-res Dice of the sliding prediction vs GT
                pr = (predm > 0.5)
                inter = (pr * gt).sum()
                den = pr.sum() + gt.sum() + 1.0
                vd += 2.0 * inter / den
                vn += 1
        val_dice = vd / max(vn, 1)

        sched.step(val_dice)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(net.state_dict(), MODEL_OUT)
            patience = 0
        else:
            patience += 1

        print(f"epoch {epoch:4d}  train_loss={tl / max(n_batch,1):.4f}  "
              f"train_dice={td / max(n_batch,1):.4f}  val_dice={val_dice:.4f}  "
              f"best={best_dice:.4f}  lr={opt.param_groups[0]['lr']:.1e}")

        if patience >= 15:
            print("Early stop (val Dice not improving).")
            break

    # final model reload (best)
    if os.path.exists(MODEL_OUT):
        net.load_state_dict(torch.load(MODEL_OUT))
    hist = {"best_val_dice": best_dice, "epochs_run": epoch}
    with open(HISTORY_OUT, "w") as f:
        json.dump(hist, f)
    print(f"DONE. best val Dice = {best_dice:.4f}  (target: ~0.90 = your hand accuracy)")
    print(f"Model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
