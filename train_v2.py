#!/usr/bin/env python3
"""
Train the MobileNetV2-UNet (frozen ImageNet-pretrained encoder) on manual
blood masks. Exports the best checkpoint as blood_unet_v2.pt and an ONNX
model (logits output, dynamic batch) as blood_unet_v2.onnx.

Usage:
  train_v2.py --images DIR --masks DIR --out DIR [--epochs 40] [--size 256]
              [--batch 4] [--lr 1e-3] [--val-split 0.3] [--seed 42]
              [--unfreeze N] [--dark-weight W]
"""

import argparse
import os
import random
import sys

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from blood_unet import BloodUNet
from load_pretrained_encoder import load_pretrained_encoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "models", "mobilenet_v2_imagenet.pth")
EARLY_STOP_PATIENCE = 10


def find_pairs(images_dir, masks_dir):
    """Return [(image_path, mask_path)] for images that have a mask."""
    pairs = []
    for f in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        mask = os.path.join(masks_dir, stem + "_mask.png")
        if os.path.isfile(mask):
            pairs.append((os.path.join(images_dir, f), mask))
    return pairs


def hue_shift(img, delta):
    """Shift hue by delta in [-0.5, 0.5] (fraction of the hue wheel)."""
    if delta == 0:
        return img
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hsv[..., 0] = (hsv[..., 0] + delta * 255.0) % 255.0
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def train_augment(img, mask, size):
    """Resize to size+32, random crop to size, flips, rot90, color jitter."""
    s = size + 32
    img = img.resize((s, s), Image.BILINEAR)
    mask = mask.resize((s, s), Image.NEAREST)
    x0 = random.randint(0, 32)
    y0 = random.randint(0, 32)
    img = img.crop((x0, y0, x0 + size, y0 + size))
    mask = mask.crop((x0, y0, x0 + size, y0 + size))
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.25:
        img = img.rotate(90, expand=True)
        mask = mask.rotate(90, expand=True)
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-0.2, 0.2))
        img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-0.2, 0.2))
        img = ImageEnhance.Color(img).enhance(1 + random.uniform(-0.2, 0.2))
        img = hue_shift(img, random.uniform(-0.05, 0.05))
    return img, mask


def to_tensor(img, mask):
    """RGB ImageNet-normalized image tensor + 0/1 float mask tensor."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    t = torch.from_numpy(arr.transpose(2, 0, 1).copy())
    m = np.asarray(mask, dtype=np.float32)
    m = (m > 127).astype(np.float32)
    return t, torch.from_numpy(m)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pairs, size, train, dark_weight=0.0):
        self.pairs = pairs
        self.size = size
        self.train = train
        self.dark_weight = dark_weight

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img = Image.open(self.pairs[i][0]).convert("RGB")
        mask = Image.open(self.pairs[i][1]).convert("L")
        if self.train:
            img, mask = train_augment(img, mask, self.size)
        else:
            img = img.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)
        t, m = to_tensor(img, mask)
        if self.dark_weight > 0:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            v = arr.max(axis=2)  # value channel 0..1
            w = 1.0 + self.dark_weight * (1.0 - v)
            w = torch.from_numpy(w.astype(np.float32))
        else:
            w = torch.ones(m.shape)
        return t, m, w


def soft_dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    eps = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


def dice_score(logits, targets):
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * inter + 1.0) / (union + 1.0)).mean().item()


def main():
    ap = argparse.ArgumentParser(description="Train MobileNetV2-UNet v2")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unfreeze", type=int, default=0,
                    help="unfreeze last N encoder blocks + head (encoder LR = lr/10)")
    ap.add_argument("--dark-weight", type=float, default=0.0,
                    help="per-pixel BCE weight for dark pixels: 1 + W*(1-V)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = find_pairs(args.images, args.masks)
    if not pairs:
        sys.exit(f"no image/mask pairs found in {args.images} / {args.masks}")
    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * args.val_split)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"pairs: {len(pairs)}  train: {len(train_pairs)}  val: {len(val_pairs)}")

    model = BloodUNet()
    n_loaded, skipped = load_pretrained_encoder(model, WEIGHTS)
    print(f"pretrained encoder: {n_loaded} keys loaded, {len(skipped)} skipped")
    for p in model.encoder.parameters():
        p.requires_grad = False
    if args.unfreeze > 0:
        for blk in model.encoder.blocks[-args.unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.encoder.head.parameters():
            p.requires_grad = True
        model.encoder.train()  # allow BN stats to update
        print(f"unfreezing last {args.unfreeze} encoder blocks + head")
    else:
        model.encoder.eval()  # freeze BN running stats during training

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [p for p in model.parameters()
                  if p.requires_grad and p not in set(model.encoder.parameters())]
    print(f"trainable params: {sum(p.numel() for p in enc_params + dec_params)}")
    opt = torch.optim.Adam([
        {"params": dec_params, "lr": args.lr},
        {"params": enc_params, "lr": args.lr / 10},
    ])

    # RAM guard: shrink batch/size when memory is tight.
    batch = args.batch
    size = args.size
    avail_mb = psutil.virtual_memory().available / 1e6
    if avail_mb < 900:
        batch = max(1, batch // 2)
        print(f"RAM guard: {avail_mb:.0f}MB available < 900MB -> batch {args.batch}->{batch}")
    if avail_mb < 600:
        size = 192
        print(f"RAM guard: {avail_mb:.0f}MB available < 600MB -> size {args.size}->{size}")

    train_ds = Dataset(train_pairs, size, train=True, dark_weight=args.dark_weight)
    val_ds = Dataset(val_pairs, size, train=False, dark_weight=args.dark_weight)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch,
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch,
                                             shuffle=False, num_workers=0)

    best_dice = -1.0
    patience = 0
    best_path = os.path.join(args.out, "blood_unet_v2.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.unfreeze == 0:
            model.encoder.eval()  # keep encoder BN frozen
        total_loss = 0.0
        n = 0
        for x, y, w in train_loader:
            opt.zero_grad()
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(
                logits, y.unsqueeze(1), pos_weight=torch.tensor(5.0),
                weight=w.unsqueeze(1))
            loss = bce + soft_dice_loss(logits, y.unsqueeze(1))
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total_loss / n

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        n = 0
        with torch.no_grad():
            for x, y, w in val_loader:
                logits = model(x)
                bce = F.binary_cross_entropy_with_logits(
                    logits, y.unsqueeze(1), pos_weight=torch.tensor(5.0),
                    weight=w.unsqueeze(1))
                loss = bce + soft_dice_loss(logits, y.unsqueeze(1))
                val_loss += loss.item() * x.size(0)
                val_dice += dice_score(logits, y.unsqueeze(1)) * x.size(0)
                n += x.size(0)
        val_loss /= n
        val_dice /= n
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  "
              f"best={max(best_dice, val_dice):.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"early stop at epoch {epoch} (patience {EARLY_STOP_PATIENCE})")
                break

    # Export the best checkpoint to ONNX (logits output, dynamic batch).
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    onnx_path = os.path.join(args.out, "blood_unet_v2.onnx")
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=13, dynamo=False)
    print(f"saved {best_path}  (best val_dice={best_dice:.4f})")
    print(f"saved {onnx_path}")


if __name__ == "__main__":
    main()