#!/usr/bin/env python3
"""Load torchvision MobileNetV2 ImageNet weights into the blood_unet encoder.

The blood_unet MobileNetV2Encoder mirrors torchvision's MobileNetV2 feature
extractor (features.*). Key remapping:

    torchvision                    blood_unet
    features.0.*                   encoder.stem.*
    features.{i}.*  (i=1..17)      encoder.blocks.{i-1}.*
    features.18.*                  encoder.head.*
    classifier.*                   (skipped)

The InvertedResidual layout matches for expand_ratio=6 blocks
(conv.0.0/0.1 expand, conv.1.0/1.1 depthwise, conv.2/3 project). The first
block (expand_ratio=1) omits the expand conv in torchvision, so its keys are
remapped explicitly. torchvision's conv_head (features.18) outputs 1280
channels vs blood_unet's 320, so those keys are skipped on shape mismatch.
"""

import torch

from blood_unet import BloodUNet


def remap_key(key):
    """Map a torchvision MobileNetV2 state_dict key to a blood_unet encoder key.

    Returns None for keys that should be skipped (classifier, unknown).
    """
    parts = key.split(".")
    if parts[0] != "features":
        return None
    idx = int(parts[1])
    if idx == 0:
        # stem: features.0.{sub} -> encoder.stem.{sub}
        return "encoder.stem." + ".".join(parts[2:])
    if idx == 18:
        # conv_head: features.18.{sub} -> encoder.head.{sub}
        return "encoder.head." + ".".join(parts[2:])
    if 1 <= idx <= 17:
        blk = idx - 1
        sub = parts[2:]
        if idx == 1:
            # expand_ratio=1 block: torchvision conv.0.0/0.1/1/2 ->
            # blood_unet conv.0/1/3/4 (no expand conv, ReLU6 shifts indices)
            if sub[:3] == ["conv", "0", "0"]:
                return f"encoder.blocks.{blk}.conv.0." + ".".join(sub[3:])
            if sub[:3] == ["conv", "0", "1"]:
                return f"encoder.blocks.{blk}.conv.1." + ".".join(sub[3:])
            if sub[:2] == ["conv", "1"]:
                return f"encoder.blocks.{blk}.conv.3." + ".".join(sub[2:])
            if sub[:2] == ["conv", "2"]:
                return f"encoder.blocks.{blk}.conv.4." + ".".join(sub[2:])
            return None
        # expand_ratio=6 block: torchvision conv.0.0/0.1/1.0/1.1/2/3 ->
        # blood_unet flat conv.0/1/3/4/6/7 (ReLU6s occupy conv.2/5)
        if sub[:3] == ["conv", "0", "0"]:
            return f"encoder.blocks.{blk}.conv.0." + ".".join(sub[3:])
        if sub[:3] == ["conv", "0", "1"]:
            return f"encoder.blocks.{blk}.conv.1." + ".".join(sub[3:])
        if sub[:3] == ["conv", "1", "0"]:
            return f"encoder.blocks.{blk}.conv.3." + ".".join(sub[3:])
        if sub[:3] == ["conv", "1", "1"]:
            return f"encoder.blocks.{blk}.conv.4." + ".".join(sub[3:])
        if sub[:2] == ["conv", "2"]:
            return f"encoder.blocks.{blk}.conv.6." + ".".join(sub[2:])
        if sub[:2] == ["conv", "3"]:
            return f"encoder.blocks.{blk}.conv.7." + ".".join(sub[2:])
        return None
    return None


def load_pretrained_encoder(model, weights_path):
    """Load ImageNet weights into model.encoder.

    Returns (n_loaded, skipped_keys). Only keys that exist in the model with
    matching shapes are loaded; everything else is reported as skipped.
    """
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    loaded = {}
    skipped = []
    for k, v in state.items():
        new_k = remap_key(k)
        if new_k is None:
            skipped.append(k)
        elif new_k in model_state and model_state[new_k].shape == v.shape:
            loaded[new_k] = v
        else:
            skipped.append(k)
    model.load_state_dict(loaded, strict=False)
    return len(loaded), skipped


def verify_encoder(model, size=64):
    """Sanity: run the encoder forward once on a random tensor."""
    x = torch.randn(1, 3, size, size)
    skips, bottleneck = model.encoder(x)
    print(f"encoder forward OK: {len(skips)} skips, "
          f"bottleneck {tuple(bottleneck.shape)}")


if __name__ == "__main__":
    from pathlib import Path
    weights = Path(__file__).resolve().parent / "models" / "mobilenet_v2_imagenet.pth"
    m = BloodUNet()
    n, skipped = load_pretrained_encoder(m, weights)
    print(f"loaded {n} keys, skipped {len(skipped)}")
    for k in skipped:
        print(f"  skipped: {k}")
    verify_encoder(m)