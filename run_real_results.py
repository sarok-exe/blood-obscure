#!/usr/bin/env python3
"""Produce the REAL blood-segmentation results (the method that actually
runs on this box — the color detector — since the CNN trainer's RAM guard
honestly refuses to train here: ~865MB free vs a 1.2GB floor).

Writes real, reproducible numbers into models/results/.
"""
import glob, os, json, datetime, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from predict_blood_cnn import predict_patches_color_heuristic as color_predict
    HAVE_COLOR = True
except Exception as e:
    HAVE_COLOR = False
    print(f"color predictor import failed: {e}")

def read_image_bgr(path):
    import cv2
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

def dice(a, b, eps=1.0):
    a = (a > 0).astype(np.float32); b = (b > 0).astype(np.float32)
    inter = (a*b).sum()
    return float(2*inter/(a.sum()+b.sum()+eps)), float(inter/(a.sum()+b.sum()-inter+1e-6))

out = {}
# --- held-out val pairs (image + hand mask) ---
val = ["istockphoto-1412306826-612x612", "istockphoto-1145017396-612x612"]
for stem in val:
    ip = glob.glob(f"/home/sarok/Pictures/selfharm/{stem}.*")
    mp = f"/home/sarok/Pictures/selfharm/masks/{stem}_mask.png"
    if not ip:
        out[stem] = {"error": "image not found"}; continue
    img = read_image_bgr(ip[0])
    gt = read_image_bgr(mp)[:,:,0]
    pred = color_predict(img)
    d, iou = dice(pred, gt)
    out[stem] = {"dice": round(d,4), "iou": round(iou,4),
                 "blood_pct_gt": round(100*float((gt>0).mean()),2),
                 "blood_pct_pred": round(100*float((pred>0).mean()),2)}
# --- test photos (no ground truth -> just predicted blood %) ---
for t in ["25","27","30"]:
    tp = glob.glob(f"/home/sarok/Pictures/test/photo_2026-09-16_12-53-{t}.jpg")
    if not tp:
        out[f"test_{t}"] = {"error": "not found"}; continue
    img = read_image_bgr(tp[0])
    pred = color_predict(img)
    out[f"test_{t}"] = {"blood_pct_pred": round(100*float((pred>0).mean()),2)}

os.makedirs("models/results", exist_ok=True)
report = {
    "generated_utc": datetime.datetime.utcnow().isoformat(),
    "method": "color-detector (the pipeline that actually runs on this box)",
    "note": "CNN trainer declined to train: 865MB free < 1.2GB RAM floor (your RAM is protected). These are genuine numbers, not CNN results.",
    "results": out,
}
path = "models/results/blood_results.json"
with open(path, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print(f"\nWROTE {path}")
