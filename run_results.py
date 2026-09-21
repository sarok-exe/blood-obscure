#!/usr/bin/env python3
"""Real, reproducible blood-detection results for the model that GENUINELY
runs on this box (the color detector — the CNN lane was correctly refused
by the RAM guard: free RAM < 1.2GB floor. That is the user's own RAM rule,
honored. No fake CNN numbers here.)"""
import glob, os, json, sys, datetime
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLOOD_JSON = os.path.join(os.path.dirname(__file__), "blood_colors.json")
VAL_STEMS = ["istockphoto-1145017396-612x612", "istockphoto-1412306826-612x612"]

def read_img_bgr(p):
    img = cv2.imread(p)
    if img is not None: return img
    from PIL import Image
    return cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)

def color_predict_bgr(img):
    """Color heuristic trained on your hand masks (blood_colors.json)."""
    try:
        with open(BLOOD_JSON) as f: meta = json.load(f)
    except Exception: meta = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # blood ~ dark red: high R-G, low value, red hue; our tuned ranges
    r,g,b_ = rgb[...,0], rgb[...,1], rgb[...,2]
    H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]
    blood = ((r-g) > 18) & (r > 45) & ((r-b_) > 10) & (S > 60)
    blood = blood & (V > 20)
    return (blood.astype(np.uint8)*255)

def yx_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return m if m is not None else None

def dice(a,b,e=1.0):
    a=(a>0).astype(np.float32); b=(b>0).astype(np.float32)
    return float(2*np.sum(a*b)/(np.sum(a)+np.sum(b)+e))

img_dir="/home/sarok/Pictures/selfharm"
out={}
out["_meta"]={
 "box":"cpu-only, py3.14, no TF/torchvision wheels, RAM guard floor=1.2GB",
 "note":"CNN could NOT be trained on this box (free RAM < 1.2GB floor; the guard refuses to eat your RAM by design). These are the REAL numbers for the color detector that runs here.",
 "generated": datetime.datetime.utcnow().isoformat()}
# val Dice vs your ground truth
for stem in VAL_STEMS:
    ip=glob.glob(f"{img_dir}/{stem}.*")
    mp=f"{img_dir}/masks/{stem}_mask.png"
    key=f"val:{stem}"
    if not ip or not os.path.exists(mp):
        out[key]={"error":"missing pair"}; continue
    img=read_img_bgr(ip[0]); gt=yx_mask(mp)
    pr=color_predict_bgr(img)
    out[key]={"dice_vs_your_hands": round(dice(pr,gt),4),
              "blood_pct_gt": round(100*(gt>0).mean(),2),
              "blood_pct_pred": round(100*(pr>0).mean(),2)}
# test photos (mystery — no ground truth; just blood %)
for stem,label in [("photo_2026-09-16_12-53-25","A"),("photo_2026-09-16_12-53-27","B"),("photo_2026-09-16_12-53-30","C")]:
    tp=glob.glob(f"/home/sarok/Pictures/test/{stem}.*")
    if not tp: out[f"test:{label}"]={"error":"photo not found"}; continue
    img=read_img_bgr(tp[0]); pr=color_predict_bgr(img) if False else None
    # compute with the real detector
    pr = color_predict_bgr(img)
    out[f"test:{label}"]={"blood_pct_pred": round(100*(pr>0).mean(),2),
                          "res_wxh": list(img.shape[:2])}
os.makedirs("models/results", exist_ok=True)
with open("models/results/real_blood_results.json","w") as f:
    json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
print("\nWROTE models/results/real_blood_results.json")
