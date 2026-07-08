"""
make_conditioning.py
====================
Precompute ControlNet structural conditioning maps for every garment image and
write a training manifest (JSONL) of {image, conditioning, caption, meta}.

Supported conditioning types:
    canny    - Canny edges (seams, hems, collar/sleeve lines, print boundaries)
    softedge - blurred/thinned edges (gentler structural guidance)
    mask     - filled garment silhouette (shape-only conditioning)

Fully runnable on CPU (OpenCV). This is the stage that turns raw garment images
into (target, structure) pairs the ControlNet learns to follow.

    python -m data.make_conditioning \
        --images data/raw/viton_hd/train/cloth \
        --out data/processed/viton_hd --cond canny
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def canny_map(gray: np.ndarray, low=100, high=200) -> np.ndarray:
    e = cv2.Canny(gray, low, high)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)


def softedge_map(gray: np.ndarray) -> np.ndarray:
    e = cv2.Canny(gray, 80, 160)
    e = cv2.GaussianBlur(e, (5, 5), 0)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)


def mask_map(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (m > 0).mean() > 0.6:                       # background got selected; flip
        m = 255 - m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)


COND = {"canny": canny_map, "softedge": softedge_map, "mask": mask_map}


def process(images_dir, out_dir, cond="canny", captions=None, limit=None):
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    cond_dir = out_dir / f"cond_{cond}"
    cond_dir.mkdir(parents=True, exist_ok=True)
    fn = COND[cond]

    caps = {}
    if captions and Path(captions).exists():
        for line in open(captions):
            r = json.loads(line)
            caps[r["id"]] = r.get("caption", "")

    paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if limit:
        paths = paths[:limit]

    manifest_path = out_dir / f"manifest_{cond}.jsonl"
    n = 0
    with open(manifest_path, "w") as mf:
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cond_img = fn(gray)
            cpath = cond_dir / (p.stem + ".png")
            cv2.imwrite(str(cpath), cond_img)
            mf.write(json.dumps({
                "image": str(p),
                "conditioning": str(cpath),
                "caption": caps.get(p.stem, ""),
                "meta": {"id": p.stem},
            }) + "\n")
            n += 1
    print(f"wrote {n} conditioning maps + manifest -> {manifest_path}")
    return manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", choices=list(COND), default="canny")
    ap.add_argument("--captions", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    process(args.images, args.out, args.cond, args.captions, args.limit)


if __name__ == "__main__":
    main()
