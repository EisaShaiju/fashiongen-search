"""
dataset.py
==========
Garment dataset for SDXL LoRA / ControlNet fine-tuning.

Each record yields:
    pixel_values   : the target garment image (SDXL native resolution)
    conditioning   : the ControlNet structural map (canny / softedge / pose /
                     mask) - the structure we want preserved
    caption        : a text prompt describing the garment (for text conditioning)

Captions come from the dataset metadata (e.g. H&M `detail_desc`, DeepFashion
attributes) via data/prepare_captions.py; conditioning maps are precomputed by
data/make_conditioning.py so training stays I/O bound rather than CPU bound.

Requires torch + torchvision. Pure-data logic (manifest parsing, caption
templating) is import-safe without torch so it can be unit-tested.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def load_manifest(path: str) -> list[dict]:
    """Read a JSONL manifest: one {image, conditioning, caption} per line."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_caption(meta: dict, templates=None) -> str:
    """Compose a natural caption from structured garment metadata.

    Falls back gracefully when fields are missing. Example output:
    "a photo of a red floral midi dress, cotton, v-neck, short sleeves,
     product shot on white background"
    """
    templates = templates or [
        "a product photo of a {color} {category}",
        "a {color} {category}, {attributes}",
        "studio product shot of a {color} {category}, {attributes}, white background",
    ]
    color = meta.get("color", "").strip()
    category = meta.get("category", "garment").strip()
    attrs = meta.get("attributes", [])
    if isinstance(attrs, str):
        attrs = [a.strip() for a in attrs.split(",") if a.strip()]
    attr_str = ", ".join(attrs[:5])
    t = random.choice(templates)
    return t.format(color=color, category=category, attributes=attr_str).replace(" ,", ",").strip(", ")


# --------------------------------------------------------------------------- #
# Torch dataset (imported lazily so this file is safe without torch)
# --------------------------------------------------------------------------- #
def make_torch_dataset(manifest_path: str, resolution: int = 1024,
                       caption_dropout: float = 0.05,
                       cond_type: str = "canny"):
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image

    rows = load_manifest(manifest_path)

    img_tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),          # SDXL VAE expects [-1, 1]
    ])
    cond_tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),                       # ControlNet expects [0, 1]
    ])

    class GarmentDataset(Dataset):
        def __init__(self):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            image = Image.open(r["image"]).convert("RGB")
            cond = Image.open(r["conditioning"]).convert("RGB")
            caption = r.get("caption") or build_caption(r.get("meta", {}))
            if random.random() < caption_dropout:    # CFG unconditional training
                caption = ""
            return {
                "pixel_values": img_tf(image),
                "conditioning_pixel_values": cond_tf(cond),
                "caption": caption,
            }

    return GarmentDataset()
