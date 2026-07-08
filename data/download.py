"""
download.py
===========
Fetch the garment datasets used by this project. Because most fashion datasets
require accepting a licence / Kaggle auth, this script automates what it can and
prints exact manual steps for the rest.

Datasets
--------
1. DeepFashion (CUHK MMLAB) - 800K+ clothing images with categories, attributes,
   landmarks, masks; plus DeepFashion-MultiModal (parsing, keypoints, text).
   Main corpus for LoRA fine-tuning + CLIP catalog embeddings.
     https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
   (non-commercial research use; some subsets need an agreement/password)

2. VITON-HD - 13,679 garment/model pairs @ 1024x768 with segmentation maps,
   agnostic masks and pose. Ideal ControlNet structural conditioning.
     https://github.com/shadow2496/VITON-HD  (Google Drive link in repo)
     https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset

3. DressCode - ~50K high-res images across upper/lower/dresses with masks.
     (request access from the authors' repo)

4. H&M Personalized Fashion Recommendations (Kaggle) - ~105K product images with
   rich text metadata (product name, colour, department, detail_desc).
   Great for CLIP text<->image search catalog.
     https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
   Enhanced HF mirror with precomputed embeddings + image URLs:
     https://huggingface.co/datasets/Qdrant/hm_ecommerce_products

To reach the "500K+ style embeddings" catalog: combine DeepFashion (~800K) and/or
H&M (~105K) product images (optionally multiple views/crops per item).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("data/raw")


def _run(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_hm(out: Path):
    """H&M via the Kaggle API (needs `pip install kaggle` + ~/.kaggle/kaggle.json)."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        _run(["kaggle", "competitions", "download", "-c",
              "h-and-m-personalized-fashion-recommendations", "-p", str(out)])
        for z in out.glob("*.zip"):
            _run(["unzip", "-o", str(z), "-d", str(out)])
    except Exception as e:
        print(f"[H&M] automated download failed ({e}).")
        print("Accept the competition rules on Kaggle, then set up the Kaggle "
              "API token (~/.kaggle/kaggle.json) and re-run.")


def download_viton_hd(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    print("[VITON-HD] Manual step: open the Google Drive link in "
          "https://github.com/shadow2496/VITON-HD and download the dataset zip, "
          f"then unzip into {out}. (Kaggle mirror also available.)")


def download_deepfashion(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    print("[DeepFashion] Manual step: request access + download from "
          "https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html "
          f"(Category/Attribute subset is open) and unzip into {out}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hm", "viton_hd", "deepfashion", "all"],
                    default="all")
    ap.add_argument("--out", default=str(DATA_ROOT))
    args = ap.parse_args()
    out = Path(args.out)

    if args.dataset in ("hm", "all"):
        download_hm(out / "hm")
    if args.dataset in ("viton_hd", "all"):
        download_viton_hd(out / "viton_hd")
    if args.dataset in ("deepfashion", "all"):
        download_deepfashion(out / "deepfashion")


if __name__ == "__main__":
    main()
