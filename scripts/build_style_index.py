"""
build_style_index.py
====================
Encode a garment catalog with CLIP and build the HNSW style index.

    python scripts/build_style_index.py --config configs/search.yaml

If embeddings already exist (`embeddings_out`), the CLIP step is skipped and the
index is (re)built from them - handy for tuning HNSW params without a GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashiongen.search.faiss_index import StyleIndex


def list_images(root: str):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/search.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    emb_path = Path(cfg["embeddings_out"])
    if emb_path.exists():
        print(f"loading cached embeddings -> {emb_path}")
        embs = np.load(emb_path)
        ids = np.load(str(emb_path) + ".ids.npy", allow_pickle=True)
    else:
        # requires GPU + CLIP; imported lazily so the cached path stays CPU-only
        from fashiongen.search.clip_encoder import ClipEncoder
        paths = list_images(cfg["catalog_images"])
        print(f"encoding {len(paths):,} images with CLIP {cfg['clip_model']} ...")
        enc = ClipEncoder(model_name=cfg["clip_model"],
                          pretrained=cfg["clip_pretrained"])
        embs = enc.encode_images(paths)
        ids = np.array([Path(p).stem for p in paths])
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embs); np.save(str(emb_path) + ".ids.npy", ids)

    print(f"building HNSW index over {embs.shape[0]:,}x{embs.shape[1]} ...")
    idx = StyleIndex(dim=embs.shape[1], hnsw_M=cfg["hnsw_M"],
                     ef_construction=cfg["ef_construction"],
                     ef_search=cfg["ef_search"]).build(embs, ids=ids)
    idx.save(cfg["index_out"])
    print(f"saved index -> {cfg['index_out']}  (backend={idx.backend})")


if __name__ == "__main__":
    main()
