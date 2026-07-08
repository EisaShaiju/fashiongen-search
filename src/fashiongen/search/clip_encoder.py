"""
clip_encoder.py
===============
Multimodal CLIP encoder that maps garment images AND free-text style queries
into one shared, L2-normalised embedding space. These embeddings are what the
FAISS/HNSW index is built over, enabling both text->garment and image->garment
("more like this") search.

Uses OpenCLIP (open_clip_torch). Requires torch + open_clip and, for real
throughput, a GPU. The rest of the search stack (indexing, querying, latency
benchmarking) does not depend on this module - it consumes the embeddings it
produces, so it can be developed and tested independently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class ClipEncoder:
    """Thin wrapper around an OpenCLIP model producing normalised embeddings."""

    def __init__(self,
                 model_name: str = "ViT-L-14",
                 pretrained: str = "laion2b_s32b_b82k",
                 device: str | None = None,
                 amp: bool = True):
        import torch
        import open_clip

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp and self.device == "cuda"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embed_dim = self.model.visual.output_dim

    @property
    def dim(self) -> int:
        return int(self.embed_dim)

    def _norm(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @  property
    def _autocast(self):
        if self.amp:
            return self.torch.autocast("cuda")
        from contextlib import nullcontext
        return nullcontext()

    def encode_images(self, images: Iterable, batch_size: int = 256) -> np.ndarray:
        """Encode a list/iterable of PIL images -> (N, D) float32, L2-normalised."""
        from PIL import Image
        torch = self.torch
        out = []
        batch = []
        def flush():
            if not batch:
                return
            t = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            with torch.no_grad(), self._autocast:
                f = self.model.encode_image(t).float()
                f = self._norm(f)
            out.append(f.cpu().numpy().astype(np.float32))
            batch.clear()
        for im in images:
            if not isinstance(im, Image.Image):
                im = Image.open(im).convert("RGB")
            batch.append(im)
            if len(batch) >= batch_size:
                flush()
        flush()
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        """Encode style queries / product descriptions -> (N, D) float32."""
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            tok = self.tokenizer(chunk).to(self.device)
            with torch.no_grad(), self._autocast:
                f = self.model.encode_text(tok).float()
                f = self._norm(f)
            out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), np.float32)


def encode_catalog(image_paths: list[str], out_path: str,
                   ids: list[str] | None = None, **kw) -> np.ndarray:
    """Encode a whole garment catalog to disk as a memmap-friendly .npy.

    Saves embeddings to `out_path` and a parallel ids file (`<out>.ids.npy`).
    """
    enc = ClipEncoder(**kw)
    embs = enc.encode_images(image_paths)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embs)
    if ids is None:
        ids = [Path(p).stem for p in image_paths]
    np.save(str(out_path) + ".ids.npy", np.array(ids))
    return embs
