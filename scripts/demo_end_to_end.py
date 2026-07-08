"""
demo_end_to_end.py
==================
A CPU-runnable proof that stitches the whole system together *without* the GPU
components, using synthetic stand-ins where CLIP/SDXL would normally run:

  1. SEARCH  - build an HNSW style index over a synthetic embedding catalog,
     run text- and image-style queries, report latency + top-k.
  2. FIDELITY- take a garment image, derive its structural conditioning, create
     a "generated" variant, and score Structural Preservation Fidelity.

Real deployment swaps the synthetic embeddings for ClipEncoder outputs and the
"generated variant" for the SDXL+ControlNet pipeline output - the surrounding
code is identical.

    python scripts/demo_end_to_end.py
"""
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashiongen.search.faiss_index import StyleIndex, _HAS_FAISS
from fashiongen.search.benchmark_latency import synth_embeddings
from fashiongen.metrics.structural_fidelity import structural_preservation_fidelity


def demo_search():
    print("=" * 60)
    print("1) STYLE SEARCH  (HNSW over CLIP-style embeddings)")
    print("=" * 60)
    N, D = 40_000, 512
    catalog = synth_embeddings(N, D, seed=1)
    ids = np.array([f"garment_{i:06d}" for i in range(N)])
    idx = StyleIndex(dim=D).build(catalog, ids=ids)
    print(f"backend={idx.backend}  catalog={N:,}x{D}")

    # a "text query" and an "image query" are just embeddings in the shared space
    text_q = synth_embeddings(1, D, seed=99)          # stands in for CLIP text
    r = idx.search(text_q, k=5)
    print("\ntext query -> top-5 ids:", [str(x) for x in r.ids[0]])
    print(f"similarities: {np.round(r.scores[0], 3)}")
    print(f"per-query latency: {r.latency_ms:.2f} ms  "
          f"({'FAISS-HNSW' if _HAS_FAISS else 'sklearn-exact fallback'})")
    return idx


def demo_fidelity():
    print("\n" + "=" * 60)
    print("2) STRUCTURAL PRESERVATION FIDELITY")
    print("=" * 60)

    # reference garment
    ref = np.full((768, 768, 3), 240, np.uint8)
    color = (70, 110, 200)
    cv2.rectangle(ref, (240, 220), (528, 660), color, -1)
    cv2.fillPoly(ref, [np.array([[240, 220], [150, 320], [190, 370], [240, 340]])], color)
    cv2.fillPoly(ref, [np.array([[528, 220], [618, 320], [578, 370], [528, 340]])], color)
    cv2.ellipse(ref, (384, 220), (60, 28), 0, 0, 180, (240, 240, 240), -1)

    # structural conditioning maps (what ControlNet would receive)
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    canny = cv2.cvtColor(cv2.Canny(gray, 100, 200), cv2.COLOR_GRAY2BGR)

    # a FAITHFUL "generated" garment: same structure, new texture + recolour
    gen = ref.copy()
    gen[np.all(np.abs(gen.astype(int) - color) < 40, axis=2)] = (60, 170, 90)
    rng = np.random.default_rng(0)
    gen = np.clip(gen + rng.normal(0, 6, gen.shape), 0, 255).astype(np.uint8)

    # a STRUCTURE-BREAKING variant for contrast
    M = np.float32([[1, 0.18, 0], [0, 1, 0]])
    broken = cv2.warpAffine(ref, M, (768, 768), borderValue=(240, 240, 240))

    good = structural_preservation_fidelity(ref, gen)
    bad = structural_preservation_fidelity(ref, broken)
    print(f"faithful regeneration : SPF = {good.as_percent():.1f}%  "
          f"(edgeF1={good.edge_f1:.2f}, IoU={good.silhouette_iou:.2f}, "
          f"SSIM={good.ssim:.2f})")
    print(f"structure-broken variant: SPF = {bad.as_percent():.1f}%")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for a, img, title in [
        (ax[0], ref, "reference garment"),
        (ax[1], canny, "ControlNet conditioning\n(canny structure)"),
        (ax[2], gen, f"faithful generation\nSPF={good.as_percent():.1f}%"),
        (ax[3], broken, f"structure broken\nSPF={bad.as_percent():.1f}%"),
    ]:
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); a.set_title(title); a.axis("off")
    plt.tight_layout()
    out = ROOT / "figures" / "fidelity_demo.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    demo_search()
    demo_fidelity()
    print("\nDone. (GPU components — CLIP/SDXL/ControlNet/TensorRT — are "
          "represented by synthetic stand-ins in this CPU demo.)")
