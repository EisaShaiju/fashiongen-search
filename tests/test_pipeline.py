"""CPU tests for the fidelity metric and the style index (no GPU needed)."""
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashiongen.metrics.structural_fidelity import structural_preservation_fidelity
from fashiongen.search.faiss_index import StyleIndex
from fashiongen.search.benchmark_latency import synth_embeddings, exact_topk, recall_at_k


def _garment(color=(70, 110, 200), size=512):
    img = np.full((size, size, 3), 240, np.uint8)
    cv2.rectangle(img, (160, 150), (352, 440), color, -1)
    cv2.ellipse(img, (256, 150), (40, 20), 0, 0, 180, (240, 240, 240), -1)
    return img


# --------------------------- fidelity ---------------------------------- #
def test_spf_identical_is_one():
    g = _garment()
    r = structural_preservation_fidelity(g, g)
    assert r.spf > 0.99


def test_spf_ignores_color_keeps_structure():
    r = structural_preservation_fidelity(_garment(), _garment(color=(200, 80, 80)))
    assert r.spf > 0.95           # recolour, same structure -> high


def test_spf_penalises_structure_break():
    ref = _garment()
    M = np.float32([[1, 0.3, 0], [0, 1, 0]])
    broken = cv2.warpAffine(ref, M, (512, 512), borderValue=(240, 240, 240))
    assert structural_preservation_fidelity(ref, broken).spf < 0.6


def test_spf_in_unit_range():
    r = structural_preservation_fidelity(_garment(), _garment(color=(10, 200, 10)))
    assert 0.0 <= r.spf <= 1.0


# ----------------------------- search ---------------------------------- #
def test_index_build_and_search_shapes():
    db = synth_embeddings(2000, 64, seed=0)
    idx = StyleIndex(dim=64).build(db)
    r = idx.search(db[:5], k=10)
    assert r.ids.shape == (5, 10)
    assert r.scores.shape == (5, 10)


def test_search_finds_self():
    db = synth_embeddings(1000, 64, seed=2)
    idx = StyleIndex(dim=64).build(db, ids=np.arange(1000))
    r = idx.search(db[:20], k=1)
    # nearest neighbour of a catalog vector should be itself
    assert (r.ids[:, 0].astype(int) == np.arange(20)).mean() > 0.95


def test_recall_metric_bounds():
    db = synth_embeddings(1500, 48, seed=3)
    q = synth_embeddings(50, 48, seed=4)
    idx = StyleIndex(dim=48).build(db)
    approx = idx.search(q, k=10).ids.astype(int)
    gt = exact_topk(db, q, 10)
    rec = recall_at_k(approx, gt)
    assert 0.0 <= rec <= 1.0
    assert rec > 0.9              # exact/HNSW should have high recall here


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
