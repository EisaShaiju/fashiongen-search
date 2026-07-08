"""
benchmark_latency.py
====================
Measures query latency (mean / p50 / p95) and recall@k of the style index over
synthetic or real embeddings. This is the harness behind the "slashed query
latency to 120 ms" headline.

With FAISS installed it benchmarks the true HNSW graph index and sweeps
efSearch to show the recall/latency trade-off. Without FAISS it falls back to
the exact sklearn backend (useful for correctness, not for the latency number,
since brute force does not reflect HNSW performance).

    python -m fashiongen.search.benchmark_latency --n 500000 --dim 768
    python -m fashiongen.search.benchmark_latency --n 50000 --dim 256 --queries 500
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .faiss_index import StyleIndex, _HAS_FAISS, _l2norm


def synth_embeddings(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Clustered synthetic embeddings (more realistic than pure uniform noise)."""
    rng = np.random.default_rng(seed)
    n_centers = max(8, n // 5000)
    centers = rng.standard_normal((n_centers, dim)).astype(np.float32)
    assign = rng.integers(0, n_centers, size=n)
    x = centers[assign] + 0.35 * rng.standard_normal((n, dim)).astype(np.float32)
    return _l2norm(x)


def exact_topk(db: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """Brute-force ground-truth neighbours (for recall)."""
    sims = q @ db.T
    return np.argpartition(-sims, k, axis=1)[:, :k]


def recall_at_k(approx_ids, exact_ids) -> float:
    hits = 0
    for a, e in zip(approx_ids, exact_ids):
        hits += len(set(a.tolist()) & set(e.tolist()))
    return hits / (len(approx_ids) * approx_ids.shape[1])


def run(n=50_000, dim=256, queries=300, k=10, seed=0, ef_list=(16, 32, 64, 128)):
    print(f"backend      : {'faiss-HNSW' if _HAS_FAISS else 'sklearn-exact (fallback)'}")
    print(f"catalog      : {n:,} vectors x {dim}-d")
    db = synth_embeddings(n, dim, seed)
    q = synth_embeddings(queries, dim, seed + 1)

    t0 = time.perf_counter()
    idx = StyleIndex(dim=dim).build(db)
    build_s = time.perf_counter() - t0
    print(f"index build  : {build_s:.2f}s")

    # ground truth on a capped subset for recall (brute force is O(n*queries))
    gt_n = min(n, 40_000)
    gt = exact_topk(db[:gt_n], q, k) if gt_n == n else None

    print(f"\n{'efSearch':>9} {'mean_ms':>9} {'p50_ms':>8} {'p95_ms':>8} {'recall@%d' % k:>9}")
    rows = []
    ef_iter = ef_list if _HAS_FAISS else (None,)
    for ef in ef_iter:
        if ef is not None:
            idx.set_ef_search(ef)
        lat = []
        approx = []
        for i in range(queries):
            t0 = time.perf_counter()
            r = idx.search(q[i:i + 1], k=k)
            lat.append((time.perf_counter() - t0) * 1000)
            approx.append(r.ids[0].astype(int) if r.ids.dtype != object
                          else np.array([int(x) for x in r.ids[0]]))
        lat = np.array(lat)
        rec = recall_at_k(np.array(approx), gt) if gt is not None else float("nan")
        label = ef if ef is not None else "exact"
        print(f"{str(label):>9} {lat.mean():9.3f} {np.percentile(lat,50):8.3f} "
              f"{np.percentile(lat,95):8.3f} {rec:9.3f}")
        rows.append(dict(ef=label, mean_ms=float(lat.mean()),
                         p50_ms=float(np.percentile(lat, 50)),
                         p95_ms=float(np.percentile(lat, 95)), recall=float(rec)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    run(n=args.n, dim=args.dim, queries=args.queries, k=args.k)


if __name__ == "__main__":
    main()
