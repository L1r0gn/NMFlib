"""Fixed-seed dataset generation for T16 (regenerable, bit-for-bit).

Four non-square datasets covering the required variety:
  A1 lowrank_small        12x9,  k=3, exact rank 3, strictly positive
  A2 lowrank_medium       40x25, k=5, exact rank 5, strictly positive
  B  noisy_lowrank        30x20, k=4, rank-4 signal + clipped Gaussian noise
  C  sparse_counts_rankdef 25x18, configured k=4 but TRUE RANK 2 (rank
     deficient: actual rank < configured k), integer counts with many exact
     zeros, no all-zero row/column.

Usage:  python generate_data.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA_SEEDS = {"A1": 101, "A2": 102, "B": 103, "C": 104}


def _lowrank(rng, m, n, r, lo=0.1, hi=1.0):
    w0 = rng.uniform(lo, hi, size=(m, r))
    h0 = rng.uniform(lo, hi, size=(r, n))
    return w0, h0, w0 @ h0


def build_datasets() -> dict:
    out = {}

    rng = np.random.default_rng(DATA_SEEDS["A1"])
    w0, h0, v = _lowrank(rng, 12, 9, 3)
    out["A1"] = dict(
        v=v, w0=w0, h0=h0, k=3, true_rank=3,
        description="low-rank synthetic, small, strictly positive",
    )

    rng = np.random.default_rng(DATA_SEEDS["A2"])
    w0, h0, v = _lowrank(rng, 40, 25, 5)
    out["A2"] = dict(
        v=v, w0=w0, h0=h0, k=5, true_rank=5,
        description="low-rank synthetic, medium, strictly positive",
    )

    rng = np.random.default_rng(DATA_SEEDS["B"])
    w0, h0, base = _lowrank(rng, 30, 20, 4)
    noise = rng.normal(0.0, 0.05 * float(base.mean()), size=base.shape)
    v = np.maximum(base + noise, 0.0)  # clip keeps every entry non-negative
    out["B"] = dict(
        v=v, w0=w0, h0=h0, k=4, true_rank=4,
        description="rank-4 signal + Gaussian noise (sigma=0.05*mean), clipped at 0",
    )

    rng = np.random.default_rng(DATA_SEEDS["C"])
    # Sparse count factors: true rank 2, configured k=4 -> rank deficient.
    w0 = rng.poisson(0.6, size=(25, 2)).astype(np.float64)
    h0 = rng.poisson(0.8, size=(2, 18)).astype(np.float64)
    # Guarantee no all-zero row of W0 / column of H0 (=> no all-zero row or
    # column of V) without raising the rank above 2.
    for i in np.flatnonzero(~w0.any(axis=1)):
        w0[i, rng.integers(0, 2)] = 1.0
    for j in np.flatnonzero(~h0.any(axis=0)):
        h0[rng.integers(0, 2), j] = 1.0
    v = w0 @ h0
    assert np.all(v >= 0) and np.any(v == 0)
    assert np.all(v.sum(axis=1) > 0) and np.all(v.sum(axis=0) > 0)
    out["C"] = dict(
        v=v, w0=w0, h0=h0, k=4, true_rank=2,
        description="sparse integer counts, exact zeros, true rank 2 < configured k=4",
    )

    for name, d in out.items():
        v = d["v"]
        assert np.all(np.isfinite(v)) and np.all(v >= 0), name
        d["seed"] = DATA_SEEDS[name]
        d["shape"] = list(v.shape)
        d["numeric_rank"] = int(np.linalg.matrix_rank(v))
        d["zero_fraction"] = float(np.mean(v == 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets()
    meta = {}
    for name, d in datasets.items():
        np.savez(
            outdir / f"{name}.npz",
            v=d["v"], w0=d["w0"], h0=d["h0"],
        )
        meta[name] = {
            key: d[key]
            for key in (
                "k", "true_rank", "numeric_rank", "seed", "shape",
                "zero_fraction", "description",
            )
        }
        print(
            f"{name}: shape={tuple(d['shape'])} k={d['k']} "
            f"true_rank={d['true_rank']} numeric_rank={d['numeric_rank']} "
            f"zeros={d['zero_fraction']:.3f}"
        )
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(datasets)} datasets to {outdir}")


if __name__ == "__main__":
    main()