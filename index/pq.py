"""Product quantization for the edge index variant (Ch. 14, Lab D)."""
from __future__ import annotations
import numpy as np


def train_pq(vecs: np.ndarray, m: int, k: int = 16, iters: int = 8, seed=0):
    n, d = vecs.shape
    sub = d // m
    rng = np.random.default_rng(seed)
    books = []
    for j in range(m):
        X = vecs[:, j * sub:(j + 1) * sub]
        C = X[rng.choice(n, size=min(k, n), replace=False)].copy()
        for _ in range(iters):
            a = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            for c in range(len(C)):
                if (a == c).any():
                    C[c] = X[a == c].mean(0)
        books.append(C)
    return books


def encode_pq(vecs: np.ndarray, books):
    m = len(books)
    sub = vecs.shape[1] // m
    out = np.zeros_like(vecs)
    for j, C in enumerate(books):
        X = vecs[:, j * sub:(j + 1) * sub]
        a = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        out[:, j * sub:(j + 1) * sub] = C[a]
    return out
