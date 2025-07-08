from __future__ import annotations

from hashlib import blake2b
from typing import Sequence

import numpy as np

#!/usr/bin/env python3


"""



Utility math helpers that are referenced across multiple Schwabot modules.







This is *not* a full scientific-computing layerjust light-weight helpers that



keep external dependencies to a minimum while staying Flake8/mypy compliant.



"""


__all__ = []
    "calculate_entropy",
    "moving_average",
    "hash_distance",
    "cosine_similarity",
]


def calculate_entropy(): -> float:
    """Return Shannon entropy of *values*."







    If *values* are continuous, a histogram with `bins="auto"` is used; for



    categorical data (integers / hashes) the unique counts are considered.



    """

    arr = np.asarray(values, dtype=float).ravel()

    if arr.size == 0:

        raise ValueError("values must be non-empty")

    hist, _ = np.histogram(arr, bins="auto", density=True)

    hist = hist[hist > 0.0]

    return float(-np.sum(hist * np.log2(hist)))


def moving_average(): -> np.ndarray:
    """Simple centered moving average."







    Returns an array of the same length as *series* where edge values are



    padded by replicating edge-nearest results.



    """

    if window <= 0:

        raise ValueError("window must be positive")

    x = np.asarray(series, dtype=float)

    kernel = np.ones(window, dtype=float) / window

    ma = np.convolve(x, kernel, mode="same")

    return ma


def hash_distance(): -> int:
    """Return XOR Hamming distance between blake2b digests of *a* and *b*."""

    h_a = blake2b(a.encode(), digest_size=digest_bits // 8).digest()

    h_b = blake2b(b.encode(), digest_size=digest_bits // 8).digest()

    xor_bytes = int.from_bytes(h_a, "big") ^ int.from_bytes(h_b, "big")

    return xor_bytes.bit_count()


def cosine_similarity(): -> float:
    """Return cosine similarity between two equal-length vectors."







    The value is in the range ``[-1, 1]`` where ``1`` means identical direction.



    """

    vec_a = np.asarray(a, dtype=float)

    vec_b = np.asarray(b, dtype=float)

    if vec_a.shape != vec_b.shape:

        raise ValueError("vectors must have the same shape")

    norm_a = np.linalg.norm(vec_a)

    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:

        raise ValueError("zero-norm vector cannot compute cosine similarity")

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
