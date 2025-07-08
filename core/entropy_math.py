from __future__ import annotations
import math
import logging
from collections import Counter
from typing import Iterable, Sequence, Union
    import cupy as cp

    import numpy as np
        import numpy as np

#!/usr/bin/env python3
"""Entropy Math 📊

Provides reusable entropy / information-theory helpers used by:
  • slot_state_mapper.py  (per-slot entropy)
  • digest_mapper.py      (entropy-of-digest, Hamming weight, transition entropy)
  • vector_registry.py    (feature extraction & similarity scoring)

Implemented metrics:
  * shannon_entropy(values, base=2)          – continuous or discrete data
  * transition_entropy(sequence)             – entropy of state changes (Markov-1)
  * hamming_weight(bits: bytes)              – number of 1-bits in digest
  * bit_entropy(digest: bytes)               – Shannon entropy of 256-bit digest
  * normalized_entropy(values)               – scale 0-1 for comparison

CUDA Integration:
- GPU-accelerated entropy calculations with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""
# CUDA Integration with Fallback
try:
    USING_CUDA = True
    _backend = 'cupy (GPU)'
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = 'numpy (CPU)'
    xp = np

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info("⚡ EntropyMath using GPU acceleration: {0}".format(_backend))
else:
    logger.info("🔄 EntropyMath using CPU fallback: {0}".format(_backend))

Number = Union[int, float]

# ---------------------------------------------------------------------------
# Core entropy helpers
# ---------------------------------------------------------------------------


def shannon_entropy(values: Sequence[Number], *, base: int = 2) -> float:
    """Return Shannon entropy of *values* (list of numbers) using histogram bins.

    For <64 samples we default to len(values) unique bins (exact frequencies).
    For large arrays we use GPU/CPU histogram with sqrt(n) bins for speed.
    """
    n = len(values)
    if n == 0:
        return 0.0

    if n > 64:
        # heuristic: sqrt(n) bins
        bins = int(math.sqrt(n))

        if USING_CUDA and cp.cuda.is_available():
            try:
                # GPU histogram
                values_gpu = cp.asarray(values, dtype=cp.float32)
                counts, _ = cp.histogram(values_gpu, bins=bins, density=False)
                probs = counts[counts > 0] / n
                probs = cp.asnumpy(probs)  # convert back to CPU for entropy calc
            except Exception as e:
                logger.warning("GPU histogram failed, falling back to CPU: {0}".format(e))
                # Fallback to CPU
                values_cpu = np.array(values, dtype=np.float32)
                counts, _ = np.histogram(values_cpu, bins=bins, density=False)
                probs = counts[counts > 0] / n
        else:
            # CPU histogram
            values_cpu = np.array(values, dtype=np.float32)
            counts, _ = np.histogram(values_cpu, bins=bins, density=False)
            probs = counts[counts > 0] / n
    else:
        counts = Counter(values)
        probs = [c / n for c in counts.values()]

    log_fn = _log_lookup(base)
    return -sum(p * log_fn(p) for p in probs if p > 0)


def transition_entropy(sequence: Sequence[int], *, base: int = 2) -> float:
    """Entropy of first-order transitions between discrete states in *sequence*."""
    if len(sequence) < 2:
        return 0.0

    if USING_CUDA and cp.cuda.is_available() and len(sequence) > 100:
        try:
            # GPU transition counting
            seq_gpu = cp.asarray(sequence, dtype=cp.int32)
            transitions = {}

            # Count transitions on GPU
            for i in range(len(seq_gpu) - 1):
                pair = (int(seq_gpu[i]), int(seq_gpu[i + 1]))
                transitions[pair] = transitions.get(pair, 0) + 1

            total = sum(transitions.values())
            probs = [c / total for c in transitions.values()]
        except Exception as e:
            logger.warning("GPU transition entropy failed, falling back to CPU: {0}".format(e))
            # Fallback to CPU
            transitions = Counter(zip(sequence, sequence[1:]))
            total = sum(transitions.values())
            probs = [c / total for c in transitions.values()]
    else:
        # CPU transition counting
        transitions = Counter(zip(sequence, sequence[1:]))
        total = sum(transitions.values())
        probs = [c / total for c in transitions.values()]

    log_fn = _log_lookup(base)
    return -sum(p * log_fn(p) for p in probs if p > 0)


def hamming_weight(bits: bytes) -> int:
    """Return number of '1' bits in *bits* (digest bytes)."""
    if USING_CUDA and cp.cuda.is_available() and len(bits) > 32:
        try:
            # GPU Hamming weight for large digests
            bits_array = cp.frombuffer(bits, dtype=cp.uint8)
            # Convert to binary and count 1s
            binary = cp.unpackbits(bits_array)
            return int(cp.sum(binary))
        except Exception as e:
            logger.warning("GPU Hamming weight failed, falling back to CPU: {0}".format(e))
            return sum(bin(b).count("1") for b in bits)
    else:
        # CPU Hamming weight
        return sum(bin(b).count("1") for b in bits)


def bit_entropy(digest: bytes) -> float:
    """Shannon entropy of 256-bit digest treated as 256 Bernoulli trials."""
    ones = hamming_weight(digest)
    zeros = len(digest) * 8 - ones
    if ones == 0 or zeros == 0:
        return 0.0
    p1 = ones / 256
    p0 = 1.0 - p1
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))


def normalized_entropy(values: Sequence[Number]) -> float:
    """Return entropy scaled to 0-1 by dividing by max possible entropy."""
    if not values:
        return 0.0
    unique = len(set(values))
    if unique <= 1:
        return 0.0
    h = shannon_entropy(values, base=2)
    h_max = math.log2(unique)
    return h / h_max if h_max else 0.0


def vector_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Calculate cosine similarity between two vectors using GPU/CPU."""
    if len(vec1) != len(vec2):
        return 0.0

    if USING_CUDA and cp.cuda.is_available():
        try:
            # GPU cosine similarity
            v1 = cp.asarray(vec1, dtype=cp.float32)
            v2 = cp.asarray(vec2, dtype=cp.float32)

            dot_product = cp.dot(v1, v2)
            norm1 = cp.linalg.norm(v1)
            norm2 = cp.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = float(dot_product / (norm1 * norm2))
            return max(-1.0, min(1.0, similarity))  # clamp to [-1, 1]
        except Exception as e:
            logger.warning("GPU vector similarity failed, falling back to CPU: {0}".format(e))
            # Fallback to CPU
            return _cpu_cosine_similarity(vec1, vec2)
    else:
        # CPU cosine similarity
        return _cpu_cosine_similarity(vec1, vec2)


def _cpu_cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """CPU implementation of cosine similarity."""
    try:
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(dot_product / (norm1 * norm2))
        return max(-1.0, min(1.0, similarity))  # clamp to [-1, 1]
    except Exception as e:
        logger.error("CPU cosine similarity failed: {0}".format(e))
        return 0.0


def hamming_distance(digest1: bytes, digest2: bytes) -> int:
    """Calculate Hamming distance between two digests using GPU/CPU."""
    if len(digest1) != len(digest2):
        return -1  # invalid

    if USING_CUDA and cp.cuda.is_available() and len(digest1) > 16:
        try:
            # GPU Hamming distance
            d1 = cp.frombuffer(digest1, dtype=cp.uint8)
            d2 = cp.frombuffer(digest2, dtype=cp.uint8)

            # XOR and count 1s
            xor_result = cp.bitwise_xor(d1, d2)
            binary = cp.unpackbits(xor_result)
            return int(cp.sum(binary))
        except Exception as e:
            logger.warning("GPU Hamming distance failed, falling back to CPU: {0}".format(e))
            return _cpu_hamming_distance(digest1, digest2)
    else:
        # CPU Hamming distance
        return _cpu_hamming_distance(digest1, digest2)


def _cpu_hamming_distance(digest1: bytes, digest2: bytes) -> int:
    """CPU implementation of Hamming distance."""
    distance = 0
    for b1, b2 in zip(digest1, digest2):
        xor_result = b1 ^ b2
        distance += bin(xor_result).count('1')
    return distance


# ---------------------------------------------------------------------------
# Internal util
# ---------------------------------------------------------------------------


def _log_lookup(base: int):
    if base == 2:
        return _log2
    elif base == math.e:
        return math.log  # natural
    else:
        inv = 1 / math.log(base)
        return lambda x: math.log(x) * inv  # type: ignore


def _log2(x: float) -> float:  # small inline faster than math.log2 for <Python3.11
    return math.log(x, 2)


# ---------------------------------------------------------------------------
# Performance monitoring
# ---------------------------------------------------------------------------


def get_backend_info() -> dict:
    """Get information about the current backend and performance."""
    info = {'backend': _backend, 'using_cuda': USING_CUDA, 'cuda_available': False}

    if USING_CUDA:
        try:
            info['cuda_available'] = cp.cuda.is_available()
            if info['cuda_available']:
                info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                info['gpu_memory'] = cp.cuda.runtime.memGetInfo()[1]  # total memory
        except Exception as e:
            logger.warning("Could not get CUDA info: {0}".format(e))

    return info


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔧 Entropy Math Backend Info:")
    print(get_backend_info())

    data = [1, 1, 1, 2, 2, 3]
    print("H", shannon_entropy(data))
    print("H_norm", normalized_entropy(data))
    print("Transition H", transition_entropy([0, 1, 0, 1, 1, 0]))
    digest = bytes.fromhex("a3" * 32)
    print("Bit entropy", bit_entropy(digest))

    # Test vector similarity
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]
    print("Cosine similarity:", vector_similarity(vec1, vec2))

    # Test Hamming distance
    d1 = bytes.fromhex("a3" * 16)
    d2 = bytes.fromhex("a3" * 16)
    print("Hamming distance:", hamming_distance(d1, d2))
