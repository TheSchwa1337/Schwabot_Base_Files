#!/usr/bin/env python3
"""
Matrix Mapper - Hash-to-matrix similarity routing logic
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional
from numpy.linalg import norm

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

def load_matrix_vectors(matrix_dir: str) -> Dict[str, Any]:
    """Load all matrix vectors from JSON files in a directory."""
    matrices = {}
    for fname in os.listdir(matrix_dir):
        if fname.endswith(".json"):
            with open(os.path.join(matrix_dir, fname), "r") as f:
                matrices[fname] = json.load(f)
    return matrices

def match_hash_to_matrix(hash_vec, matrix_dir, threshold=0.8) -> Optional[str]:
    """Match a hash vector to the closest matrix file above threshold."""
    matrices = load_matrix_vectors(matrix_dir)
    best_score = -1
    best_file = None
    for fname, vec in matrices.items():
        score = cosine_similarity(hash_vec, vec)
        if score > best_score and score >= threshold:
            best_score = score
            best_file = fname
    return best_file

__all__ = ["match_hash_to_matrix", "cosine_similarity"] 