"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwafit Core Module

Schwafit: Recursive, shape-matching, volatility-aware prediction system.
Implements delta, normalization, cosine/DTW, entropy, memory, and fit decision.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging


logger = logging.getLogger(__name__)

class SchwafitCore:
"""Class for Schwabot trading functionality."""
"""
Schwafit: Recursive, shape-matching, volatility-aware prediction system.

Implements:
- Second-order difference computation
- Vector normalization (z-score, min-max)
- Cosine similarity and DTW distance
- Shannon entropy calculation
- Pattern matching and fit scoring
- Memory-based decision making
"""

def __init__(
self,
window: int = 64,
entropy_threshold: float = 2.5,
fit_threshold: float = 0.85,
config: Optional[Dict[str, Any]] = None,
):
"""
Initialize SchwafitCore.

Args:
window: Window size for pattern analysis
entropy_threshold: Threshold for entropy-based decisions
fit_threshold: Threshold for fit-based decisions
config: Configuration dictionary
"""
self.window = window
self.entropy_threshold = entropy_threshold
self.fit_threshold = fit_threshold
self.config = config or {}

# Memory storage for fit hashes, scores, profit, volatility
self.memory: List[Dict[str, Any]] = []

logger.info(f"✅ SchwafitCore initialized with window={window}, "
f"entropy_threshold={entropy_threshold}, fit_threshold={fit_threshold}")

@staticmethod
def delta2(series: List[float]) -> np.ndarray:
"""Compute second-order difference vector."""
arr = np.array(series)
return np.diff(arr, n=2)

@staticmethod
def normalize(vec: np.ndarray, method: str = "zscore") -> np.ndarray:
"""
Normalize vector using specified method.

Args:
vec: Input vector
method: Normalization method ('zscore' or 'minmax')

Returns:
Normalized vector
"""
if method == "zscore":
mu = np.mean(vec)
sigma = np.std(vec)
return (vec - mu) / sigma if sigma > 0 else vec * 0
elif method == "minmax":
minv, maxv = np.min(vec), np.max(vec)
return (vec - minv) / (maxv - minv) if maxv > minv else vec * 0
else:
return vec

@staticmethod
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
"""Compute cosine similarity between two vectors."""
if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
return 0.0
return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@staticmethod
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
"""
Compute Dynamic Time Warping distance between two vectors.

Args:
a: First vector
b: Second vector

Returns:
DTW distance
"""
n, m = len(a), len(b)
dtw_matrix = np.full((n + 1, m + 1), np.inf)
dtw_matrix[0, 0] = 0

for i in range(1, n + 1):
for j in range(1, m + 1):
cost = abs(a[i - 1] - b[j - 1])
dtw_matrix[i, j] = cost + min(
dtw_matrix[i - 1, j],
dtw_matrix[i, j - 1],
dtw_matrix[i - 1, j - 1]
)

return float(dtw_matrix[n, m])

@staticmethod
def entropy(vec: np.ndarray) -> float:
"""Compute Shannon entropy of normalized vector."""
absvec = np.abs(vec)
p = absvec / np.sum(absvec) if np.sum(absvec) > 0 else absvec
p = p[p > 0]
return float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0

@staticmethod
def alif_certainty(psi_t: np.ndarray, f_inverse_t: np.ndarray) -> float:
"""
ALIF (Asynchronous Logic Inversion Filter) certainty calculation.

ALIF_certainty = 1 - ||Ψ(t) - f⁻(t)|| / (||Ψ(t)|| + ||f⁻(t)||)

Where:
- Ψ(t) is the current state vector
- f⁻(t) is the inverted/reflected state vector
- ||·|| represents the L2 norm (Euclidean norm)

Args:
psi_t: Current state vector Ψ(t)
f_inverse_t: Inverted/reflected state vector f⁻(t)

Returns:
ALIF certainty score between 0 and 1 (higher = more certain)

Raises:
ValueError: If vectors have different shapes or are empty
"""
try:
if psi_t.shape != f_inverse_t.shape:
raise ValueError(f"Vector shapes must match: {psi_t.shape} vs {f_inverse_t.shape}")

if len(psi_t) == 0 or len(f_inverse_t) == 0:
raise ValueError("Vectors cannot be empty")

# Calculate L2 norms
psi_norm = np.linalg.norm(psi_t)
f_inverse_norm = np.linalg.norm(f_inverse_t)

# Calculate difference norm
diff_norm = np.linalg.norm(psi_t - f_inverse_t)

# Calculate denominator (avoid division by zero)
denominator = psi_norm + f_inverse_norm
if denominator == 0:
return 1.0  # Perfect certainty when both vectors are zero

# Calculate ALIF certainty
alif_certainty = 1.0 - (diff_norm / denominator)

# Ensure result is in [0, 1] range
return float(np.clip(alif_certainty, 0.0, 1.0))

except Exception as e:
logger.error(f"Error in ALIF certainty calculation: {e}")
raise

@staticmethod
def mir4x_reflection(cycle_phases: np.ndarray) -> float:
"""
MIR4X (Mirror-Based Four-Phase Cycle Reflector) reflection calculation.

MIR4X_reflection = 1 - (1/4) * Σ|Cᵢ - C₅₋ᵢ| / max(Cᵢ, C₅₋ᵢ)

Where:
- Cᵢ represents the i-th phase value in a 4-phase cycle
- The formula compares each phase with its mirror counterpart

Args:
cycle_phases: Array of 4 phase values [C₁, C₂, C₃, C₄]

Returns:
MIR4X reflection score between 0 and 1 (higher = more reflective)

Raises:
ValueError: If not exactly 4 phases provided
"""
try:
if len(cycle_phases) != 4:
raise ValueError(f"Exactly 4 phases required, got {len(cycle_phases)}")

# Calculate mirror differences: |C₁ - C₄|, |C₂ - C₃|
mirror_diff_1 = abs(cycle_phases[0] - cycle_phases[3])  # |C₁ - C₄|
mirror_diff_2 = abs(cycle_phases[1] - cycle_phases[2])  # |C₂ - C₃|

# Calculate max values for normalization
max_1 = max(cycle_phases[0], cycle_phases[3])
max_2 = max(cycle_phases[1], cycle_phases[2])

# Avoid division by zero
if max_1 == 0 and max_2 == 0:
return 1.0  # Perfect reflection when all phases are zero

# Calculate normalized differences
norm_diff_1 = mirror_diff_1 / max_1 if max_1 > 0 else 0.0
norm_diff_2 = mirror_diff_2 / max_2 if max_2 > 0 else 0.0

# Calculate MIR4X reflection
mir4x_reflection = 1.0 - (1.0 / 4.0) * (norm_diff_1 + norm_diff_2)

# Ensure result is in [0, 1] range
return float(np.clip(mir4x_reflection, 0.0, 1.0))

except Exception as e:
logger.error(f"Error in MIR4X reflection calculation: {e}")
raise

@staticmethod
def pr1sma_alignment(matrix_a: np.ndarray, matrix_b: np.ndarray, matrix_c: np.ndarray) -> float:
"""
PR1SMA (Phase Reflex Intelligence for Strategic Matrix Alignment) calculation.

S = (1/3) * (Corr(A,A⁻) + Corr(B,B⁻) + Corr(C,C⁻))

Where:
- A, B, C are input matrices
- A⁻, B⁻, C⁻ are their respective inverses/transposes
- Corr(X,Y) is the correlation coefficient between matrices X and Y

Args:
matrix_a: First matrix A
matrix_b: Second matrix B
matrix_c: Third matrix C

Returns:
PR1SMA alignment score between -1 and 1 (higher = better alignment)

Raises:
ValueError: If matrices have incompatible shapes
"""
try:
# Calculate inverse/transpose matrices
a_inverse = np.linalg.inv(matrix_a) if matrix_a.shape[0] == matrix_a.shape[1] else matrix_a.T
b_inverse = np.linalg.inv(matrix_b) if matrix_b.shape[0] == matrix_b.shape[1] else matrix_b.T
c_inverse = np.linalg.inv(matrix_c) if matrix_c.shape[0] == matrix_c.shape[1] else matrix_c.T

# Calculate correlation coefficients
def matrix_correlation(x: np.ndarray, y: np.ndarray) -> float:
"""Calculate correlation coefficient between two matrices."""
x_flat = x.flatten()
y_flat = y.flatten()

if len(x_flat) != len(y_flat):
# Pad shorter array with zeros
max_len = max(len(x_flat), len(y_flat))
x_flat = np.pad(x_flat, (0, max_len - len(x_flat)), 'constant')
y_flat = np.pad(y_flat, (0, max_len - len(y_flat)), 'constant')

# Calculate correlation
correlation = np.corrcoef(x_flat, y_flat)[0, 1]
return float(correlation) if not np.isnan(correlation) else 0.0

# Calculate individual correlations
corr_a = matrix_correlation(matrix_a, a_inverse)
corr_b = matrix_correlation(matrix_b, b_inverse)
corr_c = matrix_correlation(matrix_c, c_inverse)

# Calculate PR1SMA alignment score
pr1sma_score = (1.0 / 3.0) * (corr_a + corr_b + corr_c)

# Ensure result is in [-1, 1] range
return float(np.clip(pr1sma_score, -1.0, 1.0))

except Exception as e:
logger.error(f"Error in PR1SMA alignment calculation: {e}")
raise

@staticmethod
def delta_mirror_envelope(current_sigma: float, max_sigma: float) -> float:
"""
Δ-Mirror Envelope risk reflection calculation.

Risk_reflect = 1 - Δσ/σ_max

Where:
- Δσ is the change in volatility (sigma)
- σ_max is the maximum observed volatility

Args:
current_sigma: Current volatility measure
max_sigma: Maximum observed volatility

Returns:
Risk reflection score between 0 and 1 (higher = lower risk)

Raises:
ValueError: If max_sigma is zero or negative
"""
try:
if max_sigma <= 0:
raise ValueError(f"max_sigma must be positive, got {max_sigma}")

if current_sigma < 0:
logger.warning(f"Negative current_sigma detected: {current_sigma}, using absolute value")
current_sigma = abs(current_sigma)

# Calculate risk reflection
risk_reflect = 1.0 - (current_sigma / max_sigma)

# Ensure result is in [0, 1] range
return float(np.clip(risk_reflect, 0.0, 1.0))

except Exception as e:
logger.error(f"Error in Δ-mirror envelope calculation: {e}")
raise

@staticmethod
def z_matrix_reversal_logic(h_matrix: np.ndarray, z_matrix: np.ndarray) -> float:
"""
Z-matrix Reversal Logic certainty calculation.

Z_certainty = H·Z(H) / (||H||·||Z(H)||)

Where:
- H is the input matrix
- Z(H) is the Z-transform or reversal of H
- ||·|| represents the Frobenius norm

Args:
h_matrix: Input matrix H
z_matrix: Z-transformed/reversed matrix Z(H)

Returns:
Z-certainty score between 0 and 1 (higher = more certain)

Raises:
ValueError: If matrices have different shapes
"""
try:
if h_matrix.shape != z_matrix.shape:
raise ValueError(f"Matrix shapes must match: {h_matrix.shape} vs {z_matrix.shape}")

# Calculate Frobenius norms
h_norm = np.linalg.norm(h_matrix, ord='fro')
z_norm = np.linalg.norm(z_matrix, ord='fro')

# Calculate matrix product (element-wise multiplication)
h_z_product = np.sum(h_matrix * z_matrix)

# Calculate denominator (avoid division by zero)
denominator = h_norm * z_norm
if denominator == 0:
return 0.0  # Zero certainty when either matrix is zero

# Calculate Z-certainty
z_certainty = h_z_product / denominator

# Ensure result is in [0, 1] range
return float(np.clip(z_certainty, 0.0, 1.0))

except Exception as e:
logger.error(f"Error in Z-matrix reversal logic calculation: {e}")
raise

def mirror_analysis(
self,
price_series: List[float],
pattern_library: List[np.ndarray],
) -> Dict[str, Any]:
"""
Comprehensive mirror analysis using all mathematical mirror systems.

Integrates ALIF, MIR4X, PR1SMA, Δ-Mirror Envelope, and Z-matrix Reversal Logic.

Args:
price_series: Historical price data
pattern_library: Library of pattern vectors

Returns:
Dictionary containing all mirror analysis results
"""
try:
if len(price_series) < self.window + 2:
logger.warning(f"Insufficient data for mirror analysis: {len(price_series)} < {self.window + 2}")
return self._empty_mirror_result()

# Compute second-order difference and normalization
v = self.delta2(price_series[-(self.window + 2):])
v_norm = self.normalize(v)

# 1. ALIF Analysis
# Create inverted vector (negative of normalized vector)
v_inverse = -v_norm
alif_score = self.alif_certainty(v_norm, v_inverse)

# 2. MIR4X Analysis
# Extract 4-phase cycle from the vector
if len(v_norm) >= 4:
# Take 4 evenly spaced points from the vector
indices = np.linspace(0, len(v_norm) - 1, 4, dtype=int)
cycle_phases = v_norm[indices]
else:
# Pad with zeros if vector is too short
cycle_phases = np.pad(v_norm, (0, 4 - len(v_norm)), 'constant')
mir4x_score = self.mir4x_reflection(cycle_phases)

# 3. PR1SMA Analysis
# Create three matrices from the vector (reshape into 2D)
vector_len = len(v_norm)
matrix_size = int(np.sqrt(vector_len)) if vector_len > 0 else 1

# Ensure we have enough elements for a square matrix
if matrix_size * matrix_size < vector_len:
matrix_size += 1

# Pad vector to make it square
target_size = matrix_size * matrix_size
padded_vector = np.pad(v_norm, (0, max(0, target_size - vector_len)), 'constant')

# Create three matrices with different rotations/transformations
matrix_a = padded_vector.reshape(matrix_size, matrix_size)
matrix_b = np.roll(padded_vector, matrix_size).reshape(matrix_size, matrix_size)
matrix_c = np.roll(padded_vector, -matrix_size).reshape(matrix_size, matrix_size)

pr1sma_score = self.pr1sma_alignment(matrix_a, matrix_b, matrix_c)

# 4. Δ-Mirror Envelope Analysis
# Calculate current and max volatility
current_sigma = np.std(v_norm)
max_sigma = np.max(np.abs(v_norm)) if len(v_norm) > 0 else 1.0
delta_mirror_score = self.delta_mirror_envelope(current_sigma, max_sigma)

# 5. Z-matrix Reversal Logic
# Create Z-transform (reverse the matrix)
z_matrix = np.flipud(matrix_a)  # Vertical flip
z_matrix_score = self.z_matrix_reversal_logic(matrix_a, z_matrix)

# Calculate composite mirror score
mirror_scores = [alif_score, mir4x_score, pr1sma_score, delta_mirror_score, z_matrix_score]
composite_mirror_score = float(np.mean(mirror_scores))

# Determine mirror-based decision
mirror_decision = composite_mirror_score > 0.6  # Threshold for mirror-based decisions

logger.info(f"Mirror analysis: ALIF={alif_score:.3f}, MIR4X={mir4x_score:.3f}, "
f"PR1SMA={pr1sma_score:.3f}, Δ={delta_mirror_score:.3f}, "
f"Z={z_matrix_score:.3f}, Composite={composite_mirror_score:.3f}")

return {
"alif_certainty": alif_score,
"mir4x_reflection": mir4x_score,
"pr1sma_alignment": pr1sma_score,
"delta_mirror_envelope": delta_mirror_score,
"z_matrix_certainty": z_matrix_score,
"composite_mirror_score": composite_mirror_score,
"mirror_decision": mirror_decision,
"normalized_vector": v_norm,
"cycle_phases": cycle_phases.tolist(),
"matrix_a": matrix_a.tolist(),
"matrix_b": matrix_b.tolist(),
"matrix_c": matrix_c.tolist(),
}

except Exception as e:
logger.error(f"Error in mirror analysis: {e}")
return self._empty_mirror_result()

def _empty_mirror_result(self) -> Dict[str, Any]:
"""Return empty mirror analysis result when insufficient data."""
return {
"alif_certainty": 0.0,
"mir4x_reflection": 0.0,
"pr1sma_alignment": 0.0,
"delta_mirror_envelope": 0.0,
"z_matrix_certainty": 0.0,
"composite_mirror_score": 0.0,
"mirror_decision": False,
"normalized_vector": [],
"cycle_phases": [0.0, 0.0, 0.0, 0.0],
"matrix_a": [[0.0]],
"matrix_b": [[0.0]],
"matrix_c": [[0.0]],
"error": "Insufficient data for mirror analysis",
}

def fit_vector(
self,
price_series: List[float],
pattern_library: List[np.ndarray],
profit_scores: List[float],
) -> Dict[str, Any]:
"""
Main fit function with integrated mirror analysis.

Returns dict with fit score, entropy, best matches, mirror analysis, and decision.

Args:
price_series: Historical price data
pattern_library: Library of pattern vectors
profit_scores: Profit scores corresponding to patterns

Returns:
Dictionary containing fit analysis results with mirror analysis
"""
if len(price_series) < self.window + 2:
logger.warning(f"Insufficient data: {len(price_series)} < {self.window + 2}")
return self._empty_fit_result()

# Compute second-order difference
v = self.delta2(price_series[-(self.window + 2):])
v_norm = self.normalize(v)
ent = self.entropy(v_norm)
v_hash = hashlib.sha256(v_norm.tobytes()).hexdigest()

# Cosine similarity to all patterns
sims = [self.cosine_similarity(v_norm, s) for s in pattern_library]
top_indices = np.argsort(sims)[-3:][:-1]  # Top 3 matches
top_scores = [sims[i] for i in top_indices]
top_profits = [profit_scores[i] for i in top_indices]

# Compute weighted fit score
fit_score = float(np.average([s * p for s, p in zip(top_scores, top_profits)])
if top_scores else 0.0)

# Perform mirror analysis
mirror_results = self.mirror_analysis(price_series, pattern_library)

# Enhanced decision logic incorporating mirror analysis
traditional_decision = fit_score > self.fit_threshold and ent < self.entropy_threshold
mirror_decision = mirror_results.get("mirror_decision", False)

# Combined decision: both traditional and mirror analysis must agree
combined_decision = traditional_decision and mirror_decision

# Calculate enhanced fit score incorporating mirror analysis
mirror_weight = 0.3  # Weight for mirror analysis in final score
enhanced_fit_score = (1 - mirror_weight) * fit_score + mirror_weight * mirror_results.get("composite_mirror_score", 0.0)

# Memory update with enhanced information
memory_entry = {
"hash": v_hash,
"fit_score": fit_score,
"enhanced_fit_score": enhanced_fit_score,
"entropy": ent,
"top_scores": top_scores,
"top_profits": top_profits,
"traditional_decision": traditional_decision,
"mirror_decision": mirror_decision,
"combined_decision": combined_decision,
"mirror_analysis": mirror_results,
"timestamp": len(self.memory),
}
self.memory.append(memory_entry)

logger.info(f"Enhanced Schwafit fit: hash={v_hash[:8]}, fit_score={fit_score:.3f}, "
f"enhanced_score={enhanced_fit_score:.3f}, entropy={ent:.3f}, "
f"mirror_score={mirror_results.get('composite_mirror_score', 0.0):.3f}, "
f"decision={combined_decision}")

return {
"fit_score": fit_score,
"enhanced_fit_score": enhanced_fit_score,
"entropy": ent,
"top_scores": top_scores,
"top_profits": top_profits,
"traditional_decision": traditional_decision,
"mirror_decision": mirror_decision,
"combined_decision": combined_decision,
"mirror_analysis": mirror_results,
"hash": v_hash,
"normalized_vector": v_norm,
}

def fit_vector_dtw(
self,
price_series: List[float],
pattern_library: List[np.ndarray],
profit_scores: List[float],
) -> Dict[str, Any]:
"""
Fit function using DTW distance instead of cosine similarity.

Args:
price_series: Historical price data
pattern_library: Library of pattern vectors
profit_scores: Profit scores corresponding to patterns

Returns:
Dictionary containing DTW-based fit analysis results
"""
if len(price_series) < self.window + 2:
return self._empty_fit_result()

v = self.delta2(price_series[-(self.window + 2):])
v_norm = self.normalize(v)
ent = self.entropy(v_norm)
v_hash = hashlib.sha256(v_norm.tobytes()).hexdigest()

# DTW distance to all patterns (lower is better)
distances = [self.dtw_distance(v_norm, s) for s in pattern_library]
top_indices = np.argsort(distances)[:3]  # Top 3 matches (lowest distance)
top_distances = [distances[i] for i in top_indices]
top_profits = [profit_scores[i] for i in top_indices]

# Convert distances to similarity scores (inverse relationship)
max_dist = max(top_distances) if top_distances else 1.0
top_scores = [1.0 - (d / max_dist) for d in top_distances]

# Compute weighted fit score
fit_score = float(np.average([s * p for s, p in zip(top_scores, top_profits)])
if top_scores else 0.0)

decision = fit_score > self.fit_threshold and ent < self.entropy_threshold

return {
"fit_score": fit_score,
"entropy": ent,
"top_scores": top_scores,
"top_profits": top_profits,
"top_distances": top_distances,
"decision": decision,
"hash": v_hash,
"method": "dtw",
}

def get_fit_memory(self) -> List[Dict[str, Any]]:
"""Return memory of all fits."""
return self.memory

def get_last_fit(self) -> Optional[Dict[str, Any]]:
"""Return the most recent fit result."""
return self.memory[-1] if self.memory else None

def clear_memory(self) -> None:
"""Clear the fit memory."""
self.memory.clear()
logger.info("✅ SchwafitCore memory cleared")

def get_memory_stats(self) -> Dict[str, Any]:
"""Get statistics about the fit memory."""
if not self.memory:
return {"count": 0, "avg_fit_score": 0.0, "avg_entropy": 0.0}

fit_scores = [entry["fit_score"] for entry in self.memory]
entropies = [entry["entropy"] for entry in self.memory]

return {
"count": len(self.memory),
"avg_fit_score": float(np.mean(fit_scores)),
"avg_entropy": float(np.mean(entropies)),
"max_fit_score": float(np.max(fit_scores)),
"min_entropy": float(np.min(entropies)),
}

def _empty_fit_result(self) -> Dict[str, Any]:
"""Return empty fit result when insufficient data."""
return {
"fit_score": 0.0,
"enhanced_fit_score": 0.0,
"entropy": 0.0,
"top_scores": [],
"top_profits": [],
"traditional_decision": False,
"mirror_decision": False,
"combined_decision": False,
"mirror_analysis": self._empty_mirror_result(),
"hash": "",
"error": "Insufficient data",
}


# Factory function for creating SchwafitCore instances
def create_schwafit_core(
window: int = 64,
entropy_threshold: float = 2.5,
fit_threshold: float = 0.85,
config: Optional[Dict[str, Any]] = None,
) -> SchwafitCore:
"""
Factory function to create a SchwafitCore instance.

Args:
window: Window size for pattern analysis
entropy_threshold: Threshold for entropy-based decisions
fit_threshold: Threshold for fit-based decisions
config: Configuration dictionary

Returns:
Initialized SchwafitCore instance
"""
return SchwafitCore(
window=window,
entropy_threshold=entropy_threshold,
fit_threshold=fit_threshold,
config=config,
)
