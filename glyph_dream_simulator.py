#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glyph Dream Simulator
====================

A recursive glyph drift and echo correction visualizer for AI memory recursion.
Features:
- Drift animation (matrix drift and echo)
- Correction (anchor, normalization)
- Drift vector logging
- Entropic residue measurement
- Bitwise drift collapse
- Lattice collapse (final, memory)
- Ready for future PNG export
"""

import os
import time
from typing import List

import numpy as np

# Define the glyph set
GLYPHS = ["1", "i", "·", " ", "⊥"]


def get_initial_glyph_matrix():
    """Return the initial glyph matrix."""
    return [
        ["1", " ", "·", " ", "i", " ", "·", " ", "·", " ", "i"],
        ["i", " ", "·", " ", "·", " ", "1", " ", "·", " ", "i"],
        ["·", " ", "i", " ", "·", " ", "·", " ", "1", " ", "·"],
        ["·", " ", "·", " ", "·", " ", "i", " ", "·", " ", "1"],
    ]


# --- Drift Vector Logging ---


def log_drift_vectors(matrix: List[List[str]]):
    """Measure left/right offset per row for anchor glyphs."""
    drift_vectors = []
    for row in matrix:
        anchors = [i for i, g in enumerate(row) if g == "1"]
        if anchors:
            pivot = int(np.mean(anchors))
            drift = [i - pivot for i, g in enumerate(row) if g in ("1", "i", "·")]
            drift_vectors.append(drift)
        else:
            drift_vectors.append([0] * len(row))
    return drift_vectors


# --- Phase Drift Engine (Animation, Core) ---


def drift_matrix(matrix: List[List[str]], t: int) -> List[List[str]]:
    """Apply a phase drift to the matrix."""
    drifted = []
    for y, row in enumerate(matrix):
        phase = int(2 * np.sin(0.2 * t + y))
        if phase > 0:
            drifted.append([" "] * abs(phase) + row[: -abs(phase)])
        elif phase < 0:
            drifted.append(row[abs(phase):] + [" "] * abs(phase))
        else:
            drifted.append(row[:])
    return drifted


# --- Echo Correction (Anchor, Normalization) ---


def correct_glyph_row(glyph_row: List[str]) -> List[str]:
    """Aligns glyphs to the mean anchor position."""
    anchors = [i for i, g in enumerate(glyph_row) if g == "1"]
    if not anchors:
        return glyph_row
    pivot = int(np.mean(anchors))
    corrected = [" "] * len(glyph_row)
    for i, g in enumerate(glyph_row):
        new_pos = i - (anchors[0] - pivot)
        if 0 <= new_pos < len(glyph_row):
            corrected[new_pos] = g
    return corrected


def correct_glyph_matrix(matrix: List[List[str]]) -> List[List[str]]:
    """Correct the entire glyph matrix."""
    return [correct_glyph_row(row) for row in matrix]


# --- Entropic Residue Function ---


def entropic_residue(glyph_row: List[str]) -> float:
    """Measures instability (drift) in a glyph row."""
    values = [1 if g == "1" else 0 for g in glyph_row]
    mu = np.mean(values)
    return sum((v - mu) ** 2 for v in values)


# --- Bitwise Drift Collapse Correction ---


def bitwise_drift_collapse(matrix: List[List[str]]) -> List[List[str]]:
    """Collapse drift using XOR between consecutive rows."""
    collapsed = [matrix[0]]
    for prev, curr in zip(matrix, matrix[1:]):
        new_row = []
        for g1, g2 in zip(prev, curr):
            if g1 == g2:
                new_row.append(g1)
            else:
                new_row.append("·")  # Mark drift
        collapsed.append(new_row)
    return collapsed


# --- Lattice Collapse Function ---


def lattice_collapse(matrix_list: List[List[List[str]]]) -> List[List[str]]:
    """Collapse a list of matrices into a final stabilized lattice."""
    arr = np.array(matrix_list)
    final = []
    for y in range(arr.shape[1]):
        row = []
        for x in range(arr.shape[2]):
            vals, counts = np.unique(arr[:, y, x], return_counts=True)
            row.append(vals[np.argmax(counts)])
        final.append(row)
    return final


# --- Animation and Main Loop ---


def animate_glyph_matrix(
    matrix: List[List[str]], steps=40, delay=0.8, correct_every=8
):
    """Animate glyphs like falling matrices, tracking stabilizing points."""
    history = []
    for t in range(steps):
        os.system("cls" if os.name == "nt" else "clear")
        drifted = drift_matrix(matrix, t)
        if t % correct_every == 0 and t > 0:
            drifted = correct_glyph_matrix(drifted)
        history.append([row[:] for row in drifted])
        for row in drifted:
            print("".join(row))
        print("\nDrift vectors:", log_drift_vectors(drifted))
        print("Entropic residue:", [round(entropic_residue(row), 2) for row in drifted])
        time.sleep(delay)
    return history


# --- Main Entrypoint ---


def main():
    """Main entry point for the glyph dream simulator."""
    print("Glyph Dream Simulator: Recursive Drift Visualizer\n")
    matrix = get_initial_glyph_matrix()
    history = animate_glyph_matrix(matrix)
    print("\nBitwise Drift Collapse:")
    collapsed = bitwise_drift_collapse(history[-1])
    for row in collapsed:
        print("".join(row))
    print("\nLattice Collapse (final, memory):")
    final = lattice_collapse(history[-10:])
    for row in final:
        print("".join(row))
    print("\nSimulation complete. Ready for integration or export.")


if __name__ == "__main__":
    main()
