# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from __future__ import annotations

# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from dual_unicore_handler import DualUnicoreHandler
from pathlib import Path
from typing import Dict, List, Sequence
import argparse
import json
import sys

import numpy as np

from utils.math_utils import cosine_similarity
from utils.safe_print import safe_print


# Initialize Unicode handler
unicore = DualUnicoreHandler()

"""tools.overlay_registry_updater
Overlay Registry Updater
========================

CLI tool to update overlay registries by reweighting overlays via cosine
similarity to live vector input.

Usage:
    python tools / overlay_registry_updater.py --file memory_stack / aleph_overlays.json --vector 0.1 0.2 0.3 0.4
"""
"""
"""


def load_overlay_registry(file_path: Path) -> Dict[str, List[float]]:
    """Load overlay registry from JSON file."""


"""
"""
   if not file_path.exists():
        raise FileNotFoundError(f"Overlay registry not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

# Validate format
    for overlay_id, vector in data.items():
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise ValueError(f"Invalid vector format for overlay {overlay_id}")

    return data


def save_overlay_registry(file_path: Path, registry: Dict[str, List[float]]) -> None:
    """Save overlay registry to JSON file."""


"""
"""
# Create backup
   if file_path.exists():
        backup_path = file_path.with_suffix('.json.backup')
        file_path.rename(backup_path)
        safe_print(f"Backup created: {backup_path}")

    with open(file_path, 'w') as f:
        json.dump(registry, f, indent=4)

    safe_print(f"Registry saved: {file_path}")


def calculate_similarity_weights(

    registry: Dict[str, List[float]],
    live_vector: Sequence[float],
) -> Dict[str, float]:
    """Calculate cosine similarity weights for each overlay."""


"""
"""
   live_arr = np.asarray(live_vector, dtype = float)
    similarities = {}

    for overlay_id, vector in registry.items():
        overlay_arr = np.asarray(vector, dtype=float)

# Handle length mismatches by truncating to minimum length
        min_len = min(len(live_arr), len(overlay_arr))
        if min_len == 0:
            similarities[overlay_id] = 0.0
            continue

        try:
            sim = cosine_similarity(live_arr[:min_len], overlay_arr[:min_len])
            similarities[overlay_id] = float(sim)
        except Exception as e:
            safe_print(f"Warning: Failed to compute similarity for {overlay_id}: {e}")
            similarities[overlay_id] = 0.0

    return similarities


def reweight_overlays(

    registry: Dict[str, List[float]],
    similarities: Dict[str, float],
    strength: float = 0.1,
) -> Dict[str, List[float]]:
    """Reweight overlay vectors based on similarity to live vector."""


"""
"""
   updated_registry = {}

    for overlay_id, vector in registry.items():
        similarity = similarities.get(overlay_id, 0.0)

# Apply similarity - based adjustment
# Positive similarity increases weights, negative decreases them
        adjustment = 1.0 + (similarity * strength)

# Apply adjustment while keeping values in reasonable range
        updated_vector = []
        for value in vector:
            new_value = value * adjustment
# Clamp to [0, 1] range
            new_value = max(0.0, min(1.0, new_value))
            updated_vector.append(new_value)

        updated_registry[overlay_id] = updated_vector

    return updated_registry


def print_similarity_report(similarities: Dict[str, float]) -> None:
    """Print a report of similarity scores."""


"""
"""
   safe_print("\\n\\u1f4ca Similarity Report:")
    safe_print("=" * 50)

# Sort by similarity score (descending)
    sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse = True)

    for overlay_id, similarity in sorted_items:
    # Format similarity with color coding (conceptual)
        sim_str = f"{similarity:+.4f}"
        if similarity > 0.5:
            status = "\\u1f7e2 HIGH"
        elif similarity > 0.0:
            status = "\\u1f7e1 MEDIUM"
        elif similarity > -0.5:
            status = "\\u1f7e0 LOW"
        else:
            status = "\\u1f534 VERY LOW"

        safe_print(f"  {overlay_id:20} : {sim_str:8} ({status})")


def main() -> None:
    """Main CLI entry point."""


"""
"""
   parser = argparse.ArgumentParser(
        description="Update overlay registry with similarity - based reweighting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
"""
        """
Examples:
    python tools / overlay_registry_updater.py --file memory_stack / aleph_overlays.json --vector 0.1 0.2 0.3 0.4 0.5 0.6
    python tools / overlay_registry_updater.py --file overlays.json --vector 0.8 0.2 --strength 0.2 --dry - run
        """,
    )

    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to overlay registry JSON file"
    )

    parser.add_argument(
        "--vector",
        type=float,
        nargs="+",
        required=True,
        help="Live vector values for similarity comparison"
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.1,
        help="Reweighting strength factor (default: 0.1)"
    )

    parser.add_argument(
        "--dry - run",
        action="store_true",
        help="Show changes without saving"
    )

    parser.add_argument(
        "--report - only",
        action="store_true",
        help="Only show similarity report, don't reweight"
    )

    args = parser.parse_args()

    try:
        safe_print("\\u1f504 Overlay Registry Updater")
        safe_print("=" * 40)

# Load registry
        safe_print(f"Loading registry: {args.file}")
        registry = load_overlay_registry(args.file)
        safe_print(f"Found {len(registry)} overlays")

# Calculate similarities
        safe_print(f"Computing similarities with vector: {args.vector}")
        similarities = calculate_similarity_weights(registry, args.vector)

# Print similarity report
        print_similarity_report(similarities)

        if args.report_only:
            safe_print("\\n\\u1f4cb Report - only mode: no changes made")
            return

# Reweight overlays
        safe_print(f"\\nReweighting with strength: {args.strength}")
        updated_registry = reweight_overlays(registry, similarities, args.strength)

        if args.dry_run:
            safe_print("\\n\\u1f50d Dry - run mode: showing sample changes")
# Show first few changes as example
            for i, (overlay_id, original_vector) in enumerate(registry.items()):
                if i >= 3:  # Show first 3 only
                    break
                updated_vector = updated_registry[overlay_id]
                safe_print(f"  {overlay_id}:")
                safe_print(f"    Original: {original_vector[:4]}...")
                safe_print(f"    Updated:  {updated_vector[:4]}...")
        else:
    # Save updated registry
            save_overlay_registry(args.file, updated_registry)
            safe_print("\\u2705 Registry updated successfully")

    except Exception as e:
        safe_print(f"\\u274c Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
