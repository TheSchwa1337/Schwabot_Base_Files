import os
import pathlib
import sys

# Ensure the project root is on sys.path so that `import core.*` works even when
# the repository isn't installed as a package (CI / local checkout).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from core.dualistic_state_machine import DualisticStateMachine, StateType
from core.data_pipeline_visualizer import (
    DataPipelineVisualizer,
    DataCategory,
)
from core.drift_shell_engine import DriftShellEngine


def test_dualistic_state_machine_basic():
    """Ensure state machine updates scores and returns a valid snapshot."""
    dsm = DualisticStateMachine()

    # Update the scores with arbitrary but valid values
    dsm.update_scores(
        nibble_score=0.65,
        rittle_score=0.35,
        quantum_phase=0.25,
        entropy_level=0.2,
    )

    snapshot = dsm.get_current_snapshot()
    assert snapshot is not None
    assert snapshot.current_state in (StateType.ALEPH, StateType.ALIF)


def test_data_pipeline_visualizer_basic():
    """Verify that data units can be added and reported in the visualizer."""
    vis = DataPipelineVisualizer()
    vis.config["compression_enabled"] = False

    unit_id = vis.add_data_unit(DataCategory.MARKET_DATA, data_size=2048)
    assert unit_id, "Expected a non-empty unit_id when adding data"

    status = vis.get_pipeline_status()
    assert status["total_units"] >= 1, "Visualizer should report at least one unit"

    # Ensure resources are cleaned up even if the UI was never shown
    vis.close()


def test_drift_shell_engine_basic():
    """Smoke-test the drift-shell engine record/evaluate cycle."""
    engine = DriftShellEngine(confidence_threshold=0.0)

    ctx = {"volatility": 0.02, "trend_strength": 0.1}
    hash_val = engine.record_memory(
        tick_id="T-0001",
        price=30_000.0,
        volume=1.0,
        context_snapshot=ctx,
    )

    result = engine.evaluate_drift(
        current_price=30_000.5,
        current_volume=1.01,
        current_hash=hash_val,
    )

    # At least one valid recall should be recognized for identical hash
    assert result["valid_recalls"], "Expected at least one valid memory recall"
    assert result["total_memories"] == 1
