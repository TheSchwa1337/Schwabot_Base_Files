"""
Phase Map Entry and Transition Tests
=================================

Tests for phase state tracking and transitions.
"""

import pytest
from pathlib import Path
import json
import shutil
from datetime import datetime

from core.phase_engine.phase_map import PhaseMap, PhaseState


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory"""
    state_dir = tmp_path / "state"  # noqa: F841
    state_dir.mkdir()
    yield state_dir
    shutil.rmtree(state_dir)


@pytest.fixture
def temp_logs_dir(tmp_path):
    """Create temporary logs directory"""
    logs_dir = tmp_path / "logs"  # noqa: F841
    logs_dir.mkdir()
    yield logs_dir
    shutil.rmtree(logs_dir)


def test_phase_state_creation():
    """Test creation of phase state"""
    state = PhaseState(
        phase="STABLE",
        urgency=0.8,
        memory_coherence=0.9,
        hash_id="abc123",
        timestamp=datetime.now(),
        metrics={"profit_trend": 0.001, "stability": 0.8}
    )

    assert state.phase == "STABLE"
    assert state.urgency == 0.8
    assert state.memory_coherence == 0.9
    assert state.hash_id == "abc123"
    assert isinstance(state.timestamp, datetime)
    assert state.metrics == {"profit_trend": 0.001, "stability": 0.8}


def test_phase_state_serialization():
    """Test phase state serialization"""
    state = PhaseState(
        phase="STABLE",
        urgency=0.8,
        memory_coherence=0.9,
        hash_id="abc123",
        timestamp=datetime.now(),
        metrics={"profit_trend": 0.001, "stability": 0.8}
    )

    # Convert to dict
    state_dict = state.to_dict()
    assert isinstance(state_dict, dict)
    assert state_dict["phase"] == "STABLE"
    assert state_dict["urgency"] == 0.8
    assert state_dict["memory_coherence"] == 0.9
    assert state_dict["hash_id"] == "abc123"
    assert "timestamp" in state_dict
    assert state_dict["metrics"] == {"profit_trend": 0.001, "stability": 0.8}

    # Create from dict
    new_state = PhaseState.from_dict(state_dict)
    assert new_state.phase == state.phase
    assert new_state.urgency == state.urgency
    assert new_state.memory_coherence == state.memory_coherence
    assert new_state.hash_id == state.hash_id
    assert new_state.metrics == state.metrics


def test_phase_map_initialization(temp_state_dir, temp_logs_dir):
    """Test phase map initialization"""
    phase_map = PhaseMap()
    assert isinstance(phase_map.state_log, list)
    assert len(phase_map.state_log) == 0


def test_phase_update(temp_state_dir, temp_logs_dir):
    """Test phase state updates"""
    phase_map = PhaseMap()

    # Update phase
    phase_map.update_phase(
        phase="STABLE",
        urgency=0.8,
        coherence=0.9,
        hash_id="abc123",
        metrics={"profit_trend": 0.001, "stability": 0.8}
    )

    # Check state
    assert len(phase_map.state_log) == 1
    state = phase_map.state_log[0]
    assert state.phase == "STABLE"
    assert state.urgency == 0.8
    assert state.memory_coherence == 0.9
    assert state.hash_id == "abc123"
    assert state.metrics == {"profit_trend": 0.001, "stability": 0.8}


def test_phase_transition_logging(temp_state_dir, temp_logs_dir):
    """Test phase transition logging"""
    phase_map = PhaseMap()

    # First phase
    phase_map.update_phase(
        phase="STABLE",
        urgency=0.8,
        coherence=0.9,
        hash_id="abc123",
        metrics={"profit_trend": 0.001, "stability": 0.8}
    )

    # Second phase
    phase_map.update_phase(
        phase="SMART_MONEY",
        urgency=0.7,
        coherence=0.8,
        hash_id="def456",
        metrics={"profit_trend": 0.002, "stability": 0.7}
    )

    # Check transition log
    transition_path = Path("logs/phase_transitions.json")
    assert transition_path.exists()

    with open(transition_path, 'r') as f:
        transitions = json.load(f)

    assert len(transitions) == 2
    assert transitions[0]["to_phase"] == "STABLE"
    assert transitions[1]["from_phase"] == "STABLE"
    assert transitions[1]["to_phase"] == "SMART_MONEY"


def test_phase_statistics(temp_state_dir, temp_logs_dir):
    """Test phase statistics calculation"""
    phase_map = PhaseMap()

    # Add multiple phases
    phase_map.update_phase(
        phase="STABLE",
        urgency=0.8,
        coherence=0.9,
        hash_id="abc123",
        metrics={"profit_trend": 0.001, "stability": 0.8}
    )

    phase_map.update_phase(
        phase="SMART_MONEY",
        urgency=0.7,
        coherence=0.8,
        hash_id="def456",
        metrics={"profit_trend": 0.002, "stability": 0.7}
    )

    phase_map.update_phase(
        phase="STABLE",
        urgency=0.6,
        coherence=0.7,
        hash_id="ghi789",
        metrics={"profit_trend": 0.001, "stability": 0.6}
    )

    # Get statistics
    stats = phase_map.get_phase_statistics()

    assert stats["total_transitions"] == 3
    assert stats["phase_counts"]["STABLE"] == 2
    assert stats["phase_counts"]["SMART_MONEY"] == 1
    assert 0.6 <= stats["average_urgency"] <= 0.8
    assert 0.7 <= stats["average_coherence"] <= 0.9
    assert stats["last_transition"]["from_phase"] == "SMART_MONEY"
    assert stats["last_transition"]["to_phase"] == "STABLE"


def test_phase_metrics_history(temp_state_dir, temp_logs_dir):
    """Test phase metrics history"""
    phase_map = PhaseMap()

    # Add phases with metrics
    phase_map.update_phase(
        phase="STABLE",
        urgency=0.8,
        coherence=0.9,
        hash_id="abc123",
        metrics={"profit_trend": 0.001, "stability": 0.8}
    )

    phase_map.update_phase(
        phase="STABLE",
        urgency=0.7,
        coherence=0.8,
        hash_id="def456",
        metrics={"profit_trend": 0.002, "stability": 0.7}
    )

    # Get metrics history
    metrics = phase_map.get_phase_metrics("STABLE")

    assert "profit_trend" in metrics
    assert "stability" in metrics
    assert len(metrics["profit_trend"]) == 2
    assert len(metrics["stability"]) == 2
    assert metrics["profit_trend"][0] == 0.001
    assert metrics["profit_trend"][1] == 0.002
    assert metrics["stability"][0] == 0.8
    assert metrics["stability"][1] == 0.7


def test_invalid_phase_transition(temp_state_dir, temp_logs_dir):
    """Test invalid phase transition handling"""
    phase_map = PhaseMap()

    # Add initial phase
    phase_map.update_phase(
        phase="STABLE",
        urgency=0.8,
        coherence=0.9,
        hash_id="abc123",
        metrics={"profit_trend": 0.001, "stability": 0.8}
    )

    # Try invalid transition
    with pytest.raises(ValueError):
        phase_map.update_phase(
            phase="INVALID",
            urgency=0.8,
            coherence=0.9,
            hash_id="def456",
            metrics={"profit_trend": 0.001, "stability": 0.8}
        )