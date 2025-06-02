"""
Tests for Shift Profit Engine
===========================

Tests the functionality of the ShiftProfitEngine class, including profit trajectory optimization,
decay-aware navigation, and profit zone analysis.
"""

import numpy as np
import pytest
from datetime import datetime
from core.shift_profit_engine import ShiftProfitEngine, ProfitTrajectory

@pytest.fixture
def engine():
    """Create a ShiftProfitEngine instance for testing"""
    return ShiftProfitEngine(num_components=3, history_size=10)

@pytest.fixture
def sample_state():
    """Create a sample state tensor for testing"""
    # Create a tensor with clear quantum resonance
    tensor = np.zeros((10, 10, 10))
    for i in range(10):
        tensor[i, i, i] = 1.0
    return tensor

def test_initialization(engine):
    """Test engine initialization"""
    assert engine.num_components == 3
    assert engine.history_size == 10
    assert engine.resonance_threshold == 0.7
    assert len(engine.profit_zones) == 4
    assert len(engine.resonance_history['magnitude']) == 0

def test_profit_zones(engine):
    """Test profit zone initialization"""
    zones = engine.profit_zones
    
    assert 'quantum' in zones
    assert 'temporal' in zones
    assert 'entropic' in zones
    assert 'resonant' in zones
    
    for zone in zones.values():
        assert 'name' in zone
        assert 'description' in zone
        assert 'thresholds' in zone
        assert 'decay_factor' in zone
        assert 'magnitude' in zone['thresholds']
        assert 'coherence' in zone['thresholds']

def test_tensor_decomposition(engine, sample_state):
    """Test tensor decomposition methods"""
    cp_factors, tucker_factors = engine.decompose_tensor(sample_state)
    
    # Check CP decomposition
    assert len(cp_factors) == 3  # Three factors for 3D tensor
    for factor in cp_factors:
        assert factor.shape[0] == 10  # Each factor should have 10 elements
    
    # Check Tucker decomposition
    assert len(tucker_factors) == 2  # Core tensor and factors
    core, factors = tucker_factors
    assert core.shape == (3, 3, 3)  # Core tensor should be 3x3x3
    assert len(factors) == 3  # Three factor matrices
    for factor in factors:
        assert factor.shape == (10, 3)  # Each factor matrix should be 10x3

def test_quantum_resonance(engine, sample_state):
    """Test quantum resonance computation"""
    resonance = engine._compute_quantum_resonance(sample_state)
    
    assert 'magnitude' in resonance
    assert 'phase' in resonance
    assert 'coherence' in resonance
    
    assert 0 <= resonance['magnitude'] <= 1
    assert -np.pi <= resonance['phase'] <= np.pi
    assert 0 <= resonance['coherence'] <= 1

def test_state_similarity(engine, sample_state):
    """Test state similarity computation"""
    # Create two similar states
    state_a = sample_state.copy()
    state_b = sample_state.copy()
    state_b[0, 0, 0] = 0.5  # Small difference
    
    # Decompose states
    factors_a = engine.decompose_tensor(state_a)
    factors_b = engine.decompose_tensor(state_b)
    
    # Compute similarity
    similarity = engine._compute_state_similarity(
        factors_a,
        factors_b,
        engine.decay_params['quantum_gamma']
    )
    
    assert 0 <= similarity <= 1
    assert similarity > 0.5  # Should be high for similar states

def test_profit_gradient(engine, sample_state):
    """Test profit gradient computation"""
    # Create target state
    target_state = sample_state.copy()
    target_state[0, 0, 0] = 2.0  # Higher value for better profit
    
    # Compute gradient
    gradient = engine.compute_profit_gradient(sample_state, target_state)
    
    assert isinstance(gradient, float)
    assert gradient > 0  # Should be positive for better target state

def test_profit_trajectory_optimization(engine, sample_state):
    """Test profit trajectory optimization"""
    trajectory = engine.optimize_profit_trajectory(sample_state)
    
    assert isinstance(trajectory, ProfitTrajectory)
    assert trajectory.path_id.startswith("path_")
    assert isinstance(trajectory.start_time, datetime)
    assert trajectory.current_state.shape == sample_state.shape
    assert isinstance(trajectory.profit_gradient, float)
    assert isinstance(trajectory.resonance_state, dict)
    assert isinstance(trajectory.decay_factors, dict)
    assert isinstance(trajectory.confidence, float)
    assert isinstance(trajectory.tensor_factors, dict)

def test_trajectory_history(engine, sample_state):
    """Test trajectory history tracking"""
    # Optimize multiple trajectories
    for _ in range(5):
        engine.optimize_profit_trajectory(sample_state)
    
    history = engine.get_trajectory_history()
    assert len(history) <= engine.history_size
    assert all(isinstance(t, ProfitTrajectory) for t in history)

def test_resonance_history(engine, sample_state):
    """Test resonance history tracking"""
    # Optimize multiple trajectories
    for _ in range(5):
        engine.optimize_profit_trajectory(sample_state)
    
    history = engine.get_resonance_history()
    assert len(history['magnitude']) <= engine.history_size
    assert len(history['phase']) <= engine.history_size
    assert len(history['coherence']) <= engine.history_size

def test_profit_trajectory_analysis(engine, sample_state):
    """Test profit trajectory analysis"""
    # Optimize multiple trajectories
    for _ in range(5):
        engine.optimize_profit_trajectory(sample_state)
    
    analysis = engine.analyze_profit_trajectories()
    assert isinstance(analysis, dict)
    assert all(0 <= v <= 1 for v in analysis.values())
    assert abs(sum(analysis.values()) - 1.0) < 1e-10

def test_profit_shift_detection(engine, sample_state):
    """Test profit shift detection"""
    # First check with no history
    assert not engine.detect_profit_shift()
    
    # Create a profit shift scenario
    for _ in range(2):
        engine.optimize_profit_trajectory(sample_state)
    
    # Modify state to create a shift
    shift_state = sample_state.copy()
    shift_state[0, 0, 0] = 2.0  # Create a significant change
    engine.optimize_profit_trajectory(shift_state)
    
    # Now check for profit shift
    is_shift = engine.detect_profit_shift()
    assert isinstance(is_shift, bool)

def test_confidence_computation(engine, sample_state):
    """Test confidence computation for profit trajectories"""
    # First optimize some trajectories to build history
    for _ in range(3):
        engine.optimize_profit_trajectory(sample_state)
    
    confidence = engine._compute_confidence(sample_state)
    assert 0 <= confidence <= 1

def test_target_state_selection(engine, sample_state):
    """Test target state selection for profit optimization"""
    # First check with no history
    target = engine._get_target_state(sample_state)
    assert np.array_equal(target, sample_state)
    
    # Create some history with varying profits
    for i in range(3):
        state = sample_state.copy()
        state[0, 0, 0] = i + 1  # Increasing values
        engine.optimize_profit_trajectory(state)
    
    # Get target state
    target = engine._get_target_state(sample_state)
    assert target[0, 0, 0] == 3.0  # Should select highest profit state 