"""
Tests for Plot Sign Engine
========================

Tests the functionality of the PlotSignEngine class \
    including tensor decomposition,
plot sign extraction, and regime shift detection.
"""

import numpy as np
import pytest
from core.plot_sign_engine import PlotSignEngine, PlotSign


@pytest.fixture
def engine():
    """Create a PlotSignEngine instance for testing"""
    return PlotSignEngine(num_components=3, history_size=10)


@pytest.fixture
def sample_tensor():
    """Create a sample market tensor for testing"""
    # Create a tensor with clear directional consensus
    tensor = np.zeros((10, 10, 10))
    for i in range(10):
        tensor[i, i, i] = 1.0
    return tensor


def test_initialization(engine):
    """Test engine initialization"""
    assert engine.num_components == 3
    assert engine.history_size == 10
    assert engine.resonance_threshold == 0.7
    assert len(engine.plot_sign_templates) == 5
    assert len(engine.resonance_history['magnitude']) == 0


def test_tensor_decomposition(engine, sample_tensor):
    """Test tensor decomposition methods"""
    cp_factors, tucker_factors = engine.decompose_tensor(sample_tensor)

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


def test_quantum_resonance(engine, sample_tensor):
    """Test quantum resonance computation"""
    resonance = engine._compute_quantum_resonance(sample_tensor)

    assert 'magnitude' in resonance
    assert 'phase' in resonance
    assert 'coherence' in resonance

    assert 0 <= resonance['magnitude'] <= 1
    assert -np.pi <= resonance['phase'] <= np.pi
    assert 0 <= resonance['coherence'] <= 1


def test_plot_sign_extraction(engine, sample_tensor):
    """Test plot sign extraction"""
    plot_signs = engine.extract_plot_signs(sample_tensor)

    assert isinstance(plot_signs, list)
    for sign in plot_signs:
        assert isinstance(sign, PlotSign)
        assert hasattr(sign, 'name')
        assert hasattr(sign, 'magnitude')
        assert hasattr(sign, 'phase')
        assert hasattr(sign, 'coherence')
        assert hasattr(sign, 'interpretation')
        assert hasattr(sign, 'confidence')
        assert hasattr(sign, 'tensor_factors')


def test_latent_factor_extraction(engine, sample_tensor):
    """Test latent factor extraction"""
    cp_factors, tucker_factors = engine.decompose_tensor(sample_tensor)
    latent_factors = engine._extract_latent_factors(cp_factors, tucker_factors)

    assert isinstance(latent_factors, np.ndarray)
    assert latent_factors.ndim == 1
    assert len(latent_factors) > 0


def test_confidence_computation(engine, sample_tensor):
    """Test confidence computation for plot signs"""
    # First extract some plot signs to build history
    engine.extract_plot_signs(sample_tensor)

    cp_factors, tucker_factors = engine.decompose_tensor(sample_tensor)
    latent_factors = engine._extract_latent_factors(cp_factors, tucker_factors)

    confidence = engine._compute_confidence(
        latent_factors,
        'directional_consensus'
    )
    assert 0 <= confidence <= 1


def test_plot_sign_history(engine, sample_tensor):
    """Test plot sign history tracking"""
    # Extract plot signs multiple times
    for _ in range(5):
        engine.extract_plot_signs(sample_tensor)

    history = engine.get_plot_sign_history()
    assert len(history) <= engine.history_size
    assert all(isinstance(signs, list) for signs in history)
    assert all(isinstance(sign, PlotSign)
               for signs in history for sign in signs)


def test_resonance_history(engine, sample_tensor):
    """Test resonance history tracking"""
    # Extract plot signs multiple times
    for _ in range(5):
        engine.extract_plot_signs(sample_tensor)

    history = engine.get_resonance_history()
    assert len(history['magnitude']) <= engine.history_size
    assert len(history['phase']) <= engine.history_size
    assert len(history['coherence']) <= engine.history_size


def test_plot_sign_sequence_analysis(engine, sample_tensor):
    """Test plot sign sequence analysis"""
    # Extract plot signs multiple times
    for _ in range(5):
        engine.extract_plot_signs(sample_tensor)

    analysis = engine.analyze_plot_sign_sequence()
    assert isinstance(analysis, dict)
    assert all(0 <= v <= 1 for v in analysis.values())
    assert abs(sum(analysis.values()) - 1.0) < 1e-10


def test_regime_shift_detection(engine, sample_tensor):
    """Test regime shift detection"""
    # First check with no history
    assert not engine.detect_regime_shift()

    # Create a phase transition scenario
    for _ in range(2):
        engine.extract_plot_signs(sample_tensor)

    # Modify tensor to create a phase transition
    transition_tensor = sample_tensor.copy()
    transition_tensor[0, 0, 0] = 2.0  # Create a significant change
    engine.extract_plot_signs(transition_tensor)

    # Now check for regime shift
    is_shift = engine.detect_regime_shift()
    assert isinstance(is_shift, bool)


def test_plot_sign_templates(engine):
    """Test plot sign template initialization"""
    templates = engine.plot_sign_templates

    assert 'directional_consensus' in templates
    assert 'pattern_conformance' in templates
    assert 'phase_transition' in templates
    assert 'strength_weakness' in templates
    assert 'event_localization' in templates

    for template in templates.values():
        assert 'name' in template
        assert 'interpretation' in template
        assert 'thresholds' in template
        assert 'magnitude' in template['thresholds']
        assert 'coherence' in template['thresholds']