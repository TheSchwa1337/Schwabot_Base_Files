"""
Tests for Basket Phase Map
=========================

Tests the phase classification and urgency calculation logic.
"""

import pytest
from datetime import datetime
from core.phase_engine import BasketPhaseMap, PhaseMetrics, PhaseRegion

def test_phase_classification():
    """Test basic phase classification logic"""
    bpm = BasketPhaseMap()
    
    # Test STABLE phase
    metrics = {
        'profit_gradient': 0.005,
        'variance_of_returns': 0.8,
        'memory_coherence_score': 0.85,
        'entropy_rate': 0.2,
        'thermal_state': 0.3
    }
    
    bpm.update_phase_entry(
        basket_id="test_stable",
        sha_key="sha123",
        phase_depth=64,
        trust_score=0.9,
        current_metrics=metrics
    )
    
    phase, urgency = bpm.check_basket_swap_condition("test_stable")
    assert phase == "STABLE"
    assert urgency > 0
    assert urgency < 1.0

def test_unstable_phase():
    """Test detection of unstable phase"""
    bpm = BasketPhaseMap()
    
    # Test UNSTABLE phase
    metrics = {
        'profit_gradient': -0.01,
        'variance_of_returns': 0.2,
        'memory_coherence_score': 0.3,
        'entropy_rate': 0.8,
        'thermal_state': 0.9
    }
    
    bpm.update_phase_entry(
        basket_id="test_unstable",
        sha_key="sha456",
        phase_depth=8,
        trust_score=0.2,
        current_metrics=metrics
    )
    
    phase, urgency = bpm.check_basket_swap_condition("test_unstable")
    assert phase == "UNSTABLE"
    assert urgency > 0.5  # High urgency for unstable phase

def test_smart_money_phase():
    """Test detection of smart money phase"""
    bpm = BasketPhaseMap()
    
    # Test SMART_MONEY phase
    metrics = {
        'profit_gradient': 0.002,
        'variance_of_returns': 0.5,
        'memory_coherence_score': 0.7,
        'entropy_rate': 0.4,
        'thermal_state': 0.5
    }
    
    bpm.update_phase_entry(
        basket_id="test_smart",
        sha_key="sha789",
        phase_depth=42,
        trust_score=0.8,
        current_metrics=metrics
    )
    
    phase, urgency = bpm.check_basket_swap_condition("test_smart")
    assert phase == "SMART_MONEY"
    assert urgency > 0
    assert urgency < 0.5  # Moderate urgency for smart money phase

def test_phase_transitions():
    """Test tracking of phase transitions"""
    bpm = BasketPhaseMap()
    
    # Initial stable state
    metrics1 = {
        'profit_gradient': 0.005,
        'variance_of_returns': 0.8,
        'memory_coherence_score': 0.85,
        'entropy_rate': 0.2,
        'thermal_state': 0.3
    }
    
    bpm.update_phase_entry(
        basket_id="test_transition",
        sha_key="sha123",
        phase_depth=64,
        trust_score=0.9,
        current_metrics=metrics1
    )
    
    # Transition to unstable
    metrics2 = {
        'profit_gradient': -0.01,
        'variance_of_returns': 0.2,
        'memory_coherence_score': 0.3,
        'entropy_rate': 0.8,
        'thermal_state': 0.9
    }
    
    bpm.update_phase_entry(
        basket_id="test_transition",
        sha_key="sha456",
        phase_depth=8,
        trust_score=0.2,
        current_metrics=metrics2
    )
    
    transitions = bpm.get_phase_transitions("test_transition")
    assert len(transitions) > 0
    assert transitions[-1][0] == "STABLE"
    assert transitions[-1][1] == "UNSTABLE"

def test_phase_history():
    """Test phase history tracking"""
    bpm = BasketPhaseMap()
    
    metrics = {
        'profit_gradient': 0.005,
        'variance_of_returns': 0.8,
        'memory_coherence_score': 0.85,
        'entropy_rate': 0.2,
        'thermal_state': 0.3
    }
    
    # Add multiple entries
    for i in range(5):
        bpm.update_phase_entry(
            basket_id="test_history",
            sha_key=f"sha{i}",
            phase_depth=64,
            trust_score=0.9,
            current_metrics=metrics
        )
    
    history = bpm.get_phase_history("test_history")
    assert len(history) == 5
    assert all(isinstance(h, PhaseMetrics) for h in history) 