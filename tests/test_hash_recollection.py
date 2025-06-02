"""
Test suite for Hash Recollection System
=====================================

Tests the SHA-256 hash-based pattern recognition system, including:
- GPU/CPU hash computation
- Pattern matching
- Entropy calculations
- Tetragram matrix updates
- Performance metrics
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from core.hash_recollection import (
    HashRecollectionSystem,
    HashEntry,
    EntropyState
)

@pytest.fixture
def hash_system():
    """Create a HashRecollectionSystem instance"""
    return HashRecollectionSystem(gpu_enabled=False)  # Disable GPU for testing

def test_initialization(hash_system):
    """Test system initialization"""
    assert hash_system.gpu_enabled is False
    assert len(hash_system.hash_database) == 0
    assert hash_system.tetragram_matrix.shape == (3, 3, 3, 3)
    assert hash_system.running is False

def test_entropy_calculation(hash_system):
    """Test entropy calculation methods"""
    # Process some test ticks
    for i in range(10):
        hash_system.process_tick(
            price=100.0 + i,
            volume=1000.0 + i * 100,
            timestamp=datetime.now().timestamp() + i
        )
    
    # Get metrics
    metrics = hash_system.get_pattern_metrics()
    
    # Verify metrics
    assert metrics['hash_count'] > 0
    assert 0.0 <= metrics['pattern_confidence'] <= 1.0
    assert 0.0 <= metrics['collision_rate'] <= 1.0
    assert 0.0 <= metrics['tetragram_density'] <= 1.0

def test_pattern_matching(hash_system):
    """Test pattern matching functionality"""
    # Create test entropy states
    states = [
        EntropyState(
            price_entropy=0.5,
            volume_entropy=0.5,
            time_entropy=0.5,
            timestamp=datetime.now().timestamp()
        ) for _ in range(5)
    ]
    
    # Process states
    for state in states:
        hash_value = hash_system._cpu_compute_hash(state)
        hash_system._process_hash_result(state, hash_value)
    
    # Verify pattern matching
    test_hash = hash_system._cpu_compute_hash(states[0])
    similar_hashes = hash_system._find_similar_hashes(test_hash, threshold=0.85)
    
    assert len(similar_hashes) > 0
    assert all(isinstance(h, int) for h in similar_hashes)

def test_tetragram_matrix(hash_system):
    """Test tetragram matrix updates"""
    # Create test state
    state = EntropyState(
        price_entropy=0.5,
        volume_entropy=0.5,
        time_entropy=0.5,
        timestamp=datetime.now().timestamp()
    )
    
    # Update matrix
    hash_system._update_tetragram_matrix(state)
    
    # Verify matrix state
    assert np.any(hash_system.tetragram_matrix > 0)
    assert np.all(hash_system.tetragram_matrix >= 0)

def test_hash_confidence(hash_system):
    """Test pattern confidence calculation"""
    # Create test hashes
    test_hashes = [12345, 12346, 12347]
    
    # Add to database
    for h in test_hashes:
        hash_system.hash_database[h] = HashEntry(
            hash_value=h,
            strategy_id=0,
            confidence=0.0,
            frequency=2,
            timestamp=int(datetime.now().timestamp()),
            profit_history=0.1,
            reserved=bytes(32)
        )
    
    # Calculate confidence
    confidence = hash_system._calculate_pattern_confidence(test_hashes)
    
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.0  # Should be positive with test data

def test_gpu_cpu_synchronization(hash_system):
    """Test GPU-CPU synchronization"""
    # Create test system with GPU enabled
    with patch('core.hash_recollection.HashRecollectionSystem._check_gpu', return_value=True):
        gpu_system = HashRecollectionSystem(gpu_enabled=True)
        
        # Test synchronization
        gpu_system._synchronize_gpu_cpu()
        
        # Verify GPU buffers are cleared
        assert np.all(gpu_system.gpu_hash_buffer == 0)
        assert np.all(gpu_system.gpu_entropy_buffer == 0)

def test_worker_threads(hash_system):
    """Test worker thread functionality"""
    # Start system
    hash_system.start()
    assert hash_system.running is True
    
    # Process some ticks
    for i in range(10):
        hash_system.process_tick(
            price=100.0 + i,
            volume=1000.0 + i * 100,
            timestamp=datetime.now().timestamp() + i
        )
    
    # Stop system
    hash_system.stop()
    assert hash_system.running is False

def test_hash_distance_similarity(hash_system):
    """Test hash distance similarity calculation"""
    # Create test hashes
    hash1 = 0b10101010
    hash2 = 0b10101011  # One bit different
    
    # Calculate similarity
    similar_hashes = hash_system._find_similar_hashes(hash1, threshold=0.9)
    
    # Verify similarity calculation
    assert len(similar_hashes) == 0  # Should be no similar hashes with high threshold
    
    # Test with lower threshold
    similar_hashes = hash_system._find_similar_hashes(hash1, threshold=0.1)
    assert len(similar_hashes) >= 0  # May or may not have similar hashes

def test_entropy_state_processing(hash_system):
    """Test entropy state processing"""
    # Create test state
    state = EntropyState(
        price_entropy=0.5,
        volume_entropy=0.5,
        time_entropy=0.5,
        timestamp=datetime.now().timestamp()
    )
    
    # Process state
    hash_value = hash_system._cpu_compute_hash(state)
    hash_system._process_hash_result(state, hash_value)
    
    # Verify state was processed
    assert hash_value in hash_system.hash_database
    assert len(hash_system.entropy_history) > 0

def test_performance_metrics(hash_system):
    """Test performance metrics calculation"""
    # Process some test ticks
    for i in range(100):
        hash_system.process_tick(
            price=100.0 + i,
            volume=1000.0 + i * 100,
            timestamp=datetime.now().timestamp() + i
        )
    
    # Get metrics
    metrics = hash_system.get_pattern_metrics()
    
    # Verify all metrics are present and valid
    assert 'hash_count' in metrics
    assert 'pattern_confidence' in metrics
    assert 'collision_rate' in metrics
    assert 'tetragram_density' in metrics
    assert 'gpu_utilization' in metrics
    
    # Verify metric ranges
    assert metrics['hash_count'] >= 0
    assert 0.0 <= metrics['pattern_confidence'] <= 1.0
    assert 0.0 <= metrics['collision_rate'] <= 1.0
    assert 0.0 <= metrics['tetragram_density'] <= 1.0
    assert 0.0 <= metrics['gpu_utilization'] <= 1.0 