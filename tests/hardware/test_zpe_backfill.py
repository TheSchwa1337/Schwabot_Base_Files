#!/usr/bin/env python3
"""Tests for ZPE and hardware backfill stubs."""
from core.zpe_core import get_quantum_density
from core.hardware_acceleration_manager import get_gpu_energy_ratio

def test_zpe_density():
    assert get_quantum_density() == 1.0

def test_gpu_ratio():
    assert get_gpu_energy_ratio() == 1.0 