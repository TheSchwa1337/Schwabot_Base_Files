from core.hardware_acceleration_manager import get_gpu_energy_ratio
from core.zpe_core import get_quantum_density

#!/usr/bin/env python3
"""Tests for ZPE and hardware backfill stubs."""



def test_zpe_density():
    assert get_quantum_density() == 1.0


def test_gpu_ratio():
    assert get_gpu_energy_ratio() == 1.0
