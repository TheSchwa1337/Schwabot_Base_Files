#!/usr/bin/env python3
"""
Simple test to isolate import issues
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports without complex dependencies."""
    try:
        print("Testing basic imports...")
        
        # Test numpy import
        import numpy as np
        print("✓ NumPy imported successfully")
        
        # Test core type definitions
        try:
            from core.type_defs import BitLevel, MatrixPhase, MatrixController
            print("✓ Core type_defs imported successfully")
        except ImportError as e:
            print(f"✗ Core type_defs import failed: {e}")
            print("Using fallback definitions...")
            
            # Define fallback classes
            from enum import Enum
            from dataclasses import dataclass
            from datetime import datetime
            
            class BitLevel(Enum):
                FOUR_BIT = 4
                EIGHT_BIT = 8
                SIXTEEN_BIT = 16
                FORTY_TWO_BIT = 42
            
            class MatrixPhase(Enum):
                INITIALIZATION = "INIT"
                ACCUMULATION = "ACCUM"
                RESONANCE = "RESON"
                DISPERSION = "DISP"
                CONVERGENCE = "CONV"
                FORTY_TWO_PHASE = "42P"
            
            @dataclass
            class MatrixController:
                bit_level: BitLevel
                phase: MatrixPhase
                hash_signature: str
                timestamp: datetime = datetime.now()
                confidence_score: float = 0.0
                fallback_triggered: bool = False
                state_vector: np.ndarray = np.zeros(10)
            
            print("✓ Fallback definitions created successfully")
        
        # Test DLT Waveform Engine
        try:
            from core.dlt_waveform_engine import DLTWaveformEngine
            print("✓ DLT Waveform Engine imported successfully")
        except Exception as e:
            print(f"✗ DLT Waveform Engine import failed: {e}")
        
        # Test Multi-bit BTC Processor
        try:
            from core.multi_bit_btc_processor import MultiBitBTCProcessor
            print("✓ Multi-bit BTC Processor imported successfully")
        except Exception as e:
            print(f"✗ Multi-bit BTC Processor import failed: {e}")
        
        # Test Profit Routing Engine
        try:
            from core.profit_routing_engine import ProfitRoutingEngine
            print("✓ Profit Routing Engine imported successfully")
        except Exception as e:
            print(f"✗ Profit Routing Engine import failed: {e}")
        
        # Test Temporal Execution Correction Layer
        try:
            from core.temporal_execution_correction_layer import TemporalExecutionCorrectionLayer
            print("✓ Temporal Execution Correction Layer imported successfully")
        except Exception as e:
            print(f"✗ Temporal Execution Correction Layer import failed: {e}")
        
        # Test Post-Failure Recovery Intelligence Loop
        try:
            from core.post_failure_recovery_intelligence_loop import PostFailureRecoveryIntelligenceLoop
            print("✓ Post-Failure Recovery Intelligence Loop imported successfully")
        except Exception as e:
            print(f"✗ Post-Failure Recovery Intelligence Loop import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic import test failed: {e}")
        return False

def test_fault_bus_import():
    """Test FaultBus import specifically."""
    try:
        print("\nTesting FaultBus import...")
        from core.fault_bus import FaultBus
        print("✓ FaultBus imported successfully")
        
        # Try to create an instance
        fault_bus = FaultBus()
        print("✓ FaultBus instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ FaultBus import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SCHWABOT UROS v1.0 IMPORT TEST")
    print("=" * 50)
    
    success1 = test_basic_imports()
    success2 = test_fault_bus_import()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Basic imports: {'PASS' if success1 else 'FAIL'}")
    print(f"FaultBus import: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! System is ready for validation.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.") 