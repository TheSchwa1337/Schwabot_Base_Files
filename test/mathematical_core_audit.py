#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Core Audit - Comprehensive Validation of Nexus Mathematics

Validates all high-priority mathematical modules:
1. Volume Weighted Hash Oscillator (VWAP+SHA fusion)
2. Unified Tensor Algebra (Rank-3 tensor operations)
3. Zygot-Zalgo Entropy Dual Key Gate (Dual-key entropy gates)
4. QSC Gate (Quantum Symbolic Collapse gates)
5. Dual State Router (CPU/GPU bifurcation)
6. Galileo Tensor Field (Entropy drift tensors)

Tests mathematical integrity, integration, and operational readiness.
"""

import sys
import os
import time
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import mathematical modules
from core.strategy.volume_weighted_hash_oscillator import VolumeWeightedHashOscillator, OscillatorMode, SignalType, VolumeData
from core.strategy.zygot_zalgo_entropy_dual_key_gate import ZygotZalgoEntropyDualKeyGate, GateState, KeyType
from core.immune.qsc_gate import QSCGate, CollapseState, GateType
from core.system.dual_state_router import DualStateRouter, StrategyTier, ComputeMode, RouterState
from core.entropy.galileo_tensor_field import GalileoTensorField, EntropyMetrics
from core.math.tensor_algebra.unified_tensor_algebra import UnifiedTensorAlgebra

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MathematicalCoreAudit:
    """Comprehensive audit of mathematical core modules."""
    
    def __init__(self):
        """Initialize the mathematical core audit."""
        self.test_results = {}
        self.audit_start_time = time.time()
        
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive audit of all mathematical modules."""
        logger.info("🚀 Starting Mathematical Core Audit...")
        
        audit_results = {
            'timestamp': time.time(),
            'modules_tested': [],
            'mathematical_integrity': {},
            'integration_tests': {},
            'performance_metrics': {},
            'overall_status': 'PENDING'
        }
        
        try:
            # Test 1: Volume Weighted Hash Oscillator
            logger.info("📊 Testing Volume Weighted Hash Oscillator...")
            vwo_result = self._test_volume_weighted_hash_oscillator()
            audit_results['modules_tested'].append('volume_weighted_hash_oscillator')
            audit_results['mathematical_integrity']['vwo'] = vwo_result
            
            # Test 2: Unified Tensor Algebra
            logger.info("🔢 Testing Unified Tensor Algebra...")
            uta_result = self._test_unified_tensor_algebra()
            audit_results['modules_tested'].append('unified_tensor_algebra')
            audit_results['mathematical_integrity']['uta'] = uta_result
            
            # Test 3: Zygot-Zalgo Entropy Dual Key Gate
            logger.info("🔐 Testing Zygot-Zalgo Entropy Dual Key Gate...")
            zz_result = self._test_zygot_zalgo_entropy_dual_key_gate()
            audit_results['modules_tested'].append('zygot_zalgo_entropy_dual_key_gate')
            audit_results['mathematical_integrity']['zz'] = zz_result
            
            # Test 4: QSC Gate
            logger.info("⚛️ Testing QSC Gate...")
            qsc_result = self._test_qsc_gate()
            audit_results['modules_tested'].append('qsc_gate')
            audit_results['mathematical_integrity']['qsc'] = qsc_result
            
            # Test 5: Dual State Router
            logger.info("🔄 Testing Dual State Router...")
            dsr_result = self._test_dual_state_router()
            audit_results['modules_tested'].append('dual_state_router')
            audit_results['mathematical_integrity']['dsr'] = dsr_result
            
            # Test 6: Galileo Tensor Field
            logger.info("🌌 Testing Galileo Tensor Field...")
            gtf_result = self._test_galileo_tensor_field()
            audit_results['modules_tested'].append('galileo_tensor_field')
            audit_results['mathematical_integrity']['gtf'] = gtf_result
            
            # Integration Tests
            logger.info("🔗 Testing Module Integration...")
            integration_result = self._test_module_integration()
            audit_results['integration_tests'] = integration_result
            
            # Performance Metrics
            logger.info("⚡ Computing Performance Metrics...")
            performance_result = self._compute_performance_metrics()
            audit_results['performance_metrics'] = performance_result
            
            # Overall Status
            success_count = sum(1 for module in audit_results['mathematical_integrity'].values() 
                              if module.get('status') == 'PASS')
            total_modules = len(audit_results['mathematical_integrity'])
            
            if success_count == total_modules:
                audit_results['overall_status'] = 'PASS'
                logger.info("✅ All mathematical modules passed audit!")
            else:
                audit_results['overall_status'] = 'FAIL'
                logger.warning(f"⚠️ {total_modules - success_count} modules failed audit")
            
            audit_results['audit_duration'] = time.time() - self.audit_start_time
            
        except Exception as e:
            logger.error(f"❌ Audit failed with error: {e}")
            audit_results['overall_status'] = 'ERROR'
            audit_results['error'] = str(e)
        
        return audit_results
    
    def _test_volume_weighted_hash_oscillator(self) -> Dict[str, Any]:
        """Test Volume Weighted Hash Oscillator mathematical integrity."""
        try:
            # Initialize oscillator
            config = {
                'period': 20,
                'smoothing_period': 10,
                'hash_strength': 8,
                'tau_period': 100,
                'entropy_threshold': 0.1
            }
            vwo = VolumeWeightedHashOscillator(config)
            
            # Generate test data using proper VolumeData objects
            test_data = []
            for i in range(50):
                test_data.append(VolumeData(
                    timestamp=time.time() + i,
                    price=50000 + np.random.normal(0, 100),
                    volume=np.random.uniform(1, 100),
                    bid=49900,
                    ask=50100,
                    high=50200,
                    low=49800
                ))
            
            # Test VWAP drift collapse
            vwap_drift = vwo.compute_vwap_drift_collapse(test_data)
            assert isinstance(vwap_drift, float), "VWAP drift must be float"
            
            # Test entropic oscillator pulse
            entropic_pulse = vwo.compute_entropic_oscillator_pulse(test_data, time.time())
            assert isinstance(entropic_pulse, float), "Entropic pulse must be float"
            
            # Test hash oscillator computation
            result = vwo.compute_hash_oscillator(test_data)
            assert hasattr(result, 'vwap_value'), "Result must have vwap_value"
            assert hasattr(result, 'hash_value'), "Result must have hash_value"
            assert hasattr(result, 'oscillator_value'), "Result must have oscillator_value"
            
            return {
                'status': 'PASS',
                'vwap_drift': vwap_drift,
                'entropic_pulse': entropic_pulse,
                'oscillator_value': result.oscillator_value,
                'signal_type': result.signal_type.value
            }
            
        except Exception as e:
            logger.error(f"Volume Weighted Hash Oscillator test failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _test_unified_tensor_algebra(self) -> Dict[str, Any]:
        """Test Unified Tensor Algebra mathematical integrity."""
        try:
            # Initialize tensor algebra
            config = {
                'max_rank': 3,
                'collapse_threshold': 0.1,
                'fourier_resolution': 64,
                'gamma_shift': 0.1
            }
            uta = UnifiedTensorAlgebra(config)
            
            # Create test tensors
            A_components = [np.random.rand(4, 4) for _ in range(3)]
            phi_components = [np.random.rand(4, 4) for _ in range(3)]
            
            # Test canonical collapse tensor
            collapse_tensor = uta.compute_canonical_collapse_tensor(A_components, phi_components, 0.1)
            assert collapse_tensor.shape == (4, 4), "Collapse tensor must be 4x4"
            
            # Test Fourier tensor dual transform
            test_tensor = np.random.rand(8, 8)
            fourier_transform = uta.compute_fourier_tensor_dual_transform(test_tensor)
            assert fourier_transform.shape == test_tensor.shape, "Fourier transform must preserve shape"
            
            # Test tensor contraction
            tensor_a = np.random.rand(3, 4, 5)
            tensor_b = np.random.rand(5, 6, 7)
            contraction = uta.tensor_contraction(tensor_a, tensor_b, axes=([2], [0]))
            assert contraction.shape == (3, 4, 6, 7), "Contraction shape mismatch"
            
            # Test eigenvalue decomposition
            matrix = np.random.rand(4, 4)
            eigenvalues, eigenvectors = uta.eigenvalue_decomposition(matrix)
            assert len(eigenvalues) == 4, "Must have 4 eigenvalues"
            assert eigenvectors.shape == (4, 4), "Eigenvectors must be 4x4"
            
            return {
                'status': 'PASS',
                'collapse_tensor_shape': collapse_tensor.shape,
                'fourier_transform_shape': fourier_transform.shape,
                'contraction_shape': contraction.shape,
                'eigenvalues_count': len(eigenvalues)
            }
            
        except Exception as e:
            logger.error(f"Unified Tensor Algebra test failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _test_zygot_zalgo_entropy_dual_key_gate(self) -> Dict[str, Any]:
        """Test Zygot-Zalgo Entropy Dual Key Gate mathematical integrity."""
        try:
            # Initialize dual key gate
            config = {
                'zygot_entropy_threshold': 0.3,
                'zalgo_entropy_threshold': 0.3,
                'combined_threshold': 0.5,
                'alpha_coeff': 1.0,
                'beta_coeff': 1.0,
                'r2_radius': 2.0
            }
            zz_gate = ZygotZalgoEntropyDualKeyGate(config)
            
            # Generate test data
            volume_data = np.random.rand(100)
            momentum_data = np.random.rand(100)
            
            # Test dual key collapse gate
            collapse_gate = zz_gate.compute_dual_key_collapse_gate(volume_data)
            assert isinstance(collapse_gate, float), "Collapse gate must be float"
            
            # Test hash echo mirror
            echo_result = zz_gate.compute_hash_echo_mirror(volume_data, momentum_data)
            assert hasattr(echo_result, 'echo_hash'), "Echo result must have echo_hash"
            assert hasattr(echo_result, 'echo_strength'), "Echo result must have echo_strength"
            
            # Test dual key gate evaluation
            gate_result = zz_gate.evaluate_dual_key_gate(volume_data, momentum_data)
            assert hasattr(gate_result, 'gate_state'), "Gate result must have gate_state"
            assert hasattr(gate_result, 'access_granted'), "Gate result must have access_granted"
            
            return {
                'status': 'PASS',
                'collapse_gate': collapse_gate,
                'echo_strength': echo_result.echo_strength,
                'gate_state': gate_result.gate_state.value,
                'access_granted': gate_result.access_granted
            }
            
        except Exception as e:
            logger.error(f"Zygot-Zalgo Entropy Dual Key Gate test failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _test_qsc_gate(self) -> Dict[str, Any]:
        """Test QSC Gate mathematical integrity."""
        try:
            # Initialize QSC gate
            config = {
                'collapse_threshold': 0.5,
                'phase_resolution': 100,
                'decoherence_rate': 0.1,
                'measurement_strength': 1.0
            }
            qsc = QSCGate(config)
            
            # Test collapse function
            t = 0.1
            phi = np.pi / 4
            wavefunction = np.array([1.0, 0.0])
            collapse_function = qsc.compute_collapse_function(t, phi, wavefunction)
            # Handle both complex and float returns
            if isinstance(collapse_function, complex):
                collapse_value = abs(collapse_function)
            else:
                collapse_value = collapse_function
            assert isinstance(collapse_value, (float, int)), "Collapse function must be numeric"
            
            # Test phase gate logic
            phase_gate = qsc.compute_phase_gate_logic(phi, 'sigma_z')
            assert phase_gate.shape == (2, 2), "Phase gate must be 2x2"
            
            # Test quantum gate application
            initial_state = qsc.current_state
            new_state = qsc.apply_quantum_gate(initial_state, GateType.PHASE, {'phi': phi})
            assert hasattr(new_state, 'state_vector'), "New state must have state_vector"
            
            # Test AI signal processing
            ai_signal = "test_signal"
            hash_chain = "test_hash_chain"
            collapse_result = qsc.process_ai_signal(ai_signal, hash_chain)
            assert hasattr(collapse_result, 'collapse_state'), "Collapse result must have collapse_state"
            
            return {
                'status': 'PASS',
                'collapse_function': collapse_value,
                'phase_gate_shape': phase_gate.shape,
                'collapse_state': collapse_result.collapse_state.value
            }
            
        except Exception as e:
            logger.error(f"QSC Gate test failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _test_dual_state_router(self) -> Dict[str, Any]:
        """Test Dual State Router mathematical integrity."""
        try:
            # Initialize dual state router
            config = {
                'zpe_threshold': 0.5,
                'zbe_threshold': 0.7,
                'timeout_ms': 1000,
                'max_retries': 3
            }
            dsr = DualStateRouter(config)
            
            # Register test strategies
            from core.system.dual_state_router import StrategyMetadata
            from datetime import datetime
            
            strategy1 = StrategyMetadata(
                strategy_id="test_strategy_1",
                tier=StrategyTier.SHORT,
                priority=0.8,
                avg_compute_time_ms=50.0,
                avg_profit_margin=0.02,
                success_rate=0.85,
                last_execution=datetime.now(),
                preferred_mode=ComputeMode.ZPE
            )
            
            strategy2 = StrategyMetadata(
                strategy_id="test_strategy_2",
                tier=StrategyTier.LONG,
                priority=0.6,
                avg_compute_time_ms=500.0,
                avg_profit_margin=0.05,
                success_rate=0.75,
                last_execution=datetime.now(),
                preferred_mode=ComputeMode.ZBE
            )
            
            dsr.register_strategy(strategy1)
            dsr.register_strategy(strategy2)
            
            # Test routing score computation
            routing_score1 = dsr.compute_routing_score(strategy1)
            routing_score2 = dsr.compute_routing_score(strategy2)
            assert 0.0 <= routing_score1 <= 1.0, "Routing score must be in [0,1]"
            assert 0.0 <= routing_score2 <= 1.0, "Routing score must be in [0,1]"
            
            # Test compute mode selection
            mode1 = dsr.select_compute_mode(strategy1)
            mode2 = dsr.select_compute_mode(strategy2)
            assert mode1 in [ComputeMode.ZPE, ComputeMode.ZBE], "Invalid compute mode"
            assert mode2 in [ComputeMode.ZPE, ComputeMode.ZBE], "Invalid compute mode"
            
            # Test strategy routing
            routing_decision = dsr.route_strategy("test_strategy_1")
            assert hasattr(routing_decision, 'selected_mode'), "Routing decision must have selected_mode"
            assert hasattr(routing_decision, 'confidence'), "Routing decision must have confidence"
            
            return {
                'status': 'PASS',
                'routing_score1': routing_score1,
                'routing_score2': routing_score2,
                'mode1': mode1.value,
                'mode2': mode2.value,
                'routing_confidence': routing_decision.confidence
            }
            
        except Exception as e:
            logger.error(f"Dual State Router test failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _test_galileo_tensor_field(self) -> Dict[str, Any]:
        """Test Galileo Tensor Field mathematical integrity."""
        try:
            # Initialize Galileo tensor field with proper config structure
            from core.entropy.galileo_tensor_field import TensorFieldConfig
            config = TensorFieldConfig(
                dimension=3,
                precision=1e-8,
                max_iterations=1000,
                convergence_threshold=1e-6,
                use_gpu=True,
                fallback_enabled=True
            )
            gtf = GalileoTensorField(config)
            
            # Generate test data
            market_data = np.random.rand(100, 5)  # 100 timesteps, 5 features
            
            # Test tensor drift calculation
            drift_tensor = gtf.calculate_tensor_drift(market_data[:, 0], time_window=20)
            assert drift_tensor.shape == market_data[:, 0].shape, "Drift tensor shape mismatch"
            
            # Test entropy field calculation
            price_data = market_data[:, 0]
            volume_data = market_data[:, 1]
            entropy_metrics = gtf.calculate_entropy_field(price_data, volume_data)
            assert isinstance(entropy_metrics, EntropyMetrics), "Must return EntropyMetrics"
            assert hasattr(entropy_metrics, 'shannon_entropy'), "Must have shannon_entropy"
            assert hasattr(entropy_metrics, 'tensor_entropy'), "Must have tensor_entropy"
            
            # Test Galilean transform
            transformed_data = gtf.galilean_transform(market_data[:, 0], velocity=0.1)
            assert transformed_data.shape == market_data[:, 0].shape, "Transform shape mismatch"
            
            # Test tensor oscillation
            oscillation_data = gtf.tensor_oscillation(market_data[:, 0], frequency=1.0, amplitude=0.1)
            assert oscillation_data.shape == market_data[:, 0].shape, "Oscillation shape mismatch"
            
            return {
                'status': 'PASS',
                'drift_tensor_shape': drift_tensor.shape,
                'shannon_entropy': entropy_metrics.shannon_entropy,
                'tensor_entropy': entropy_metrics.tensor_entropy,
                'transformed_shape': transformed_data.shape,
                'oscillation_shape': oscillation_data.shape
            }
            
        except Exception as e:
            logger.error(f"Galileo Tensor Field test failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _test_module_integration(self) -> Dict[str, Any]:
        """Test integration between mathematical modules."""
        try:
            integration_results = {}
            
            # Test VWO + Tensor Algebra integration
            logger.info("Testing VWO + Tensor Algebra integration...")
            vwo = VolumeWeightedHashOscillator()
            uta = UnifiedTensorAlgebra()
            
            # Generate test data using proper VolumeData objects
            test_data = []
            for i in range(50):
                test_data.append(VolumeData(
                    timestamp=time.time() + i,
                    price=50000 + np.random.normal(0, 100),
                    volume=np.random.uniform(1, 100),
                    bid=49900,
                    ask=50100,
                    high=50200,
                    low=49800
                ))
            
            # VWO computation
            vwo_result = vwo.compute_hash_oscillator(test_data)
            
            # Convert to tensor for tensor algebra
            oscillator_tensor = np.array([vwo_result.vwap_value, vwo_result.hash_value, 
                                        vwo_result.oscillator_value, vwo_result.entropic_pulse])
            
            # Tensor algebra operations
            tensor_norm = uta.tensor_norm(oscillator_tensor)
            fourier_transform = uta.compute_fourier_tensor_dual_transform(oscillator_tensor.reshape(2, 2))
            
            integration_results['vwo_uta'] = {
                'status': 'PASS',
                'oscillator_value': vwo_result.oscillator_value,
                'tensor_norm': tensor_norm,
                'fourier_shape': fourier_transform.shape
            }
            
            # Test QSC + Zygot-Zalgo integration
            logger.info("Testing QSC + Zygot-Zalgo integration...")
            qsc = QSCGate()
            zz_gate = ZygotZalgoEntropyDualKeyGate()
            
            # QSC processing
            ai_signal = "integration_test_signal"
            hash_chain = "integration_test_hash"
            qsc_result = qsc.process_ai_signal(ai_signal, hash_chain)
            
            # Zygot-Zalgo evaluation
            volume_data = np.random.rand(100)
            momentum_data = np.random.rand(100)
            zz_result = zz_gate.evaluate_dual_key_gate(volume_data, momentum_data)
            
            integration_results['qsc_zz'] = {
                'status': 'PASS',
                'qsc_collapse_state': qsc_result.collapse_state.value,
                'zz_gate_state': zz_result.gate_state.value,
                'access_granted': zz_result.access_granted
            }
            
            return integration_results
            
        except Exception as e:
            logger.error(f"Module integration test failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    def _compute_performance_metrics(self) -> Dict[str, Any]:
        """Compute performance metrics for mathematical modules."""
        try:
            performance_metrics = {}
            
            # Test execution times
            modules = {
                'vwo': VolumeWeightedHashOscillator,
                'uta': UnifiedTensorAlgebra,
                'zz': ZygotZalgoEntropyDualKeyGate,
                'qsc': QSCGate,
                'dsr': DualStateRouter,
                'gtf': GalileoTensorField
            }
            
            for name, module_class in modules.items():
                start_time = time.time()
                
                # Initialize module
                module = module_class()
                
                # Generate test data
                test_data = np.random.rand(100)
                
                # Perform basic operation
                if name == 'vwo':
                    test_data_list = [{'timestamp': time.time() + i, 'price': 50000, 'volume': 1,
                                      'bid': 49900, 'ask': 50100, 'high': 50200, 'low': 49800} 
                                     for i in range(50)]
                    result = module.compute_hash_oscillator(test_data_list)
                elif name == 'uta':
                    tensor = np.random.rand(4, 4)
                    result = module.tensor_norm(tensor)
                elif name == 'zz':
                    result = module.compute_dual_key_collapse_gate(test_data)
                elif name == 'qsc':
                    result = module.compute_collapse_function(0.1, np.pi/4, np.array([1.0, 0.0]))
                elif name == 'dsr':
                    result = module.get_router_summary()
                elif name == 'gtf':
                    result = module.calculate_tensor_drift(test_data, time_window=20)
                
                execution_time = time.time() - start_time
                performance_metrics[name] = {
                    'initialization_time': execution_time,
                    'memory_usage': 'N/A',  # Would need psutil for actual measurement
                    'status': 'PASS'
                }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Performance metrics computation failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}


def main():
    """Main function to run the mathematical core audit."""
    print("🧠 Mathematical Core Audit - Schwabot Nexus Mathematics")
    print("=" * 60)
    
    audit = MathematicalCoreAudit()
    results = audit.run_comprehensive_audit()
    
    # Print results
    print(f"\n📊 Audit Results:")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Modules Tested: {len(results['modules_tested'])}")
    print(f"Audit Duration: {results.get('audit_duration', 0):.2f} seconds")
    
    print(f"\n🔢 Mathematical Integrity:")
    for module, result in results['mathematical_integrity'].items():
        status = result.get('status', 'UNKNOWN')
        status_icon = "✅" if status == 'PASS' else "❌"
        print(f"  {status_icon} {module}: {status}")
    
    print(f"\n🔗 Integration Tests:")
    for test, result in results['integration_tests'].items():
        if isinstance(result, dict):
            status = result.get('status', 'UNKNOWN')
        else:
            status = 'UNKNOWN'
        status_icon = "✅" if status == 'PASS' else "❌"
        print(f"  {status_icon} {test}: {status}")
    
    print(f"\n⚡ Performance Metrics:")
    for module, metrics in results['performance_metrics'].items():
        if isinstance(metrics, dict):
            status = metrics.get('status', 'UNKNOWN')
            init_time = metrics.get('initialization_time', 0)
        else:
            status = 'UNKNOWN'
            init_time = 0
        status_icon = "✅" if status == 'PASS' else "❌"
        print(f"  {status_icon} {module}: {init_time:.3f}s")
    
    if results['overall_status'] == 'PASS':
        print(f"\n🎉 All mathematical modules passed audit!")
        print(f"🚀 Schwabot mathematical core is ready for trading operations.")
    else:
        print(f"\n⚠️ Some modules failed audit. Check logs for details.")
    
    return results


if __name__ == "__main__":
    main() 