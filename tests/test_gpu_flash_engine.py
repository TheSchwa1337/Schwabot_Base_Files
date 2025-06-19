from unittest.mock import Mock
from unittest.mock import patch

"""
GPU Flash Engine v0.5 Test Suite
================================

Comprehensive tests for the quantum-coherent GPU flash orchestrator.
Tests event-driven architecture, configuration management, \
    and fractal integration.
"""

from core.bus_core import BusEvent, BusCore  # noqa: F401
from core.gpu_flash_engine import GPUFlashEngine, FlashState, GPUFlashConfig  # noqa: F401
import unittest  # noqa: F401
import numpy as np  # noqa: F401
import json  # noqa: F401
import time  # noqa: F401
from pathlib import Path  # noqa: F401
import tempfile  # noqa: F401
import os  # noqa: F401

# Import the modules we're testing
import sys  # noqa: F401
sys.path.append(str(Path(__file__).parent.parent))


class TestGPUFlashConfig(unittest.TestCase):
    """Test GPU Flash configuration management"""

    def test_default_config_creation(self):
        """Test that default configuration is created properly"""
        config = GPUFlashConfig()

        self.assertEqual(config.cooldown_period, 0.1)
        self.assertEqual(config.binding_energy_default, 7.5)
        self.assertIsInstance(config.risk_thresholds, dict)
        self.assertIn('critical', config.risk_thresholds)
        self.assertEqual(config.risk_thresholds['critical'], 0.9)

    def test_custom_config_creation(self):
        """Test creation with custom parameters"""
        config = GPUFlashConfig(
            cooldown_period=0.2,
            binding_energy_default=10.0,
            risk_thresholds={'critical': 0.8, 'high': 0.6}
        )

        self.assertEqual(config.cooldown_period, 0.2)
        self.assertEqual(config.binding_energy_default, 10.0)
        self.assertEqual(config.risk_thresholds['critical'], 0.8)


class TestFlashState(unittest.TestCase):
    """Test FlashState dataclass"""

    def test_flash_state_creation(self):
        """Test creation of FlashState with all fields"""
        state = FlashState(
            timestamp=time.time(),
            z_score=1.5,
            phase_angle=np.pi,
            entropy_class="stable",
            matrix_state="matrix_safe",
            is_safe=True,
            risk_entropy=0.3,
            fractal_depth=2,
            event_id="test_123",
            coherence_score=0.8,
            binding_energy=8.0
        )

        self.assertTrue(state.is_safe)
        self.assertEqual(state.entropy_class, "stable")
        self.assertEqual(state.fractal_depth, 2)
        self.assertEqual(state.coherence_score, 0.8)


class TestGPUFlashEngine(unittest.TestCase):
    """Test the main GPU Flash Engine functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for state files
        self.temp_dir = tempfile.mkdtemp()

        # Mock configuration
        self.mock_config = {
            'cooldown_period': 0.1,
            'binding_energy_default': 7.5,
            'risk_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            },
            'max_cascade_memory': 100,
            'max_history_size': 1000,
            'enable_fractal_corrections': True
        }

        # Create mock bus core
        self.mock_bus = MagicMock(spec=BusCore)

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil  # noqa: F401
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_engine_initialization(self, mock_load_config):
        """Test GPU Flash Engine initialization"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        self.assertIsNotNone(engine.math_core)
        self.assertIsNotNone(engine.config)
        self.assertEqual(len(engine.flash_history), 0)
        self.assertEqual(engine.fractal_depth, 0)

        # Verify event handlers were registered
        expected_events = [
            'entropy.update',
            'phase.drift',
            'risk.cascade',
            'system.shutdown',
            'flash.request'
        ]

        for event in expected_events:
            self.mock_bus.register_handler.assert_any_call(
                event,
                unittest.mock.ANY
            )

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_flash_permission_safe_case(self, mock_load_config):
        """Test flash permission check for safe conditions"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Mock the math core methods
        engine.math_core.classify_entropy_shell = MagicMock(
            return_value='stable')
        engine.math_core.classify_phase_shell = MagicMock()
        engine.math_core.classify_phase_shell.return_value.shell_type = 'drift_positive'
        engine.math_core.classify_phase_shell.return_value.stability = 0.8
        engine.math_core.compute_matrix_stability = MagicMock(
            return_value='matrix_safe')

        is_permitted, reason, state = engine.check_flash_permission(
            z_score=1.0,
            phase_angle=np.pi + 0.1
        )

        self.assertTrue(is_permitted)
        self.assertEqual(reason, 'matrix_safe')
        self.assertIsInstance(state, FlashState)
        self.assertTrue(state.is_safe)
        self.assertEqual(len(engine.flash_history), 1)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_flash_permission_critical_entropy(self, mock_load_config):
        """Test flash permission blocked due to critical entropy"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)
        engine.math_core.classify_entropy_shell = MagicMock(
            return_value='critical_bloom')

        is_permitted, reason, state = engine.check_flash_permission(
            z_score=3.0,
            phase_angle=np.pi
        )

        self.assertFalse(is_permitted)
        self.assertEqual(reason, 'critical_entropy')
        self.assertFalse(state.is_safe)

        # Verify anomaly was published
        self.mock_bus.dispatch_event.assert_called()

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_cooldown_period_enforcement(self, mock_load_config):
        """Test that cooldown period is enforced"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)
        engine.math_core.classify_entropy_shell = MagicMock(
            return_value='stable')
        engine.math_core.classify_phase_shell = MagicMock()
        engine.math_core.classify_phase_shell.return_value.shell_type = 'drift_positive'
        engine.math_core.classify_phase_shell.return_value.stability = 0.8
        engine.math_core.compute_matrix_stability = MagicMock(
            return_value='matrix_safe')

        # First flash should succeed
        is_permitted1, _, _ = engine.check_flash_permission(1.0, np.pi)
        self.assertTrue(is_permitted1)

        # Immediate second flash should be blocked by cooldown
        is_permitted2, reason2, _ = engine.check_flash_permission(1.0, np.pi)
        self.assertFalse(is_permitted2)
        self.assertEqual(reason2, 'cooldown_period')

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_risk_entropy_calculation(self, mock_load_config):
        """Test risk entropy calculation"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Test with known values
        risk = engine._calculate_risk_entropy(
            z_score=2.0,
            phase_angle=np.pi / 2,
            phase_stability=0.5
        )

        self.assertGreater(risk, 0)
        self.assertLess(risk, 1)
        self.assertIsInstance(risk, float)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_dynamic_binding_energy_calculation(self, mock_load_config):
        """Test dynamic binding energy calculation with context"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Test with no context
        energy1 = engine._calculate_dynamic_binding_energy(None)
        self.assertEqual(energy1, 7.5)  # Default value

        # Test with high volatility context
        context = {'high_volatility': True}
        energy2 = engine._calculate_dynamic_binding_energy(context)
        self.assertGreater(energy2, energy1)

        # Test with multiple context factors
        context = {'high_volatility': True, 'news_event': True}
        energy3 = engine._calculate_dynamic_binding_energy(context)
        self.assertGreater(energy3, energy2)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_event_handling_entropy_update(self, mock_load_config):
        """Test handling of entropy update events"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Simulate entropy update event
        event = BusEvent(
            type='entropy.update',
            data={'z_score': 2.5},
            timestamp=time.time()
        )

        initial_cascade_length = len(engine.entropy_cascade)
        engine._handle_entropy_update(event)

        self.assertEqual(
            len(engine.entropy_cascade),
            initial_cascade_length + 1
        )
        self.assertEqual(engine.entropy_cascade[-1], 2.5)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_event_handling_phase_drift(self, mock_load_config):
        """Test handling of phase drift events"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Simulate phase drift event
        event = BusEvent(
            type='phase.drift',
            data={'phase_angle': np.pi + 0.1},
            timestamp=time.time()
        )

        initial_memory_length = len(engine.phase_memory)
        engine._handle_phase_drift(event)

        self.assertEqual(len(engine.phase_memory), initial_memory_length + 1)
        self.assertEqual(engine.phase_memory[-1], np.pi + 0.1)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_phase_resonance_detection(self, mock_load_config):
        """Test phase resonance detection"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Add similar phases to trigger resonance
        similar_phases = [np.pi, np.pi + 0.001, np.pi + 0.002]
        engine.phase_memory.extend(similar_phases)

        engine._check_phase_resonance()

        # Should have dispatched a phase.resonance event
        self.mock_bus.dispatch_event.assert_called()

        # Check that the dispatched event is correct
        call_args = self.mock_bus.dispatch_event.call_args
        event = call_args[0][0]
        self.assertEqual(event.type, 'phase.resonance')
        self.assertIn('coherence', event.data)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_flash_request_handling(self, mock_load_config):
        """Test handling of flash request events"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)
        engine.math_core.classify_entropy_shell = MagicMock(
            return_value='stable')
        engine.math_core.classify_phase_shell = MagicMock()
        engine.math_core.classify_phase_shell.return_value.shell_type = 'drift_positive'
        engine.math_core.classify_phase_shell.return_value.stability = 0.8
        engine.math_core.compute_matrix_stability = MagicMock(
            return_value='matrix_safe')

        # Simulate flash request event
        event = BusEvent(
            type='flash.request',
            data={
                'z_score': 1.0,
                'phase_angle': np.pi,
                'context': {'high_volatility': False},
                'request_id': 'test_123'
            },
            timestamp=time.time()
        )

        engine._handle_flash_request(event)

        # Should have dispatched a flash.result event
        self.mock_bus.dispatch_event.assert_called()

        # Check that the result event contains the request_id
        call_args = self.mock_bus.dispatch_event.call_args
        result_event = call_args[0][0]
        self.assertEqual(result_event.type, 'flash.result')
        self.assertEqual(result_event.data['request_id'], 'test_123')

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_quantum_stats_calculation(self, mock_load_config):
        """Test quantum statistics calculation"""
        mock_load_config.return_value = self.mock_config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Add some test flash states
        test_states = [
            FlashState(
                timestamp=time.time(),
                z_score=1.0,
                phase_angle=np.pi,
                entropy_class='stable',
                matrix_state='matrix_safe',
                is_safe=True,
                risk_entropy=0.3,
                fractal_depth=1,
                event_id='test1',
                coherence_score=0.8
            ),
            FlashState(
                timestamp=time.time(),
                z_score=2.0,
                phase_angle=np.pi + 0.1,
                entropy_class='unstable',
                matrix_state='matrix_unsafe',
                is_safe=False,
                risk_entropy=0.7,
                fractal_depth=2,
                event_id='test2',
                coherence_score=0.6
            )
        ]
        engine.flash_history.extend(test_states)

        stats = engine.get_quantum_stats()

        self.assertIn('entropy', stats)
        self.assertIn('phase', stats)
        self.assertIn('risk', stats)
        self.assertIn('fractal', stats)
        self.assertIn('coherence', stats)
        self.assertIn('safety_rate', stats)

        # Check specific calculations
        self.assertEqual(stats['safety_rate'], 0.5)  # 1 safe out of 2 total
        self.assertEqual(stats['total_flashes'], 2)
        self.assertEqual(stats['fractal']['max_depth'], 2)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_state_persistence(self, mock_load_config):
        """Test state saving and loading"""
        mock_load_config.return_value = self.mock_config

        # Create engine with temporary state file
        engine = GPUFlashEngine(bus_core=self.mock_bus)
        engine.state_file = Path(self.temp_dir) / "test_state.json"

        # Add some test data
        engine.flash_history.append(FlashState(
            timestamp=time.time(),
            z_score=1.0,
            phase_angle=np.pi,
            entropy_class='stable',
            matrix_state='matrix_safe',
            is_safe=True,
            risk_entropy=0.3
        ))
        engine.phase_memory = [np.pi, np.pi + 0.1]
        engine.entropy_cascade = [1.0, 1.5]
        engine.fractal_depth = 2

        # Save state
        engine._save_state()
        self.assertTrue(engine.state_file.exists())

        # Create new engine and load state
        engine2 = GPUFlashEngine(bus_core=self.mock_bus)
        engine2.state_file = engine.state_file
        engine2._load_state()

        # Verify data was restored
        self.assertEqual(len(engine2.flash_history), 1)
        self.assertEqual(engine2.phase_memory, [np.pi, np.pi + 0.1])
        self.assertEqual(engine2.entropy_cascade, [1.0, 1.5])
        self.assertEqual(engine2.fractal_depth, 2)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_memory_trimming(self, mock_load_config):
        """Test that memory is properly trimmed to avoid unbounded growth"""
        config = self.mock_config.copy()
        config['max_cascade_memory'] = 5
        config['max_history_size'] = 3
        mock_load_config.return_value = config

        engine = GPUFlashEngine(bus_core=self.mock_bus)

        # Fill entropy cascade beyond limit
        for i in range(10):
            engine.entropy_cascade.append(float(i))

        # Trigger trim
        event = BusEvent(
            type='entropy.update',
            data={'z_score': 999},
            timestamp=time.time()
        )
        engine._handle_entropy_update(event)

        # Should be trimmed to max_cascade_memory
        self.assertEqual(len(engine.entropy_cascade), 5)
        self.assertEqual(
            engine.entropy_cascade[-1],
            999
        )  # Latest value preserved

        # Test flash history trimming
        for i in range(10):
            engine.flash_history.append(FlashState(
                timestamp=time.time(),
                z_score=float(i),
                phase_angle=np.pi,
                entropy_class='stable',
                matrix_state='matrix_safe',
                is_safe=True,
                risk_entropy=0.3
            ))

        engine._trim_history()
        self.assertEqual(len(engine.flash_history), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for GPU Flash Engine with real bus core"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil  # noqa: F401
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('core.gpu_flash_engine.load_yaml_config')
    def test_full_event_flow(self, mock_load_config):
        """Test complete event flow from entropy update to flash decision"""
        mock_config = {
            'cooldown_period': 0.1,
            'binding_energy_default': 7.5,
            'risk_thresholds': {'critical': 0.9, 'high': 0.7},
            'max_cascade_memory': 100,
            'max_history_size': 1000
        }
        mock_load_config.return_value = mock_config

        # Create real bus core and engine
        bus_core = BusCore()
        engine = GPUFlashEngine(bus_core=bus_core)
        engine.state_file = Path(self.temp_dir) / "integration_state.json"

        # Mock math core for predictable results
        engine.math_core.classify_entropy_shell = MagicMock(
            return_value='stable')
        engine.math_core.classify_phase_shell = MagicMock()
        engine.math_core.classify_phase_shell.return_value.shell_type = 'drift_positive'
        engine.math_core.classify_phase_shell.return_value.stability = 0.8
        engine.math_core.compute_matrix_stability = MagicMock(
            return_value='matrix_safe')

        # Track events for verification
        events_received = []

        def event_tracker(event):
            events_received.append(event.type)

        bus_core.register_handler('flash.executed', event_tracker)
        bus_core.register_handler('phase.resonance', event_tracker)

        # Send entropy update
        bus_core.dispatch_event(BusEvent(
            type='entropy.update',
            data={'z_score': 1.5},
            timestamp=time.time()
        ))

        # Send phase drift updates (to trigger resonance)
        for phase in [np.pi, np.pi + 0.001, np.pi + 0.002]:
            bus_core.dispatch_event(BusEvent(
                type='phase.drift',
                data={'phase_angle': phase},
                timestamp=time.time()
            ))

        # Send flash request
        bus_core.dispatch_event(BusEvent(
            type='flash.request',
            data={
                'z_score': 1.0,
                'phase_angle': np.pi,
                'request_id': 'integration_test'
            },
            timestamp=time.time()
        ))

        # Verify entropy and phase data was captured
        self.assertGreater(len(engine.entropy_cascade), 0)
        self.assertGreater(len(engine.phase_memory), 0)

        # Verify flash was executed
        self.assertGreater(len(engine.flash_history), 0)

        # Verify phase resonance was detected
        self.assertIn('phase.resonance', events_received)


if __name__ == '__main__':
    # Set up logging for tests
    import logging  # noqa: F401
    logging.basicConfig(level=logging.DEBUG)

    # Run the tests
    unittest.main(verbosity=2)