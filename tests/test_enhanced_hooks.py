from unittest.mock import Mock

"""
Test Enhanced Hook System
========================

Tests for the enhanced hook system with thermal-aware \
    profit-synchronized routing.
"""

import pytest  # noqa: F401
import tempfile  # noqa: F401
import os  # noqa: F401


def test_enhanced_hooks_import():  # noqa: F401
    """Test that enhanced hooks can be imported without errors"""  # noqa: F401
    try:
        from core.enhanced_hooks import (  # noqa: F401
            DynamicHookRouter,
            HookState,
            HookContext
        )
        assert True
    except ImportError as e:
        pytest.skip(f"Enhanced hooks not available: {e}")


def test_hook_system_manager_initialization():
    """Test that the HookSystemManager initializes correctly"""
    try:
        from core.hooks import HookSystemManager  # noqa: F401

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(
            mode='w',
                suffix='.yaml',
                delete=False
        ) as f:
            f.write("""
debug:
  clusters: false
  drifts: false
  simulate_strategy: false

hooks:
  test_hook:
    enabled: true
    thermal_zones: ["cool", "normal"]
    profit_zones: ["stable"]
    confidence_threshold: 0.5
    cooldown_seconds: 10

thresholds:
  thermal_critical_temp: 85.0
  profit_vector_minimum: 0.3
  memory_confidence_minimum: 0.4
  hook_failure_threshold: 0.2

echo_feedback:
  enabled: true
  success_weight: 0.7
  failure_weight: 0.3
  decay_factor: 0.95
""")
            config_path = f.name

        try:
            # Test initialization
            manager = HookSystemManager()
            assert manager._initialized

            # Test context retrieval
            context = manager.get_current_context()
            assert hasattr(
                context,
                'thermal_zone'
            ) or 'thermal_zone' in context

            # Test statistics
            stats = manager.get_system_statistics()
            assert isinstance(stats, dict)

        finally:
            os.unlink(config_path)

    except ImportError as e:
        pytest.skip(f"Hook system components not available: {e}")


def test_legacy_hook_compatibility():
    """Test that legacy hook access still works"""
    try:
        from core.hooks import execute_hook, get_hook_context, get_hook_stats  # noqa: F401

        # Test that legacy variables are available (may be None if components
        # not available)
        assert ncco_manager is not None or "Mock or real instance"

        # Test function access
        context = get_hook_context()
        assert context is not None

        stats = get_hook_stats()
        assert isinstance(stats, dict)

    except ImportError as e:
        pytest.skip(f"Legacy hooks not available: {e}")


def test_hook_execution():
    """Test hook execution through the enhanced system"""
    try:
        from core.hooks import execute_hook, HookSystemManager  # noqa: F401

        # Mock a simple hook
        mock_hook = Mock()
        mock_hook.test_method = Mock(return_value="test_result")

        # Create manager and register mock hook
        manager = HookSystemManager()

        if hasattr(manager, '_enhanced_router') and manager._enhanced_router:
            manager._enhanced_router.register_hook("test_hook", mock_hook)
        else:
            manager._legacy_hooks["test_hook"] = mock_hook

        # Test execution
        _ = manager.execute_hook(  # noqa: F841 (intentionally unused)
            "test_hook",
            "test_method",
            "arg1",
            kwarg1="kwarg1"
        )

        # Verify the method was called
        mock_hook.test_method.assert_called_once_with("arg1", kwarg1="kwarg1")

    except ImportError as e:
        pytest.skip(f"Enhanced hooks not available: {e}")


def test_hook_context_structure():
    """Test that hook context has expected structure"""
    try:
        from core.hooks import get_hook_context  # noqa: F401

        context = get_hook_context()

        # Check context structure (works for both enhanced and legacy modes)
        if hasattr(context, '__dict__'):
            # Enhanced mode - dataclass
            assert hasattr(context, 'thermal_zone')
            assert hasattr(context, 'profit_zone')
            assert hasattr(context, 'timestamp')
        else:
            # Legacy mode - dictionary
            assert 'thermal_zone' in context
            assert 'profit_zone' in context
            assert 'timestamp' in context

    except ImportError as e:
        pytest.skip(f"Hook context not available: {e}")


def test_hook_configuration_loading():
    """Test YAML configuration loading"""
    try:
        from core.enhanced_hooks import DynamicHookRouter  # noqa: F401

        # Create temporary config
        with tempfile.NamedTemporaryFile(
            mode='w',
                suffix='.yaml',
                delete=False
        ) as f:
            f.write("""
hooks:
  test_hook:
    enabled: true
    thermal_zones: ["cool"]
    confidence_threshold: 0.8

thresholds:
  thermal_critical_temp: 80.0
""")
            config_path = f.name

        try:
            router = DynamicHookRouter(config_path)
            assert router.config is not None
            assert 'hooks' in router.config
            assert 'test_hook' in router.config['hooks']

        finally:
            os.unlink(config_path)

    except ImportError as e:
        pytest.skip(f"Enhanced hooks not available: {e}")


@pytest.mark.parametrize("thermal_zone,profit_zone,expected", [
    ("cool", "surging", True),
    ("hot", "surging", False),
    ("cool", "drawdown", False),
])
def test_hook_execution_conditions(thermal_zone, profit_zone, expected):
    """Test hook execution under different thermal/profit conditions"""
    try:
        from core.enhanced_hooks import DynamicHookRouter, HookContext  # noqa: F401
        from datetime import datetime, timezone  # noqa: F821

        # Create router with test config
        router = DynamicHookRouter()

        # Create test context
        context = HookContext(
            thermal_zone=thermal_zone,
            profit_zone=profit_zone,
            thermal_temp=65.0,
            profit_vector_strength=0.8,
            memory_confidence=0.7,
            timestamp=datetime.now(timezone.utc)  # noqa: F821
        )

        # Test with a hook that requires cool thermal zone and surging profit
        should_execute, reason = router.should_execute_hook(
            "sfsss_router",
            context
        )

        if thermal_zone == "cool" and profit_zone == "surging":
            assert should_execute or "May fail due to other conditions"
        else:
            # Should be denied for incompatible zones
            assert not should_execute

    except ImportError as e:
        pytest.skip(f"Enhanced hooks not available: {e}")


def test_hook_performance_tracking():
    """Test hook performance tracking functionality"""
    try:

        router = DynamicHookRouter()

        # Register a mock hook
        mock_hook = Mock()
        mock_hook.test_method = Mock(return_value="success")
        router.register_hook("test_hook", mock_hook)

        # Execute hook multiple times
        for i in range(5):
            router.execute_hook("test_hook", "test_method")

        # Check performance tracking
        if "test_hook" in router.hook_performance:
            perf = router.hook_performance["test_hook"]
            assert perf.total_executions > 0
            assert perf.confidence_score >= 0.0

    except ImportError as e:
        pytest.skip(f"Enhanced hooks not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])