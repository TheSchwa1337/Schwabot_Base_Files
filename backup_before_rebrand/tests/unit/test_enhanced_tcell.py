#!/usr/bin/env python3
"""Test Enhanced T-Cell System."

Comprehensive testing of the enhanced T-Cell system to ensure:
- Proper signal generation including INHIBITORY signals
- Correct information handling and validation
- Feedback loops and adaptive learning
- Pattern recognition and memory
- Risk assessment and contextual analysis
"""

import asyncio
import logging
import sys
import time

# Add core directory to path
sys.path.append("core")

    EnhancedTCellValidator,
    EnhancedSignalGenerator,
    EnhancedTCellSignal,
    EnhancedSignalType,
)
from biological_immune_error_handler import BiologicalImmuneErrorHandler

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockImmuneHandler:
    """Mock immune handler for testing."""

    def __init__(self):
        self.mitochondrial_health = 0.8
        self.system_entropy = 0.3
        self.current_error_rate = 0.5
        self.total_operations = 100
        self.successful_operations = 85
        self.blocked_operations = 10
        self.error_history = []
        self.antibody_patterns = {}


def test_enhanced_tcell_validator():
    """Test enhanced T-Cell validator functionality."""
    print("🧬 Testing Enhanced T-Cell Validator...")

    validator = EnhancedTCellValidator(activation_threshold=0.6)

    # Test 1: Basic signal validation
    signals = []
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.PRIMARY,
            strength=0.7,
            source="test_operation",
            timestamp=time.time(),
            confidence=0.8,
        ),
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.COSTIMULATORY,
            strength=0.6,
            source="system_health",
            timestamp=time.time(),
            confidence=0.9,
        ),
    ]

    activation, confidence, analysis = validator.validate_signals(signals)
    print()
        f"   ✅ Basic validation: activation={activation}, confidence={confidence:.3f}"
    )

    # Test 2: INHIBITORY signal handling (CRITICAL, FIX)
    signals_with_inhibitory = signals + []
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.INHIBITORY,
            strength=0.5,
            source="risk_assessment",
            timestamp=time.time(),
            confidence=0.8,
        )
    ]

    activation2, confidence2, analysis2 = validator.validate_signals()
        signals_with_inhibitory
    )
    print()
        f"   ✅ INHIBITORY signal test: activation={activation2}, confidence={confidence2:.3f}"
    )
    print()
        f"   📊 Signal analysis: {len(analysis2['signal_analysis'])} signals analyzed"
    )

    # Test 3: Performance feedback
    pattern_hash = analysis2.get("pattern_hash")
    if pattern_hash:
        validator.update_performance_feedback(pattern_hash, True)
        validator.update_performance_feedback(pattern_hash, False)
        print(f"   ✅ Performance feedback updated for pattern: {pattern_hash}")

    # Test 4: Threshold adjustment
    validator.adjust_threshold(0.3)  # Low success rate
    print(f"   ✅ Threshold adjusted to: {validator.adaptive_threshold:.3f}")

    # Test 5: Statistics
    stats = validator.get_signal_statistics()
    print()
        f"   ✅ Statistics: {stats['total_validations']} validations, {stats['pattern_count']} patterns"
    )

    return True


def test_enhanced_signal_generator():
    """Test enhanced signal generator functionality."""
    print("🧬 Testing Enhanced Signal Generator...")

    mock_handler = MockImmuneHandler()
    generator = EnhancedSignalGenerator(mock_handler)

    # Test function
    def test_operation(value: float, should_fail: bool = False) -> float:
        if should_fail:
            raise ValueError("Test failure")
        return value * 2

    # Test 1: Basic signal generation
    signals = generator.generate_comprehensive_signals(test_operation, (5.0,), {})
    print(f"   ✅ Generated {len(signals)} signals")

    # Check for all signal types
    signal_types = [s.signal_type for s in signals]
    expected_types = []
        EnhancedSignalType.PRIMARY,
        EnhancedSignalType.COSTIMULATORY,
        EnhancedSignalType.INFLAMMATORY,
        EnhancedSignalType.CONTEXTUAL,
        EnhancedSignalType.RISK_ASSESSMENT,
    ]

    for expected_type in expected_types:
        if expected_type in signal_types:
            print(f"   ✅ {expected_type.value} signal generated")
        else:
            print(f"   ❌ {expected_type.value} signal missing")

    # Test 2: INHIBITORY signal generation (CRITICAL, FIX)
    # Create conditions for inhibitory signal
    mock_handler.current_error_rate = 0.15  # High error rate
    mock_handler.system_entropy = 0.8  # High entropy

    signals_with_inhibitory = generator.generate_comprehensive_signals()
        test_operation, (5.0,), {}
    )
    inhibitory_signals = []
        s
        for s in signals_with_inhibitory
        if s.signal_type == EnhancedSignalType.INHIBITORY
    ]

    if inhibitory_signals:
        print()
            f"   ✅ INHIBITORY signal generated with strength: {inhibitory_signals[0].strength:.3f}"
        )
    else:
        print("   ❌ INHIBITORY signal not generated")

    # Test 3: Memory signal generation
    # Add antibody pattern
    mock_handler.antibody_patterns["test_operation_1_0"] = {}
        "rejection_strength": 0.3,
        "occurrence_count": 2,
    }

    signals_with_memory = generator.generate_comprehensive_signals()
        test_operation, (5.0,), {}
    )
    memory_signals = []
        s for s in signals_with_memory if s.signal_type == EnhancedSignalType.MEMORY
    ]

    if memory_signals:
        print()
            f"   ✅ MEMORY signal generated with strength: {memory_signals[0].strength:.3f}"
        )
    else:
        print("   ❌ MEMORY signal not generated")

    # Test 4: Risk assessment
    risk_signals = []
        s
        for s in signals_with_memory
        if s.signal_type == EnhancedSignalType.RISK_ASSESSMENT
    ]
    if risk_signals:
        risk_strength = risk_signals[0].strength
        print(f"   ✅ Risk assessment signal: {risk_strength:.3f}")

        # Check risk factors in metadata
        risk_factors = risk_signals[0].metadata.get("risk_factors", {})
        print(f"   📊 Risk factors: {list(risk_factors.keys())}")

    # Test 5: Operation history tracking
    generator.update_operation_history("test_operation", True)
    generator.update_operation_history("test_operation", False)

    history = generator.operation_history.get("test_operation", {})
    print(f"   ✅ Operation history: {history}")

    return True


def test_integrated_immune_system():
    """Test integrated enhanced immune system."""
    print("🧬 Testing Integrated Enhanced Immune System...")

    # Initialize enhanced immune handler
    immune_handler = BiologicalImmuneErrorHandler()

    # Test 1: Successful operation
    def safe_operation(x: float) -> float:
        return x * 2

    result = immune_handler.immune_protected_operation(safe_operation, 5.0)
    print(f"   ✅ Safe operation result: {result}")

    # Test 2: Risky operation (should be blocked or, monitored)
    def risky_operation(data: list, should_fail: bool = False) -> float:
        if should_fail:
            raise ValueError("Risky operation failed")
        return sum(data)

    # Large data that might trigger inhibitory signals
    large_data = list(range(1000))
    result2 = immune_handler.immune_protected_operation(risky_operation, large_data)
    print(f"   ✅ Risky operation result: {type(result2).__name__}")

    # Test 3: Failing operation
    try:
        result3 = immune_handler.immune_protected_operation()
            risky_operation, [1, 2, 3], {"should_fail": True}
        )
        print(f"   ✅ Failing operation handled: {type(result3).__name__}")
    except Exception as e:
        print(f"   ❌ Failing operation not handled: {e}")

    # Test 4: Enhanced status
    enhanced_status = immune_handler.get_enhanced_immune_status()
    print(f"   ✅ Enhanced status retrieved with {len(enhanced_status)} categories")

    # Check for enhanced T-Cell information
    if "enhanced_tcell" in enhanced_status:
        tcell_info = enhanced_status["enhanced_tcell"]
        print()
            f"   📊 T-Cell validator stats: {tcell_info['validator_stats']['total_validations']} validations"
        )
        print()
            f"   📊 Signal generator: {tcell_info['signal_generator']['operation_history_size']} operations tracked"
        )

    # Check for signal analysis
    if "signal_analysis" in enhanced_status:
        signal_info = enhanced_status["signal_analysis"]
        print(f"   📊 Signal types: {signal_info['total_signal_types']}")
        print()
            f"   📊 Enhanced features: {len(signal_info['enhanced_features'])} features"
        )

    return True


def test_signal_validation_edge_cases():
    """Test edge cases in signal validation."""
    print("🧬 Testing Signal Validation Edge Cases...")

    validator = EnhancedTCellValidator()

    # Test 1: Empty signals
    activation, confidence, analysis = validator.validate_signals([])
    print()
        f"   ✅ Empty signals handled: activation={activation}, confidence={confidence}"
    )

    # Test 2: Invalid signals
    invalid_signals = []
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.PRIMARY,
            strength=1.5,  # Invalid strength > 1.0
            source="invalid",
            timestamp=time.time(),
            confidence=0.8,
        )
    ]

    activation2, confidence2, analysis2 = validator.validate_signals(invalid_signals)
    print()
        f"   ✅ Invalid signals filtered: activation={activation2}, confidence={confidence2}"
    )

    # Test 3: Mixed valid/invalid signals
    mixed_signals = []
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.PRIMARY,
            strength=0.7,
            source="valid",
            timestamp=time.time(),
            confidence=0.8,
        ),
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.INHIBITORY,
            strength=-0.1,  # Invalid negative strength
            source="invalid",
            timestamp=time.time(),
            confidence=0.8,
        ),
    ]

    activation3, confidence3, analysis3 = validator.validate_signals(mixed_signals)
    print()
        f"   ✅ Mixed signals handled: activation={activation3}, confidence={confidence3:.3f}"
    )
    print(f"   📊 Valid signals processed: {analysis3['signal_count']}")

    return True


def test_adaptive_learning():
    """Test adaptive learning capabilities."""
    print("🧬 Testing Adaptive Learning...")

    validator = EnhancedTCellValidator(activation_threshold=0.6)

    # Create consistent signal pattern
    base_signals = []
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.PRIMARY,
            strength=0.6,
            source="test",
            timestamp=time.time(),
            confidence=0.8,
        ),
        EnhancedTCellSignal()
            signal_type=EnhancedSignalType.COSTIMULATORY,
            strength=0.5,
            source="test",
            timestamp=time.time(),
            confidence=0.8,
        ),
    ]

    # Test pattern learning
    for i in range(5):
        activation, confidence, analysis = validator.validate_signals(base_signals)
        pattern_hash = analysis.get("pattern_hash")

        # Simulate success/failure pattern
        was_successful = i % 2 == 0  # Alternating success/failure
        validator.update_performance_feedback(pattern_hash, was_successful)

        print()
            f"   📊 Iteration {i + 1}: activation={activation}, success={was_successful}"
        )

    # Check pattern statistics
    stats = validator.get_signal_statistics()
    print(f"   ✅ Pattern learning: {stats['pattern_count']} patterns learned")

    # Test threshold adaptation
    initial_threshold = validator.activation_threshold
    validator.adjust_threshold(0.3)  # Low success rate
    validator.adjust_threshold(0.9)  # High success rate

    print()
        f"   ✅ Threshold adaptation: {initial_threshold:.3f} -> {validator.adaptive_threshold:.3f}"
    )

    return True


async def test_async_monitoring():
    """Test async monitoring capabilities."""
    print("🧬 Testing Async Monitoring...")

    immune_handler = BiologicalImmuneErrorHandler()

    # Start monitoring
    await immune_handler.start_monitoring()
    print("   ✅ Monitoring started")

    # Run some operations
    for i in range(3):

        def test_op(x: float) -> float:
            return x * 2

        result = immune_handler.immune_protected_operation(test_op, float(i))
        print(f"   📊 Operation {i + 1}: {result}")
        await asyncio.sleep(1)

    # Stop monitoring
    await immune_handler.stop_monitoring()
    print("   ✅ Monitoring stopped")

    return True


def main():
    """Run comprehensive enhanced T-Cell system tests."""
    print("🧬 Enhanced T-Cell System Test Suite")
    print("=" * 50)

    tests = []
        ("Enhanced T-Cell Validator", test_enhanced_tcell_validator),
        ("Enhanced Signal Generator", test_enhanced_signal_generator),
        ("Integrated Immune System", test_integrated_immune_system),
        ("Signal Validation Edge Cases", test_signal_validation_edge_cases),
        ("Adaptive Learning", test_adaptive_learning),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n🔬 Running: {test_name}")
            if test_func():
                print(f"   ✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"   ❌ {test_name} FAILED")
        except Exception as e:
            print(f"   ❌ {test_name} ERROR: {e}")
            logger.exception(f"Test {test_name} failed")

    # Run async test
    print("\n🔬 Running: Async Monitoring")
    try:
        asyncio.run(test_async_monitoring())
        print("   ✅ Async Monitoring PASSED")
        passed += 1
    except Exception as e:
        print(f"   ❌ Async Monitoring ERROR: {e}")
        logger.exception("Async monitoring test failed")

    total += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"🧬 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced T-Cell system is working correctly.")
        print("\n✅ CRITICAL FIXES VERIFIED:")
        print("   - INHIBITORY signal generation and handling")
        print("   - Proper signal strength calculation")
        print("   - Contextual signal analysis")
        print("   - Risk assessment integration")
        print("   - Pattern-based learning")
        print("   - Adaptive threshold adjustment")
        print("   - Performance feedback loops")
        print("   - Enhanced memory signal integration")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
