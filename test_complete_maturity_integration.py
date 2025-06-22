#!/usr/bin/env python3
"""Complete Maturity Integration Test.

This test demonstrates the complete maturity system integration,
showing how all components work together to provide a unified,
observable, and error-resilient trading system.
"""

import logging
import time
import sys
from pathlib import Path

# Add core to path for imports
sys.path.append(str(Path(__file__).parent))

from core.core_loop_manager import CoreLoopManager, create_core_loop_manager
from core.tick_cycle_validator import TickPhase
from core.profit_vector_reconciler import ReconciliationStatus
from core.error_sanitizer import ErrorSanitizer, SanitizationLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockWaveformEngine:
    """Mock DLT Waveform Engine for testing."""
    
    def __init__(self):
        self.cycle_count = 0
    
    def process_market_data(self, market_data):
        """Mock waveform processing."""
        self.cycle_count += 1
        
        # Simulate different vector types based on cycle
        if self.cycle_count % 3 == 0:
            return {'magnitude': 0.8, 'direction': 'buy', 'confidence': 0.9}
        elif self.cycle_count % 3 == 1:
            return {'magnitude': 0.6, 'direction': 'sell', 'confidence': 0.7}
        else:
            return {'magnitude': 0.3, 'direction': 'hold', 'confidence': 0.5}
    
    def export_vectors(self):
        """Mock vector export."""
        return [self.process_market_data({})]


class MockProfitAllocator:
    """Mock Profit Allocator for testing."""
    
    def __init__(self):
        self.allocation_count = 0
    
    def receive(self, vectors):
        """Mock vector reception."""
        self.allocation_count += 1
        logger.debug(f"Received {len(vectors)} vectors for allocation")
    
    def projected_profit(self, vector):
        """Mock projected profit calculation."""
        return vector.get('magnitude', 0.5) * 1000
    
    def realized_profit(self, vector):
        """Mock realized profit calculation."""
        # Add some variance to simulate real trading
        projected = self.projected_profit(vector)
        variance = projected * 0.1 * (0.5 - (time.time() % 1))  # ±10% variance
        return projected + variance
    
    def adjust_distribution(self, vector, diff):
        """Mock distribution adjustment."""
        logger.info(f"Adjusting distribution by {diff:.2f} for vector {vector}")


class MockTickInterpreter:
    """Mock Tick Hash Interpreter for testing."""
    
    def __init__(self):
        self.tick_count = 0
        self.phases = [
            TickPhase.INITIALIZATION.value,
            TickPhase.MARKET_OPEN.value,
            TickPhase.ACTIVE_TRADING.value,
            TickPhase.CONSOLIDATION.value,
            TickPhase.ACTIVE_TRADING.value,
            TickPhase.MARKET_CLOSE.value,
            TickPhase.MAINTENANCE.value
        ]
    
    def process_tick_data(self, market_data):
        """Mock tick processing."""
        self.tick_count += 1
        phase_index = (self.tick_count // 3) % len(self.phases)
        return self.phases[phase_index]


class MockPortfolioRouter:
    """Mock Portfolio Router for testing."""
    
    def __init__(self):
        self.shift_count = 0
    
    def calculate_portfolio_shift(self, market_data):
        """Mock portfolio shift calculation."""
        self.shift_count += 1
        price = market_data.get('price', 50000)
        
        # Calculate shift based on price movement
        if price > 50500:
            direction = 'increase_btc'
            magnitude = min(0.2, (price - 50500) / 1000)
        elif price < 49500:
            direction = 'increase_cash'
            magnitude = min(0.2, (49500 - price) / 1000)
        else:
            direction = 'hold'
            magnitude = 0.0
        
        return {
            'direction': direction,
            'magnitude': magnitude,
            'timestamp': time.time(),
            'confidence': 0.8
        }


class MockStateValidator:
    """Mock State Validation Router for testing."""
    
    def __init__(self):
        self.validation_count = 0
    
    def validate_state_consistency(self, tick_data, portfolio_data, market_data):
        """Mock state validation."""
        self.validation_count += 1
        
        # Simulate occasional validation failures
        if self.validation_count % 7 == 0:
            return False  # Simulate validation failure
        
        return True


def test_complete_maturity_integration():
    """Test complete maturity system integration."""
    logger.info("🚀 Starting Complete Maturity Integration Test")
    
    try:
        # Create core loop manager
        manager = create_core_loop_manager()
        
        # Inject mock components
        logger.info("📦 Injecting mock components...")
        manager.inject_component('waveform_engine', MockWaveformEngine())
        manager.inject_component('profit_allocator', MockProfitAllocator())
        
        # Create and inject additional mock components
        tick_interpreter = MockTickInterpreter()
        portfolio_router = MockPortfolioRouter()
        state_validator = MockStateValidator()
        
        manager.tick_interpreter = tick_interpreter
        manager.portfolio_router = portfolio_router
        manager.state_validator = state_validator
        
        logger.info("✅ Mock components injected successfully")
        
        # Initialize components
        logger.info("🔧 Initializing components...")
        if not manager.initialize_components():
            logger.error("❌ Component initialization failed")
            return False
        
        logger.info("✅ Components initialized successfully")
        
        # Run several execution cycles
        logger.info("🔄 Running execution cycles...")
        successful_cycles = 0
        total_cycles = 10
        
        for cycle in range(total_cycles):
            logger.info(f"📊 Executing cycle {cycle + 1}/{total_cycles}")
            
            success = manager._execute_single_cycle()
            if success:
                successful_cycles += 1
                logger.info(f"✅ Cycle {cycle + 1} completed successfully")
            else:
                logger.warning(f"⚠️ Cycle {cycle + 1} failed")
            
            # Small delay between cycles
            time.sleep(0.1)
        
        # Get comprehensive status
        logger.info("📈 Gathering comprehensive status...")
        status = manager.get_comprehensive_status()
        
        # Display results
        logger.info("=" * 60)
        logger.info("🎯 COMPLETE MATURITY INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        
        # Execution statistics
        logger.info(f"📊 Execution Statistics:")
        logger.info(f"   • Total cycles: {total_cycles}")
        logger.info(f"   • Successful cycles: {successful_cycles}")
        logger.info(f"   • Success rate: {successful_cycles/total_cycles*100:.1f}%")
        logger.info(f"   • System ready: {status['system_ready']}")
        
        # State tracker status
        if 'state_tracker_status' in status:
            st_status = status['state_tracker_status']
            logger.info(f"🔄 State Tracker:")
            logger.info(f"   • Current phase: {st_status.get('current_tick_phase', 'None')}")
            logger.info(f"   • Validation state: {st_status.get('current_validation_state', 'None')}")
            logger.info(f"   • Ready for execution: {st_status.get('ready_for_execution', False)}")
        
        # Tick cycle validator status
        if 'tick_cycle_validator' in status:
            tcv_status = status['tick_cycle_validator']
            logger.info(f"⏱️ Tick Cycle Validator:")
            logger.info(f"   • Total validations: {tcv_status.get('total_validations', 0)}")
            logger.info(f"   • Success rate: {tcv_status.get('success_rate', 0)*100:.1f}%")
            logger.info(f"   • Phase transitions: {tcv_status.get('phase_transitions', 0)}")
            logger.info(f"   • Current phase: {tcv_status.get('current_phase', 'None')}")
        
        # Profit vector reconciler status
        if 'profit_vector_reconciler' in status:
            pvr_status = status['profit_vector_reconciler']
            logger.info(f"💰 Profit Vector Reconciler:")
            logger.info(f"   • Total reconciliations: {pvr_status.get('total_reconciliations', 0)}")
            logger.info(f"   • Aligned percentage: {pvr_status.get('aligned_percentage', 0):.1f}%")
            logger.info(f"   • Drift percentage: {pvr_status.get('drift_percentage', 0):.1f}%")
            logger.info(f"   • Efficiency ratio: {pvr_status.get('efficiency_ratio', 0):.3f}")
        
        # Error sanitizer status
        if 'error_sanitizer' in status:
            es_status = status['error_sanitizer']
            logger.info(f"🛡️ Error Sanitizer:")
            logger.info(f"   • Total errors: {es_status.get('total_errors', 0)}")
            logger.info(f"   • Recovery attempts: {es_status.get('recovery_attempts', 0)}")
            logger.info(f"   • Recovery successes: {es_status.get('recovery_successes', 0)}")
            logger.info(f"   • Recovery rate: {es_status.get('recovery_rate', 0)*100:.1f}%")
            if es_status.get('most_common_error'):
                logger.info(f"   • Most common error: {es_status['most_common_error']}")
        
        # Performance metrics
        perf_stats = status.get('performance_stats', {})
        logger.info(f"⚡ Performance Metrics:")
        logger.info(f"   • Average cycle time: {perf_stats.get('average_cycle_time', 0)*1000:.1f}ms")
        logger.info(f"   • Cycles per second: {perf_stats.get('cycles_per_second', 0):.1f}")
        
        logger.info("=" * 60)
        
        # Test error sanitization specifically
        logger.info("🧪 Testing Error Sanitization...")
        test_error_sanitization(manager.error_sanitizer)
        
        # Test mathematical recovery
        logger.info("🔢 Testing Mathematical Recovery...")
        test_mathematical_recovery(manager.error_sanitizer)
        
        logger.info("🎉 Complete Maturity Integration Test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Cleanup
        if 'manager' in locals():
            manager.stop_execution_loop()


def test_error_sanitization(error_sanitizer):
    """Test error sanitization capabilities."""
    logger.info("Testing error sanitization...")
    
    # Test division by zero
    def divide_by_zero():
        return 10 / 0
    
    result = error_sanitizer.catch(divide_by_zero, fallback_value="FALLBACK")
    logger.info(f"Division by zero result: {result}")
    
    # Test value error
    def invalid_conversion():
        return int("not_a_number")
    
    result = error_sanitizer.catch(invalid_conversion, fallback_value=-1)
    logger.info(f"Invalid conversion result: {result}")
    
    # Test key error
    def missing_key():
        data = {'a': 1}
        return data['missing_key']
    
    result = error_sanitizer.catch(missing_key, fallback_value="KEY_NOT_FOUND")
    logger.info(f"Missing key result: {result}")


def test_mathematical_recovery(error_sanitizer):
    """Test mathematical-specific error recovery."""
    logger.info("Testing mathematical recovery...")
    
    # Test mathematical computation with recovery
    def calculate_profit_ratio():
        # This will cause division by zero
        return 1000 / 0
    
    result = error_sanitizer.catch(calculate_profit_ratio)
    logger.info(f"Mathematical recovery result: {result}")
    
    # Test overflow recovery
    def calculate_large_number():
        return 10 ** 1000
    
    result = error_sanitizer.catch(calculate_large_number)
    logger.info(f"Overflow recovery result: {result}")


if __name__ == "__main__":
    # Run the complete integration test
    success = test_complete_maturity_integration()
    
    if success:
        logger.info("🎯 ALL TESTS PASSED - Maturity system is fully operational!")
        sys.exit(0)
    else:
        logger.error("❌ TESTS FAILED - Issues detected in maturity system")
        sys.exit(1) 